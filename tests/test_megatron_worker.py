"""Megatron backend pieces that do not need megatron installed.

Neither megatron-core nor megatron-bridge is a dependency, so most of the worker
cannot run here. The parts that can are the ones that would fail silently: the
data-parallel sharding hooks (wrong shard -> tensor-parallel peers train on
different tokens), the DDP buffer offload's pointer arithmetic (wrong offset ->
parameters alias the wrong weights after a wake_up), and the chunked logprob
head's sequence-first/batch-first bookkeeping (wrong transpose -> training on
shifted labels).

The checkpoint cases are here for a different reason. They do not fail silently
at all -- they kill the run -- but they only fire at a save boundary, so on a
long run the failure arrives hours after the commit that caused it and takes the
training with it. A stub bridge runs the same path in milliseconds.

The round-trip cases are the other half of that: the fixes reached for after
such a crash mostly work by making the save stop complaining, which converts a
dead run into a checkpoint nothing can be resumed from.
"""

import importlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import unittest.mock

import torch
from safetensors.torch import load_file, save_file

import training.megatron_worker as megatron_worker
from training.megatron_worker import OPTIMIZER_SUBDIR, MegatronTrainingWorker, chunked_target_logprobs
from training.models import gemma4
from training.trainer_worker import BaseTrainerWorker, Datum


def datum(token: int) -> Datum:
  return Datum(
    model_input=[token, token],
    loss_fn_inputs={"target_tokens": {"data": [token, token]}, "weights": {"data": [1.0, 1.0]}},
  )


class RecordingWorker(BaseTrainerWorker):
  """A BaseTrainerWorker that records which datums its shard actually saw."""

  def __init__(self, shard_rank: int = 0, shard_count: int = 1):
    super().__init__()
    self._shard_rank = shard_rank
    self._shard_count = shard_count
    self.seen: list[int] = []

  def shard_rank(self) -> int:
    return self._shard_rank

  def shard_count(self) -> int:
    return self._shard_count

  def shard_all_reduce_sum(self, value: float) -> float:
    return value

  def shard_all_gather_object(self, value):
    # Stand in for the other ranks: every datum this shard did not own comes
    # back from a peer, exactly as the real all_gather does.
    peers = {idx: {"logprobs": {"data": [0.0, 0.0], "dtype": "float32", "shape": [2]}} for idx in range(6)}
    return [value, peers]

  def compute_target_logprobs(self, model, input_ids, attention_mask, target_token_ids):
    self.seen.extend(int(row[0]) for row in input_ids)
    return torch.zeros_like(target_token_ids, dtype=torch.float32, requires_grad=True)


class ShardingHookTest(unittest.TestCase):
  """forward_backward must shard through the hooks, not through global rank.

  Under tensor parallelism the ranks holding one model are not independent data
  shards -- they hold different slices of the same weights and have to be fed
  the same tokens. The Megatron worker points these hooks at its data-parallel
  subgroup; if forward_backward ignored them it would hand each tensor-parallel
  peer a different batch and every gradient would be wrong.
  """

  def test_overridden_hooks_select_the_shard(self) -> None:
    worker = RecordingWorker(shard_rank=1, shard_count=2)
    result = worker.forward_backward(torch.nn.Linear(1, 1), [datum(i) for i in range(6)], "cross_entropy")

    self.assertEqual(sorted(worker.seen), [1, 3, 5])
    # Every datum still gets an output; the ones this shard skipped come back
    # from the gather.
    self.assertEqual(len(result["loss_fn_outputs"]), 6)

  def test_default_hooks_keep_single_process_behaviour(self) -> None:
    worker = RecordingWorker()
    worker.forward_backward(torch.nn.Linear(1, 1), [datum(i) for i in range(6)], "cross_entropy")

    self.assertEqual(sorted(worker.seen), [0, 1, 2, 3, 4, 5])


class FakeBucket:
  def __init__(self, param_data: torch.Tensor):
    self.param_data = param_data
    self.grad_data = None


class FakeBuffer:
  """The shape of megatron-core's _ParamAndGradBuffer that offload depends on."""

  def __init__(self, numel: int):
    self.param_data = torch.arange(numel, dtype=torch.float32)
    self.grad_data = torch.zeros(numel, dtype=torch.float32)
    self.buckets = [FakeBucket(self.param_data[: numel // 2]), FakeBucket(self.param_data[numel // 2 :])]


class FakeChunk:
  def __init__(self, buffer: FakeBuffer, shapes: list[tuple[int, int]]):
    self.buffers = [buffer]
    self._params = []
    offset = 0
    for shape in shapes:
      size = shape[0] * shape[1]
      param = torch.nn.Parameter(torch.empty(shape))
      # Exactly how Megatron builds them: a view into the flat buffer.
      param.data = buffer.param_data[offset : offset + size].view(shape)
      param.grad = buffer.grad_data[offset : offset + size].view(shape)
      self._params.append(param)
      offset += size

  def parameters(self):
    return iter(self._params)


class BufferOffloadTest(unittest.TestCase):
  """sleep()/wake_up() move the flat buffer and rebuild every view into it.

  nn.Module.to() cannot do this: each parameter's .data is a view, so .to()
  copies the views and leaves the flat buffer -- the tensor that holds the
  memory -- on the GPU. The offload frees nothing and the next grad reduction
  writes into storage no parameter reads. move_flat_buffers instead moves the
  flat tensor and re-derives each view's offset from its old data_ptr, so the
  arithmetic below is what keeps weights attached to themselves.
  """

  def setUp(self) -> None:
    self.buffer = FakeBuffer(24)
    self.chunk = FakeChunk(self.buffer, [(2, 4), (4, 4)])
    self.worker = MegatronTrainingWorker.__new__(MegatronTrainingWorker)
    self.worker.model_chunks = [self.chunk]

  def test_flat_buffers_are_discovered(self) -> None:
    self.assertEqual(self.worker.flat_buffers(), [self.buffer])

  def test_a_buffer_without_param_data_disables_offload(self) -> None:
    self.worker.model_chunks = [type("Bare", (), {"buffers": [object()], "parameters": lambda self: iter(())})()]
    self.assertIsNone(self.worker.flat_buffers())

  def test_views_into_finds_params_grads_and_buckets(self) -> None:
    flat = self.buffer.param_data
    low = flat.data_ptr()
    high = low + flat.numel() * flat.element_size()
    views = self.worker.views_into(self.buffer, low, high)

    # Two parameters plus two bucket views alias param_data; the two gradient
    # views alias grad_data and must not be swept up with them.
    self.assertEqual(len(views), 4)
    self.assertEqual(sorted(view.numel() for view in views), [8, 12, 12, 16])

  def test_offsets_reconstruct_every_view_after_a_move(self) -> None:
    flat = self.buffer.param_data
    base, itemsize = flat.data_ptr(), flat.element_size()
    views = self.worker.views_into(self.buffer, base, base + flat.numel() * itemsize)
    originals = [view.clone() for view in views]

    # Same arithmetic move_flat_buffers runs, against a relocated storage.
    moved = flat.clone()
    for view, original in zip(views, originals, strict=True):
      offset = (view.data_ptr() - base) // itemsize
      rebuilt = moved[offset : offset + view.numel()].view(view.shape)
      self.assertTrue(torch.equal(rebuilt, original))
      self.assertEqual(rebuilt.data_ptr(), moved.data_ptr() + offset * itemsize)

  def test_move_to_the_same_device_is_a_no_op(self) -> None:
    pointers = [param.data.data_ptr() for param in self.chunk.parameters()]
    self.worker.move_flat_buffers("cpu")
    self.assertEqual([param.data.data_ptr() for param in self.chunk.parameters()], pointers)


class FakeOutputLayer:
  """A ColumnParallelLinear as chunked_target_logprobs uses it: called with
  [rows, 1, hidden], handed an explicit weight, returning (logits, bias)."""

  def __init__(self):
    self.sequence_parallel = False
    self.disable_grad_reduce = False
    self.weight = None
    self.row_counts: list[int] = []

  def __call__(self, hidden, weight):
    self.row_counts.append(hidden.shape[0])
    return torch.nn.functional.linear(hidden, weight), None


def cross_entropy(labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
  """Megatron's compute_language_model_loss: labels [batch, seq], logits [seq, batch, vocab]."""
  labels = labels.transpose(0, 1)
  loss = torch.logsumexp(logits, dim=-1) - logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
  return loss.transpose(0, 1)


class ChunkedProjectionTest(unittest.TestCase):
  """The chunked head must agree with an unchunked one, values and gradients.

  Everything risky here is bookkeeping. Megatron is sequence-first and this
  interface is batch-first, so the flatten/transpose pair has to line each row
  of hidden state up with the right target; get it wrong and the loss is still
  finite, still decreases, and trains the model on shifted labels. The chunk
  boundary and the checkpointed backward are the other two ways it can be
  quietly wrong, so the sizes below deliberately do not divide evenly.
  """

  seq_len, batch, hidden_size, vocab, chunk = 5, 2, 4, 7, 3

  def setUp(self) -> None:
    torch.manual_seed(0)
    self.hidden = torch.randn(self.seq_len, self.batch, self.hidden_size, requires_grad=True)
    self.weight = torch.randn(self.vocab, self.hidden_size, requires_grad=True)
    self.labels = torch.randint(0, self.vocab, (self.batch, self.seq_len))
    self.output_layer = FakeOutputLayer()
    # Patched for the whole test, not just the forward call: checkpointing
    # re-runs the projection from inside backward().
    patcher = unittest.mock.patch.object(megatron_worker, "MEGATRON_LOGPROB_CHUNK", self.chunk)
    patcher.start()
    self.addCleanup(patcher.stop)

  def reference(self) -> torch.Tensor:
    logits = torch.nn.functional.linear(self.hidden, self.weight)
    return -cross_entropy(self.labels, logits)

  def run_chunked(self, output_weight: torch.Tensor | None = None) -> torch.Tensor:
    return chunked_target_logprobs(
      hidden_states=self.hidden,
      output_layer=self.output_layer,
      output_weight=output_weight,
      labels=self.labels,
      config=types.SimpleNamespace(sequence_parallel=False),
      compute_language_model_loss=cross_entropy,
      scale_logits=lambda logits: logits,
    )

  def test_values_match_an_unchunked_projection(self) -> None:
    self.output_layer.weight = self.weight
    chunked = self.run_chunked()

    self.assertEqual(chunked.shape, (self.batch, self.seq_len))
    # 10 rows in chunks of 3: the last one is short, which is the boundary that
    # a stop index computed from the chunk size rather than the tensor gets wrong.
    self.assertEqual(self.output_layer.row_counts, [3, 3, 3, 1])
    torch.testing.assert_close(chunked, self.reference())

  def test_gradients_survive_checkpointing(self) -> None:
    self.output_layer.weight = self.weight
    self.run_chunked().sum().backward()
    chunked_grads = (self.hidden.grad.clone(), self.weight.grad.clone())

    self.hidden.grad = self.weight.grad = None
    self.reference().sum().backward()

    torch.testing.assert_close(chunked_grads[0], self.hidden.grad)
    torch.testing.assert_close(chunked_grads[1], self.weight.grad)

  def test_tied_embedding_weight_is_preferred(self) -> None:
    # Gemma ties embeddings, so the output layer allocates no weight of its own
    # and GPTModel hands the shared one to the processor instead.
    self.assertIsNone(self.output_layer.weight)
    torch.testing.assert_close(self.run_chunked(output_weight=self.weight), self.reference())

  def test_sequence_parallel_flags_are_restored(self) -> None:
    # The layer must not be left with the overrides the projection sets, or the
    # next unchunked caller silently skips the gather and the dgrad reduction.
    self.output_layer.weight = self.weight
    self.output_layer.sequence_parallel = True
    gathered = []

    mappings = types.ModuleType("megatron.core.tensor_parallel.mappings")
    mappings.gather_from_sequence_parallel_region = lambda hidden: gathered.append(hidden) or hidden
    modules = {
      "megatron": types.ModuleType("megatron"),
      "megatron.core": types.ModuleType("megatron.core"),
      "megatron.core.tensor_parallel": types.ModuleType("megatron.core.tensor_parallel"),
      "megatron.core.tensor_parallel.mappings": mappings,
    }
    with unittest.mock.patch.dict(sys.modules, modules):
      chunked_target_logprobs(
        hidden_states=self.hidden,
        output_layer=self.output_layer,
        output_weight=None,
        labels=self.labels,
        config=types.SimpleNamespace(sequence_parallel=True),
        compute_language_model_loss=cross_entropy,
        scale_logits=lambda logits: logits,
      )

    self.assertEqual(len(gathered), 1, "the gather must happen once, not once per chunk")
    self.assertTrue(self.output_layer.sequence_parallel)
    self.assertFalse(self.output_layer.disable_grad_reduce)


class FlexBlockMaskTest(unittest.TestCase):
  """The block mask is the whole definition of who attends to whom.

  FlexAttention has no separate mask; get the predicate wrong and attention is
  silently wrong -- a window one token short, or a causal layer that peeks --
  with no shape error to catch it. create_block_mask itself is torch's, so what
  is worth testing is the predicate handed to it and the cache around it.
  """

  def setUp(self) -> None:
    self.built: list[tuple] = []

    def fake_create_block_mask(keep, *, B, H, Q_LEN, KV_LEN, device):  # noqa: N803
      self.built.append((keep, Q_LEN, KV_LEN))
      return f"mask{len(self.built)}"

    patcher = unittest.mock.patch.object(gemma4, "_create_block_mask_compiled", fake_create_block_mask)
    patcher.start()
    self.addCleanup(patcher.stop)
    cache = unittest.mock.patch.object(gemma4, "_flex_block_masks", {})
    cache.start()
    self.addCleanup(cache.stop)

  def allowed(self, window: int | None, seq: int = 8) -> torch.Tensor:
    gemma4.flex_block_mask(window, seq, seq, "cpu")
    keep = self.built[-1][0]
    q_idx = torch.arange(seq).unsqueeze(1)
    kv_idx = torch.arange(seq).unsqueeze(0)
    return keep(None, None, q_idx, kv_idx)

  def test_causal_layers_attend_to_every_earlier_token(self) -> None:
    seq = 8
    expected = torch.arange(seq).unsqueeze(1) >= torch.arange(seq).unsqueeze(0)
    torch.testing.assert_close(self.allowed(None, seq), expected)

  def test_sliding_layers_keep_the_window_inclusive_of_both_ends(self) -> None:
    # Megatron's own get_sliding_window_causal_mask keeps kv in [q - w, q], so a
    # window of 3 is four tokens wide. Off by one here and the two paths differ.
    allowed = self.allowed(3, seq=8)
    self.assertEqual(allowed[7].tolist(), [False, False, False, False, True, True, True, True])
    self.assertEqual(allowed[0].tolist(), [True] + [False] * 7)

  def test_masks_are_built_once_per_window_and_shape(self) -> None:
    first = gemma4.flex_block_mask(1023, 4096, 4096, "cpu")
    again = gemma4.flex_block_mask(1023, 4096, 4096, "cpu")
    self.assertIs(first, again)
    self.assertEqual(len(self.built), 1, "a cache hit must not rebuild the mask")

    # The two things that make a mask different: the window and the shape.
    gemma4.flex_block_mask(None, 4096, 4096, "cpu")
    gemma4.flex_block_mask(1023, 8192, 8192, "cpu")
    self.assertEqual(len(self.built), 3)

  def test_the_cache_is_bounded_because_every_step_brings_a_new_length(self) -> None:
    # The packer pads to a multiple of TP, so lengths rarely repeat and an
    # unbounded cache is a leak of quadratically-sized masks.
    for seq in range(1024, 1024 * 40, 1024):
      gemma4.flex_block_mask(1023, seq, seq, "cpu")
    self.assertEqual(len(gemma4._flex_block_masks), gemma4._FLEX_BLOCK_MASK_CACHE_SIZE)

  def test_the_shape_in_flight_survives_eviction(self) -> None:
    # A single forward alternates the sliding and global masks of one shape
    # across 48 layers. Evicting either mid-pass would rebuild it 24 times.
    size = gemma4._FLEX_BLOCK_MASK_CACHE_SIZE
    for _layer in range(size * 3):
      gemma4.flex_block_mask(1023, 4096, 4096, "cpu")
      gemma4.flex_block_mask(None, 4096, 4096, "cpu")
    self.assertEqual(len(self.built), 2, "the pair in use must never be evicted")

  def test_eviction_is_least_recently_used_not_first_inserted(self) -> None:
    size = gemma4._FLEX_BLOCK_MASK_CACHE_SIZE
    oldest = gemma4.flex_block_mask(1023, 4096, 4096, "cpu")
    for seq in range(8192, 8192 + 1024 * (size - 1), 1024):
      gemma4.flex_block_mask(1023, seq, seq, "cpu")
    # Touch the oldest entry, then overflow by one: the untouched one goes.
    self.assertIs(gemma4.flex_block_mask(1023, 4096, 4096, "cpu"), oldest)
    gemma4.flex_block_mask(1023, 999424, 999424, "cpu")
    self.assertIn((1023, 4096, 4096), gemma4._flex_block_masks)
    self.assertNotIn((1023, 8192, 8192), gemma4._flex_block_masks)

  def test_disabling_the_knob_leaves_megatrons_attention_alone(self) -> None:
    with unittest.mock.patch.object(gemma4, "MEGATRON_FLEX_ATTENTION", False):
      self.assertFalse(gemma4.install_flex_attention())

  def test_a_missing_megatron_is_not_fatal(self) -> None:
    # Best-effort by contract: without the bridge the run should fall back to
    # megatron's attention, not fail to load the model.
    with unittest.mock.patch.object(gemma4, "MEGATRON_FLEX_ATTENTION", True):
      self.assertFalse(gemma4.install_flex_attention())


class TensorParallelAgreementTest(unittest.TestCase):
  """A TP-group batch mismatch must raise here, not hang in NCCL.

  This is the check that would have named the run31 deadlock in a second
  instead of eight minutes: the ranks entered attention with different
  sequence lengths and sat in an all-reduce that could never match.
  """

  def worker_seeing(self, peers: list[tuple[int, int]]) -> MegatronTrainingWorker:
    worker = MegatronTrainingWorker()
    worker.tp_size = len(peers)
    worker.device = torch.device("cpu")

    def fake_all_gather(out_list, _tensor, group=None):
      for slot, peer in zip(out_list, peers):
        slot.copy_(torch.tensor(peer, dtype=torch.long))

    self.enterContext(unittest.mock.patch.object(megatron_worker.dist, "is_initialized", return_value=True))
    self.enterContext(unittest.mock.patch.object(megatron_worker.dist, "all_gather", fake_all_gather))
    self.enterContext(unittest.mock.patch.object(megatron_worker, "require_megatron", return_value=(None, unittest.mock.MagicMock())))
    return worker

  def test_matching_ranks_pass(self) -> None:
    worker = self.worker_seeing([(3, 6), (3, 6)])
    worker.assert_tp_ranks_agree([datum(i) for i in range(3)])

  def test_a_different_datum_count_is_reported(self) -> None:
    worker = self.worker_seeing([(3, 6), (4, 8)])
    with self.assertRaisesRegex(RuntimeError, r"different batches.*\(3, 6\).*\(4, 8\)"):
      worker.assert_tp_ranks_agree([datum(i) for i in range(3)])

  def test_same_count_different_lengths_is_reported(self) -> None:
    # The run31 shape: equal datum counts, different token totals.
    worker = self.worker_seeing([(3, 6), (3, 9)])
    with self.assertRaisesRegex(RuntimeError, "different batches"):
      worker.assert_tp_ranks_agree([datum(i) for i in range(3)])

  def test_tp1_does_not_collective(self) -> None:
    worker = MegatronTrainingWorker()
    worker.tp_size = 1
    with unittest.mock.patch.object(megatron_worker.dist, "all_gather", side_effect=AssertionError("collective")):
      worker.assert_tp_ranks_agree([datum(0)])


class BlockModeloptImport(unittest.mock.MagicMock):
  """A meta_path finder that makes `import modelopt` fail as it does in prod.

  Without this the tests would pass or fail depending on whether the machine
  running them happens to have modelopt installed, which is exactly the
  difference between the box and CI that let run32 ship.
  """

  def find_spec(self, name, path=None, target=None):
    if name == "modelopt" or name.startswith("modelopt."):
      raise ModuleNotFoundError(f"No module named {name!r}", name=name)
    return None


def modelopt_module(**attrs) -> types.ModuleType:
  module = types.ModuleType("modelopt.torch.quantization.utils")
  for key, value in attrs.items():
    setattr(module, key, value)
  return module


class ModeloptGuardTest(unittest.TestCase):
  """ensure_modelopt_importable has to hold in all three dependency states.

  megatron-bridge's save_hf_weights does an unguarded `from
  modelopt.torch.quantization.utils import is_quantized` and uses the answer only
  to decide whether to strip `_quantizer.` tensors. A bf16 run needs False and
  cannot get it without the package, so every save dies. The guard supplies the
  symbol -- but only when it is genuinely missing, or it would shadow a real
  modelopt and answer False for a model that really is quantized.
  """

  def setUp(self) -> None:
    saved = {name: module for name, module in sys.modules.items() if name.split(".")[0] == "modelopt"}

    def restore() -> None:
      for name in [name for name in sys.modules if name.split(".")[0] == "modelopt"]:
        del sys.modules[name]
      sys.modules.update(saved)

    self.addCleanup(restore)
    for name in saved:
      del sys.modules[name]

  def block_disk_imports(self) -> None:
    finder = BlockModeloptImport()
    sys.meta_path.insert(0, finder)
    self.addCleanup(sys.meta_path.remove, finder)

  def resolve(self):
    from modelopt.torch.quantization.utils import is_quantized

    return is_quantized

  def test_a_missing_modelopt_gets_the_one_symbol_the_save_path_reads(self) -> None:
    self.block_disk_imports()
    with self.assertRaises(ModuleNotFoundError):
      self.resolve()

    megatron_worker.ensure_modelopt_importable()

    # False is the answer, not a placeholder: with no modelopt installed nothing
    # could have quantized this model in the first place.
    self.assertFalse(self.resolve()(object()))

  def test_a_partial_install_is_repaired_rather_than_trusted(self) -> None:
    # The box's actual state, left by a half-finished shim: the package imports
    # and satisfies find_spec, and the symbol still is not there. A probe on the
    # package would return early here and the save would die anyway.
    self.block_disk_imports()
    for name in ("modelopt", "modelopt.torch", "modelopt.torch.quantization"):
      sys.modules[name] = types.ModuleType(name)
    sys.modules["modelopt.torch.quantization.utils"] = modelopt_module()
    # The package imports; only the name inside it is missing. Anything that
    # probes at package granularity sees a working modelopt here.
    importlib.import_module("modelopt")
    with self.assertRaises(ImportError):
      self.resolve()

    megatron_worker.ensure_modelopt_importable()

    self.assertFalse(self.resolve()(object()))

  def test_a_real_modelopt_is_left_alone(self) -> None:
    for name in ("modelopt", "modelopt.torch", "modelopt.torch.quantization"):
      sys.modules[name] = types.ModuleType(name)
    sys.modules["modelopt.torch.quantization.utils"] = modelopt_module(is_quantized=lambda _model: True)

    megatron_worker.ensure_modelopt_importable()

    # Shadowing a real install would answer False for a genuinely quantized
    # model and silently write `_quantizer.` tensors into the checkpoint.
    self.assertTrue(self.resolve()(object()))

  def test_calling_it_twice_is_stable(self) -> None:
    self.block_disk_imports()
    megatron_worker.ensure_modelopt_importable()
    first = self.resolve()
    megatron_worker.ensure_modelopt_importable()
    self.assertIs(self.resolve(), first)


class UpstreamBridge:
  """save_hf_pretrained as megatron-bridge actually implements it.

  The unguarded import is the whole point: this is the line that killed run32 at
  its first checkpoint, five and a half hours in.
  """

  def __init__(self, calls: list[str], fail: bool = False):
    self.calls = calls
    self.fail = fail

  def save_hf_pretrained(self, _chunks, path: str) -> None:
    from modelopt.torch.quantization.utils import is_quantized

    self.calls.append(f"save_hf_pretrained(is_quantized={is_quantized(None)})")
    if self.fail:
      raise RuntimeError("write failed")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "model.safetensors"), "w") as handle:
      handle.write("weights")


class CheckpointSaveTest(unittest.TestCase):
  """The save path, exercised without megatron, a GPU, or four hours of training.

  Every Megatron run so far has died at a checkpoint boundary rather than in
  training, and always after enough steps that the failure cost the whole run.
  save_every decides how late that test runs; these cases run it at import time.
  """

  def setUp(self) -> None:
    saved = {name: module for name, module in sys.modules.items() if name.split(".")[0] == "modelopt"}

    def restore() -> None:
      for name in [name for name in sys.modules if name.split(".")[0] == "modelopt"]:
        del sys.modules[name]
      sys.modules.update(saved)

    self.addCleanup(restore)
    for name in saved:
      del sys.modules[name]
    finder = BlockModeloptImport()
    sys.meta_path.insert(0, finder)
    self.addCleanup(sys.meta_path.remove, finder)

    self.calls: list[str] = []
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, True)
    self.enterContext(unittest.mock.patch.object(megatron_worker, "is_primary", return_value=True))
    self.enterContext(unittest.mock.patch.object(megatron_worker, "barrier", lambda: None))

  def worker(self, fail: bool = False, optimizer=None) -> MegatronTrainingWorker:
    worker = MegatronTrainingWorker.__new__(MegatronTrainingWorker)
    worker.model_chunks = [object()]
    worker.bridge = UpstreamBridge(self.calls, fail=fail)
    worker.tokenizer = None
    worker.optimizer = optimizer
    worker.base_model_name = "google/gemma-4-12B-it"
    return worker

  def read_metadata(self, path: str) -> dict:
    with open(os.path.join(path, "metadata.json")) as handle:
      return json.load(handle)

  def test_a_save_survives_a_machine_without_modelopt(self) -> None:
    # Delete the guard and this case is run32: ModuleNotFoundError, no
    # checkpoint, four steps of training with nowhere to land.
    path = os.path.join(self.root, "state")
    self.worker().save_checkpoint(path, {"kind": "state"})

    self.assertEqual(self.calls, ["save_hf_pretrained(is_quantized=False)"])
    self.assertTrue(os.path.exists(os.path.join(path, "model.safetensors")))
    self.assertEqual(self.read_metadata(path), {"kind": "state"})

  def test_no_staging_directory_is_left_where_a_loader_would_find_it(self) -> None:
    path = os.path.join(self.root, "state")
    self.worker().save_checkpoint(path, {})
    self.assertEqual(os.listdir(self.root), ["state"])

  def test_a_failed_save_does_not_damage_the_previous_checkpoint(self) -> None:
    # The reason for the staging dance: a save killed mid-write must not leave a
    # directory that vLLM or a resume can load as a mix of old and new shards.
    path = os.path.join(self.root, "state")
    os.makedirs(path)
    with open(os.path.join(path, "model.safetensors"), "w") as handle:
      handle.write("old weights")

    with self.assertRaises(RuntimeError):
      self.worker(fail=True).save_checkpoint(path, {})

    with open(os.path.join(path, "model.safetensors")) as handle:
      self.assertEqual(handle.read(), "old weights")

  def test_a_second_save_replaces_the_first(self) -> None:
    path = os.path.join(self.root, "state")
    worker = self.worker()
    worker.save_checkpoint(path, {"step": 1})
    worker.save_checkpoint(path, {"step": 2})

    self.assertEqual(self.read_metadata(path), {"step": 2})
    self.assertEqual(os.listdir(self.root), ["state"])

  def test_save_state_records_what_load_from_state_requires(self) -> None:
    # load_from_state refuses a checkpoint without base_model and skips the
    # optimizer unless has_optimizer says it is there, so both are load-bearing.
    path = os.path.join(self.root, "state")
    worker = self.worker(optimizer=unittest.mock.MagicMock())
    with unittest.mock.patch.object(worker, "save_optimizer") as save_optimizer:
      worker.save_state("model-1", path, include_optimizer=True)

    metadata = self.read_metadata(path)
    self.assertEqual(metadata["base_model"], "google/gemma-4-12B-it")
    self.assertEqual(metadata["model_id"], "model-1")
    self.assertTrue(metadata["has_optimizer"])
    save_optimizer.assert_called_once()

  def test_include_optimizer_without_an_optimizer_is_recorded_honestly(self) -> None:
    path = os.path.join(self.root, "state")
    self.worker().save_state("model-1", path, include_optimizer=True)
    # Claiming an optimizer that was never written would make a resume restore
    # nothing and report success.
    self.assertFalse(self.read_metadata(path)["has_optimizer"])

  def test_a_sampler_save_reaching_save_state_writes_a_whole_checkpoint(self) -> None:
    # Sampler weights are published as an adapter by the processor, which never
    # calls this. Anything that does get here wants the merged model.
    path = os.path.join(self.root, "sampler")
    self.worker().save_state("model-1", path, kind="sampler")

    self.assertEqual(self.read_metadata(path)["kind"], "sampler")
    self.assertEqual(self.calls, ["save_hf_pretrained(is_quantized=False)"])


class CheckpointingException(Exception):
  """megatron.core.dist_checkpointing.core.CheckpointingException, by value."""


class FormatPickingOptimizer:
  """Stands in for megatron-core's choice of optimizer sharding format.

  One thing about that seam matters and it is not obvious: sharded_state_dict
  picks a format by name out of metadata, and the name it defaults to cannot be
  written. 'fully_sharded_model_space' builds every ShardedTensor with a
  flattened_range and then validates it against a validator that rejects any
  flattened_range at all -- the writer and the validator ship in the same
  release. Passing no metadata therefore raises, on every rank, and only at a
  checkpoint boundary: run35 spent four hours reaching that line.
  """

  WRITABLE = frozenset({"fully_reshardable", "dp_reshardable", "dp_zero_gather_scatter", "fsdp_dtensor"})

  def __init__(self) -> None:
    self.requested: list[str] = []

  def sharded_state_dict(self, model_sharded_state_dict, is_loading=False, metadata=None):
    name = (metadata or {}).get("distrib_optim_sharding_type", "fully_sharded_model_space")
    self.requested.append(name)
    if name not in self.WRITABLE:
      raise CheckpointingException("ShardedTensor.flattened_range is not supported.")
    return {"param_state_sharding_type": name}

  def load_state_dict(self, state_dict) -> None:
    self.loaded = state_dict


class OptimizerFormatTest(unittest.TestCase):
  """Which optimizer sharding format the save and the load ask megatron for."""

  def setUp(self) -> None:
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, True)
    self.enterContext(unittest.mock.patch.object(megatron_worker, "is_primary", return_value=True))
    self.enterContext(unittest.mock.patch.object(megatron_worker, "barrier", lambda: None))

    self.saved: list[str] = []
    dist_checkpointing = types.ModuleType("megatron.core.dist_checkpointing")
    dist_checkpointing.save = lambda state, path: self.saved.append(path)
    dist_checkpointing.load = lambda state, path: {"restored": path}
    core = types.ModuleType("megatron.core")
    core.dist_checkpointing = dist_checkpointing
    modules = {"megatron": types.ModuleType("megatron"), "megatron.core": core}
    self.enterContext(unittest.mock.patch.dict(sys.modules, modules))

    self.optimizer = FormatPickingOptimizer()
    self.worker = MegatronTrainingWorker.__new__(MegatronTrainingWorker)
    self.worker.optimizer = self.optimizer
    self.worker.model_chunks = []

  def test_the_save_asks_for_a_format_megatron_can_actually_write(self) -> None:
    # Drop the metadata argument and this is run35: CheckpointingException on
    # all four ranks, at the first checkpoint, with the run's work unsaved.
    path = os.path.join(self.root, OPTIMIZER_SUBDIR)
    self.worker.save_optimizer(path)

    self.assertEqual(self.optimizer.requested, ["fully_reshardable"])
    self.assertEqual(self.saved, [path])

  def test_the_load_asks_for_the_same_format_the_save_wrote(self) -> None:
    # The format is recorded in the checkpoint as param_state_sharding_type and
    # dispatched on at load, so a load that names a different one reads the
    # bytes back through the wrong layout.
    path = os.path.join(self.root, OPTIMIZER_SUBDIR)
    self.worker.save_optimizer(path)
    self.worker.load_optimizer(path, {})

    self.assertEqual(self.optimizer.requested, ["fully_reshardable", "fully_reshardable"])
    self.assertEqual(self.optimizer.loaded, {"restored": path})


# google/gemma-4-12B-it's checkpoint index in miniature: a language tower the
# Megatron export does produce, four multimodal tensors it does not, and -- the
# detail that turned run34 from a partial save into no save -- one shard file
# holding all of them.
LANGUAGE_KEYS = (
  "model.language_model.embed_tokens.weight",
  "model.language_model.layers.0.self_attn.q_proj.weight",
  "model.language_model.layers.1.self_attn.q_proj.weight",
  "model.language_model.norm.weight",
)
MULTIMODAL_KEYS = (
  "model.embed_audio.embedding_projection.weight",
  "model.embed_vision.embedding_projection.weight",
  "model.vision_embedder.patch_dense.weight",
  "model.vision_embedder.pos_embedding",
)
SINGLE_SHARD_INDEX = {key: "model.safetensors" for key in LANGUAGE_KEYS + MULTIMODAL_KEYS}
# The config save_artifacts copies out of the source snapshot, verbatim, before
# a single tensor is written. vision_config is the part that matters: it is what
# tells vLLM and transformers to expect a vision tower in the weights.
SOURCE_CONFIG = {
  "architectures": ["Gemma4ForConditionalGeneration"],
  "model_type": "gemma4_unified",
  "vision_config": {"hidden_size": 2},
}
# A distinct value per key, so a save that writes the right names with the wrong
# tensors is caught as well as one that writes nothing.
FULL_EXPORT = {key: torch.full((2, 2), float(n + 1), dtype=torch.bfloat16) for n, key in enumerate(SINGLE_SHARD_INDEX)}
# What the run34 export actually yielded: the language tower and nothing else.
LANGUAGE_EXPORT = {key: FULL_EXPORT[key] for key in LANGUAGE_KEYS}
# Every tensor produced, under names the source index has never heard of -- the
# shape of a bridge whose key mapping drifted from the checkpoint's.
RENAMED_EXPORT = {key.replace("model.", "model.decoder."): value for key, value in FULL_EXPORT.items()}


class ShardedSaveBridge:
  """save_hf_pretrained with the shard-completeness rule the real bridge enforces.

  megatron-bridge's SafeTensorsStateSource.save_generator writes a source shard
  file only once every key the source checkpoint assigned to that shard has come
  out of the export generator, and drops any yielded name the source does not
  know. Under strict=True an incomplete shard is skipped and the save raises
  after the complete ones are written; under strict=False the incomplete shard is
  written anyway, short whatever is missing, and the save returns normally.
  Either way save_artifacts has already copied the source config.json into the
  output directory, before the first tensor is looked at.

  Both branches were checked against megatron-bridge 0.6.1 on CPU with a toy
  checkpoint of this shape before this stub was written. run34 is the strict=True
  branch, and the reason it lost everything is that google/gemma-4-12B-it ships
  as one unsharded model.safetensors with no index file: all 677 keys belong to
  one shard, so the 11 the text-only bridge never yields took the 666 that were
  yielded down with them.
  """

  def __init__(self, index: dict[str, str], export: dict[str, torch.Tensor], strict: bool = True):
    self.index = index
    self.export = export
    self.strict = strict

  def save_hf_pretrained(self, _chunks, path: str) -> None:
    # The unguarded import the real save path does on its way in.
    from modelopt.torch.quantization.utils import is_quantized

    assert not is_quantized(None)
    os.makedirs(path, exist_ok=True)
    # save_artifacts runs first and unconditionally: config.json describes the
    # source model whatever the weight write goes on to do.
    with open(os.path.join(path, "config.json"), "w") as handle:
      json.dump(SOURCE_CONFIG, handle)

    shards: dict[str, list[str]] = {}
    for key, filename in self.index.items():
      shards.setdefault(filename, []).append(key)

    written = 0
    for filename, keys in shards.items():
      present = {key: self.export[key] for key in keys if key in self.export}
      if len(present) < len(keys) and self.strict:
        continue
      save_file(present, os.path.join(path, filename))
      written += len(present)

    if written < len(self.index) and self.strict:
      raise RuntimeError(
        f"{len(self.index) - written} tensors from the original checkpoint were not written. "
        "Re-run with strict=False to save the partial checkpoint instead of failing."
      )


class ResumingWorker(MegatronTrainingWorker):
  """A worker whose load side reads the HF checkpoint back off disk.

  load_from_state hands the checkpoint directory to load_base_model, which in the
  real worker is AutoBridge.from_hf_pretrained streaming the safetensors into
  Megatron's sharded layout. Reading every shard in the directory -- what a
  loader does when there is no index file, which is this model's layout -- is the
  part of that the round trip turns on: it decides which weights come back.
  """

  def load_base_model(self, base_model_name: str) -> None:
    self.restored = {}
    for filename in sorted(os.listdir(base_model_name)):
      if filename.endswith(".safetensors"):
        self.restored.update(load_file(os.path.join(base_model_name, filename)))
    self.model_chunks = [object()]

  def prepare_model_for_training(self) -> None:
    pass


class CheckpointRoundTripTest(unittest.TestCase):
  """save_state -> load_from_state, against a bridge that can write less than it got.

  run34 died in the middle of this round trip at step 1. The crash is the cheap
  half: it costs one run and says exactly what happened. The expensive half is
  the fix its error message recommends -- "Re-run with strict=False to save the
  partial checkpoint instead of failing" -- which turns the stop into a save that
  reports success over a checkpoint the model cannot be rebuilt from. Nothing
  downstream would catch that: load_from_state validates metadata.json, and
  metadata.json is written by us, after the bridge, whatever the bridge did.
  """

  def setUp(self) -> None:
    saved = {name: module for name, module in sys.modules.items() if name.split(".")[0] == "modelopt"}

    def restore() -> None:
      for name in [name for name in sys.modules if name.split(".")[0] == "modelopt"]:
        del sys.modules[name]
      sys.modules.update(saved)

    self.addCleanup(restore)
    for name in saved:
      del sys.modules[name]
    finder = BlockModeloptImport()
    sys.meta_path.insert(0, finder)
    self.addCleanup(sys.meta_path.remove, finder)

    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, True)
    self.enterContext(unittest.mock.patch.object(megatron_worker, "is_primary", return_value=True))
    self.enterContext(unittest.mock.patch.object(megatron_worker, "barrier", lambda: None))

  def worker(self, export: dict[str, torch.Tensor], strict: bool = True) -> ResumingWorker:
    worker = ResumingWorker.__new__(ResumingWorker)
    worker.model_chunks = [object()]
    worker.bridge = ShardedSaveBridge(SINGLE_SHARD_INDEX, export, strict=strict)
    worker.tokenizer = None
    worker.optimizer = None
    worker.base_model_name = "google/gemma-4-12B-it"
    worker.restored = {}
    return worker

  def save_or_refuse(self, worker: ResumingWorker, path: str) -> bool:
    """Run the save; report whether it committed anything, and hold it to that.

    Deliberately does not pin down which way a fix goes. Refusing to commit a
    checkpoint the export could not fill and getting the export to fill it are
    both correct outcomes, and the cases below pass under either. Returning
    success over a checkpoint the next process cannot use is the one outcome
    that is not correct, because it is the one that spends GPU-hours instead of
    ending them.
    """
    try:
      worker.save_state("model-1", path)
    except RuntimeError:
      self.assertFalse(os.path.exists(path), "a save that raised must not leave a checkpoint a resume would pick up")
      return False
    return True

  def assert_resumable_or_loud(self, worker: ResumingWorker, path: str) -> None:
    """Whatever the save committed, a resume must get every tensor back intact."""
    if not self.save_or_refuse(worker, path):
      return

    worker.load_from_state("model-1", path)
    self.assertEqual(set(worker.restored), set(SINGLE_SHARD_INDEX))
    for key, expected in FULL_EXPORT.items():
      self.assertTrue(torch.equal(worker.restored[key], expected), key)

  def test_a_complete_export_round_trips_every_tensor(self) -> None:
    self.assert_resumable_or_loud(self.worker(FULL_EXPORT), os.path.join(self.root, "state"))

  def test_run34_an_export_one_shard_short_writes_nothing_and_says_so(self) -> None:
    # run34 scaled down. The export is only missing the four multimodal tensors,
    # but the source index keeps all eight keys in one model.safetensors, so no
    # shard completes and the language tensors that were produced go in the bin
    # with the rest. The count in the message is the whole model, not the gap.
    path = os.path.join(self.root, "state")
    with self.assertRaisesRegex(RuntimeError, "8 tensors from the original checkpoint were not written"):
      self.worker(LANGUAGE_EXPORT).save_state("model-1", path)

    self.assertFalse(os.path.exists(path))

  def test_a_failed_save_leaves_the_previous_checkpoint_resumable(self) -> None:
    # The staging dance exists for this: run34 crashed at step 1, and whatever
    # step 0 wrote had to still be there to resume from.
    path = os.path.join(self.root, "state")
    self.worker(FULL_EXPORT).save_state("model-0", path)
    with self.assertRaises(RuntimeError):
      self.worker(LANGUAGE_EXPORT).save_state("model-1", path)

    resumed = self.worker(FULL_EXPORT)
    resumed.load_from_state("model-0", path)
    self.assertEqual(set(resumed.restored), set(SINGLE_SHARD_INDEX))

  def test_a_partial_save_is_not_committed_as_a_resumable_checkpoint(self) -> None:
    # strict=False, the fix the bridge's own error message recommends. Measured
    # against megatron-bridge 0.6.1: it writes the shard minus the tensors the
    # export never produced, rewrites model.safetensors.index.json so the gap is
    # not even visible as a missing key, and returns. At run34's scale that is
    # 666 of 677 tensors with the whole vision tower gone, reported as a save.
    self.assert_resumable_or_loud(self.worker(LANGUAGE_EXPORT, strict=False), os.path.join(self.root, "state"))

  def test_a_save_that_writes_no_tensors_at_all_is_not_reported_as_success(self) -> None:
    # Same strict=False, but with the bridge's key mapping drifted off the
    # checkpoint's instead of the export being short: every yielded name misses
    # the source map, all of them are skipped with a warning, and an empty
    # model.safetensors is written. Measured on megatron-bridge 0.6.1 the file is
    # 16 bytes and holds zero tensors, and save_generator returns.
    self.assert_resumable_or_loud(self.worker(RENAMED_EXPORT, strict=False), os.path.join(self.root, "state"))

  def test_the_config_committed_beside_the_weights_describes_them(self) -> None:
    # save_artifacts copies the source config.json into the checkpoint before a
    # single tensor is looked at, so a text-only export under strict=False lands
    # a config advertising a vision tower next to weights that have none. Nothing
    # in this process notices: the mismatch surfaces in whatever loads the
    # checkpoint next, which is vLLM or a resume, both of which size the model
    # from this file. Making the save loud is what keeps the pair consistent --
    # a checkpoint that is never committed cannot disagree with itself.
    path = os.path.join(self.root, "state")
    worker = self.worker(LANGUAGE_EXPORT, strict=False)
    if not self.save_or_refuse(worker, path):
      return
    worker.load_from_state("model-1", path)

    with open(os.path.join(path, "config.json")) as handle:
      config = json.load(handle)
    self.assertEqual(
      "vision_config" in config,
      any(key.startswith("model.vision_embedder.") for key in worker.restored),
      "config.json describes a model the weights committed beside it do not contain",
    )


class AdapterBridge:
  """save_hf_adapter as megatron-bridge implements it, plus the base key list.

  Exports hub-layout names by default (model.language_model.layers.N...), which
  is what the real bridge was measured emitting under GEMMA4_CONVERSION_MODE=text.
  """

  def __init__(self, base_keys: tuple[str, ...], adapter_names: list[str] | None = None):
    self.hf_pretrained = types.SimpleNamespace(state=types.SimpleNamespace(source=types.SimpleNamespace(get_all_keys=lambda: list(base_keys))))
    self.adapter_names = adapter_names
    self.saved_config: dict | None = None

  def save_hf_adapter(self, _chunks, path: str, peft_config, base_model_name_or_path=None, show_progress=True) -> None:
    names = self.adapter_names
    if names is None:
      names = [
        f"base_model.model.{key[: -len('.weight')]}.lora_{side}.weight" for key in LANGUAGE_KEYS if key.endswith("q_proj.weight") for side in "AB"
      ]
    os.makedirs(path, exist_ok=True)
    save_file({name: torch.zeros(2, 2) for name in names}, os.path.join(path, "adapter_model.safetensors"))
    self.saved_config = {"r": peft_config.dim, "base_model_name_or_path": base_model_name_or_path}
    with open(os.path.join(path, "adapter_config.json"), "w") as handle:
      json.dump(self.saved_config, handle)


class AdapterPublishTest(unittest.TestCase):
  """Publishing sampler weights as a LoRA adapter instead of merged base weights.

  The case that matters is not a crash. vLLM applies the adapter modules whose
  names it recognises and silently applies none of the rest, so an adapter in
  the wrong layout trains for a whole run against the base model and the only
  symptom is a flat reward curve -- which is why write_adapter checks its own
  output against the base checkpoint's key list rather than trusting the bridge.
  """

  def setUp(self) -> None:
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, True)
    self.enterContext(unittest.mock.patch.dict(os.environ, {"OPEN_RL_SNAPSHOT_DIR": os.path.join(self.root, "peft")}))
    self.enterContext(unittest.mock.patch.object(megatron_worker, "MEGATRON_LORA_RANK", 32))
    self.enterContext(unittest.mock.patch.object(megatron_worker, "barrier", lambda: None))

  def worker(self, bridge: AdapterBridge) -> MegatronTrainingWorker:
    worker = MegatronTrainingWorker.__new__(MegatronTrainingWorker)
    worker.model_chunks = [object()]
    worker.bridge = bridge
    worker.base_model_name = "google/gemma-4-12B-it"
    worker.lora = types.SimpleNamespace(dim=32, alpha=32, dropout=0.0)
    return worker

  def test_the_adapter_lands_where_the_sampler_looks_for_it(self) -> None:
    # gateway.sampler_adapter_path resolves a sampling ref to
    # peft/<model>/<label>; anywhere else and the sampler gets a 404 for a file
    # the trainer swears it wrote.
    final = self.worker(AdapterBridge(LANGUAGE_KEYS)).write_adapter("model-1", session_label="sampler-7")

    self.assertEqual(final, os.path.join(self.root, "peft", "model-1", "sampler-7"))
    self.assertTrue(os.path.exists(os.path.join(final, "adapter_model.safetensors")))
    self.assertTrue(os.path.exists(os.path.join(final, "adapter_config.json")))
    # No staging directory left behind for a loader to trip over.
    self.assertEqual(sorted(os.listdir(os.path.join(self.root, "peft", "model-1"))), ["metadata.json", "sampler-7"])

  def test_an_adapter_naming_modules_the_base_lacks_is_refused(self) -> None:
    # peft's text layout, which is what the FSDP path emits before its remap.
    # vLLM would load this, match nothing, and serve the base model all run.
    bridge = AdapterBridge(LANGUAGE_KEYS, adapter_names=["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"])
    with self.assertRaisesRegex(RuntimeError, "base checkpoint does not have"):
      self.worker(bridge).write_adapter("model-1", session_label="sampler-7")

  def test_a_refused_adapter_leaves_nothing_for_the_sampler_to_load(self) -> None:
    bridge = AdapterBridge(LANGUAGE_KEYS, adapter_names=["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"])
    with self.assertRaises(RuntimeError):
      self.worker(bridge).write_adapter("model-1", session_label="sampler-7")

    self.assertFalse(os.path.exists(os.path.join(self.root, "peft", "model-1", "sampler-7")))

  def test_an_empty_adapter_is_refused(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "wrote no tensors"):
      self.worker(AdapterBridge(LANGUAGE_KEYS, adapter_names=[])).write_adapter("model-1", session_label="sampler-7")

  def test_full_parameter_training_refuses_to_publish_an_adapter(self) -> None:
    # Rank 0 means there is no adapter. Publishing an empty one would leave the
    # samplers on the base model with nothing in any log to say so.
    with unittest.mock.patch.object(megatron_worker, "MEGATRON_LORA_RANK", 0):
      worker = self.worker(AdapterBridge(LANGUAGE_KEYS))
      self.assertFalse(worker.publishes_sampler_adapter())
      with self.assertRaisesRegex(RuntimeError, "no adapter to publish"):
        worker.write_adapter("model-1", session_label="sampler-7")

  def test_the_alias_ref_resolves_to_the_snapshot(self) -> None:
    # tinker://<id>/sampler_weights/final is returned to the caller as a real
    # ref; without the symlink it names a directory that was never written.
    worker = self.worker(AdapterBridge(LANGUAGE_KEYS))
    worker.write_adapter("model-1", alias="final", session_label="sampler-7")

    alias = os.path.join(self.root, "peft", "model-1", "final")
    self.assertTrue(os.path.islink(alias))
    self.assertTrue(os.path.exists(os.path.join(alias, "adapter_model.safetensors")))

  def test_a_second_publish_does_not_disturb_the_first(self) -> None:
    # Rollouts issued against the previous snapshot are still in flight.
    worker = self.worker(AdapterBridge(LANGUAGE_KEYS))
    worker.write_adapter("model-1", session_label="sampler-7")
    worker.write_adapter("model-1", session_label="sampler-8")

    for label in ("sampler-7", "sampler-8"):
      self.assertTrue(os.path.exists(os.path.join(self.root, "peft", "model-1", label, "adapter_model.safetensors")))

  def test_the_config_carries_the_rank_the_run_trained(self) -> None:
    bridge = AdapterBridge(LANGUAGE_KEYS)
    self.worker(bridge).write_adapter("model-1", session_label="sampler-7")

    self.assertEqual(bridge.saved_config, {"r": 32, "base_model_name_or_path": "google/gemma-4-12B-it"})


class BackendGuardTest(unittest.TestCase):
  def test_pipeline_parallelism_is_refused_at_construction(self) -> None:
    import training.megatron_worker as megatron_worker

    with unittest.mock.patch.object(megatron_worker, "MEGATRON_PP", 2):
      with self.assertRaisesRegex(RuntimeError, "tensor and data parallelism only"):
        MegatronTrainingWorker()

  def test_missing_dependency_names_both_packages(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "megatron-core and megatron-bridge"):
      MegatronTrainingWorker().load_base_model("google/gemma-4-12B-it")


if __name__ == "__main__":
  unittest.main()
