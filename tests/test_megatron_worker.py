"""Megatron backend pieces that do not need megatron installed.

Neither megatron-core nor megatron-bridge is a dependency, so most of the worker
cannot run here. Three parts can, and they are the three that would fail
silently: the data-parallel sharding hooks (wrong shard -> tensor-parallel peers
train on different tokens), the DDP buffer offload's pointer arithmetic (wrong
offset -> parameters alias the wrong weights after a wake_up), and the chunked
logprob head's sequence-first/batch-first bookkeeping (wrong transpose ->
training on shifted labels).
"""

import sys
import types
import unittest
import unittest.mock

import torch

import training.megatron_worker as megatron_worker
from training.megatron_worker import MegatronTrainingWorker, chunked_target_logprobs
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
