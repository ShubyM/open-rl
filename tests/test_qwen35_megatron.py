"""The Qwen3.5 Megatron path, checked against transformers on a tiny model.

Every test states one invariant a training run rests on and checks it on the
fixture in qwen35_fixture against a reference that shares no kernel with the
code under test: transformers' own Qwen3.5 on its pure-torch gated-deltanet
path, in float64. The Megatron side runs in fp32 with TF32 switched off at
the process level (see below), which puts its distance from the fp64 truth at
the fp32 noise floor -- measured 7e-7 on the logits -- so the tolerances here
are set from that floor with headroom, not loosened until green. A mapping
that is wrong by one transposed projection is O(1) away; nothing can hide.

Why the environment variables have to be set before torch initialises CUDA:
Transformer Engine's fp32 GEMMs run at TF32 precision and ignore
torch.backends.cuda.matmul.allow_tf32 entirely (measured: 2.5e-4 relative
error against fp64 with the flag off, same as with it on). Only cuBLAS's own
NVIDIA_TF32_OVERRIDE reaches them, and it is read when the cuBLAS handle is
created. Triton's tl.dot has the same default and its own switch. Neither
matters to a bf16 training run; both matter to an fp32 correctness test.

These need a GPU, megatron-bridge and fla, so they skip everywhere else, except
PassthroughTest, which needs nothing. On the box:

  PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 ~/megatron-probe/.venv/bin/python \\
    -m unittest tests.test_qwen35_megatron -v
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")
# What the worker runs with, so the backward test exercises the same kernel
# selection: fla routes chunk_bwd_dqkwg to tilelang on Hopper.
os.environ.setdefault("FLA_TILELANG", "1")

import importlib.util  # noqa: E402
import multiprocessing as mp  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
import unittest.mock  # noqa: E402

import torch  # noqa: E402

from tests import qwen35_fixture as fixture  # noqa: E402
from training.models import qwen35  # noqa: E402

HAVE_STACK = (
  torch.cuda.is_available()
  and importlib.util.find_spec("megatron.bridge") is not None
  and importlib.util.find_spec("fla.ops") is not None
)
FIXTURE_DIR = os.path.join(tempfile.gettempdir(), "qwen35-tiny-fixture")

# fp32 against fp64 on this model measures ~7e-7 relative at the logits and
# ~1e-7 per layer; 1e-5 is an order of magnitude of headroom over that and
# five under any wiring error.
FP32_TOLERANCE = 1e-5


def relative_error(actual: torch.Tensor, expected: torch.Tensor, scale: torch.Tensor | None = None) -> float:
  """max|actual - expected| over max|expected| (or over max|scale| if given).

  A shard of a tensor-parallel gradient is judged against the whole tensor's
  scale, not its own: rank 1's slice of A_log carries heads whose true
  gradient is ~1e-8 next to a head at ~1e-5 on rank 0, and bf16 noise on the
  small heads is a large fraction of *them* while being nothing at all.
  """
  reference = expected if scale is None else scale
  return ((actual.double() - expected.double()).abs().max() / reference.double().abs().max().clamp_min(1e-30)).item()


@unittest.skipUnless(HAVE_STACK, "needs a GPU with megatron-bridge and fla-core")
class ForwardFidelityTest(unittest.TestCase):
  """Invariant 1: the Megatron model computes the same function as the HF one.

  Everything downstream -- gradients, the adapter vLLM loads, the checkpoint an
  eval job reads -- assumes this. It covers the text-only bridge registration,
  TextConfigView, the fused in_proj mapping, zero-centred RMSNorm gammas, and
  plain rope standing in for interleaved mrope on text (all three position
  streams are equal for text tokens, so the two agree exactly, or this fails).
  """

  @classmethod
  def setUpClass(cls) -> None:
    cls.path = fixture.write_tiny_checkpoint(FIXTURE_DIR)
    cls.reference = fixture.reference_model(cls.path)
    cls.bridge, cls.chunks = fixture.megatron_model(cls.path)
    cls.model = cls.chunks[0]
    cls.model.eval()
    torch.manual_seed(7)
    # 100 tokens: one full 64-token gated-deltanet chunk and one partial, so
    # both the intra-chunk and the cross-chunk recurrence are exercised.
    cls.input_ids = torch.randint(0, fixture.VOCAB, (1, 100), device="cuda")

  def test_logits_match_the_reference_to_fp32_precision(self) -> None:
    with torch.no_grad():
      expected = self.reference(input_ids=self.input_ids).logits
      actual = fixture.megatron_logits(self.model, self.input_ids)
    self.assertEqual(actual.shape, expected.shape)
    self.assertLess(relative_error(actual, expected), FP32_TOLERANCE)

  def test_the_vision_tower_and_mtp_head_are_not_in_the_model(self) -> None:
    names = [n for n, _ in self.model.named_parameters()]
    self.assertFalse([n for n in names if "visual" in n or "mtp" in n], names[:5])
    language = sum(
      p.numel() for n, p in self.reference.named_parameters() if n.startswith("model.language_model.") or n == "lm_head.weight"
    )
    self.assertEqual(sum(p.numel() for p in self.model.parameters()), language)


@unittest.skipUnless(HAVE_STACK, "needs a GPU with megatron-bridge and fla-core")
class BackwardFidelityTest(unittest.TestCase):
  """Invariant 2: the Megatron backward computes the same gradients as autograd.

  This is the path no run in the repo had taken before run42: fla's
  gated-deltanet backward, routed to tilelang, under full recompute. The
  reference is fp64 autograd through transformers' pure-torch delta rule, and
  the parameters compared map one-to-one between the models and between them
  cover every gradient path (see identity_mapped_parameters).

  It comes in two parts because an fp32 backward through gated-deltanet is
  impossible on this hardware: fla refuses its Triton backward on Hopper for
  any dtype (chunk_o.py, issue #640) and tilelang, the only other backend,
  has no fp32 kernel (tvm raises "Get different layout for b_dq"). So the
  attention layer, final norm and head are checked at fp32 precision --
  torch.autograd.grad walks only the subgraph the requested parameters need,
  and the last layer's parameters need no gated-deltanet backward -- and
  everything, including the embedding, whose gradient exists only by way of
  all four layers, is checked in bf16, the dtype the run uses, against a
  tolerance calibrated by the reference's own bf16 error rather than picked.

  Only the bf16 half runs under the worker's full recompute. Megatron's
  checkpointing is the reentrant kind: parameters used inside a checkpointed
  layer are not recorded in the graph at forward time, so autograd.grad cannot
  reach them and only loss.backward() -- which runs every layer's backward --
  populates their .grad.
  """

  @classmethod
  def setUpClass(cls) -> None:
    cls.path = fixture.write_tiny_checkpoint(FIXTURE_DIR)
    torch.manual_seed(11)
    cls.input_ids = torch.randint(0, fixture.VOCAB, (1, 100), device="cuda")

  def megatron_gradients(self, dtype: torch.dtype, pairs: list[tuple[str, str]], recompute: bool) -> list[torch.Tensor]:
    _, chunks = fixture.megatron_model(self.path, dtype=dtype, recompute=recompute)
    model = chunks[0]
    model.train()
    params = fixture.parameters_by_name(model)
    wanted = [params[m] for m, _ in pairs]
    loss = fixture.next_token_nll(fixture.megatron_logits(model, self.input_ids), self.input_ids)
    if not recompute:
      return [g.detach() for g in torch.autograd.grad(loss, wanted)]
    loss.backward()
    for (m, _), p in zip(pairs, wanted):
      self.assertIsNotNone(p.grad, f"{m} received no gradient")
    return [p.grad.detach() for p in wanted]

  def reference_gradients(self, dtype: torch.dtype, pairs: list[tuple[str, str]]) -> list[torch.Tensor]:
    model = fixture.reference_model(self.path, dtype=dtype)
    model.train()
    params = dict(model.named_parameters())
    loss = fixture.next_token_nll(model(input_ids=self.input_ids).logits, self.input_ids)
    return [g.detach() for g in torch.autograd.grad(loss, [params[h] for _, h in pairs])]

  def test_attention_norm_and_head_gradients_match_at_fp32_precision(self) -> None:
    last = len(fixture.LAYER_TYPES) - 1
    self.assertEqual(fixture.LAYER_TYPES[last], "full_attention")
    pairs = [
      (m, h) for m, h in fixture.identity_mapped_parameters()
      if f"layers.{last}." in m or "final_layernorm" in m or m == "output_layer.weight"
    ]
    self.assertGreaterEqual(len(pairs), 6)
    actual = self.megatron_gradients(torch.float32, pairs, recompute=False)
    expected = self.reference_gradients(torch.float64, pairs)
    errors = {m: relative_error(a[: e.shape[0]], e) for (m, _), a, e in zip(pairs, actual, expected)}
    worst = max(errors, key=errors.get)
    self.assertLess(errors[worst], FP32_TOLERANCE, f"worst {worst} at {errors[worst]:.2e}; all: {errors}")

  def test_every_gradient_matches_within_bf16_noise_including_gated_deltanet(self) -> None:
    pairs = fixture.identity_mapped_parameters()
    truth = self.reference_gradients(torch.float64, pairs)
    yardstick = self.reference_gradients(torch.bfloat16, pairs)
    actual = self.megatron_gradients(torch.bfloat16, pairs, recompute=True)
    report, failures = [], []
    for (m, _), t, y, a in zip(pairs, truth, yardstick, actual):
      self.assertEqual(tuple(a[: t.shape[0]].shape), tuple(t.shape), m)
      own = relative_error(y, t)
      ours = relative_error(a[: t.shape[0]], t)
      # Four times what transformers' bf16 manages on the same tensor, with a
      # floor of two bf16 roundings (unit roundoff 2^-8) for tensors where the
      # reference happens to land nearly exactly.
      allowed = max(4 * own, 2 * 2**-8)
      report.append(f"{m}: ours {ours:.1e} vs reference-bf16 {own:.1e} (allowed {allowed:.1e})")
      if not ours < allowed:
        failures.append(report[-1])
    self.assertFalse(failures, "\n" + "\n".join(failures) + "\n--- all ---\n" + "\n".join(report))


@unittest.skipUnless(HAVE_STACK, "needs a GPU with megatron-bridge and fla-core")
class AdapterExportTest(unittest.TestCase):
  """Invariant 3: base model + the exported adapter is the model being trained.

  This is what the samplers run. write_adapter calls save_hf_adapter, vLLM
  applies the resulting PEFT directory to the base checkpoint, and every
  rollout comes from that composition. If it differs from the Megatron model
  with its adapters, training optimises one policy while sampling another.
  Megatron fuses the gated-deltanet checkpoint's four in_proj_* into one
  in_proj, so the export has to split each adapter back into four that vLLM
  can pack -- the one place the adapter format does real work.
  """

  @classmethod
  def setUpClass(cls) -> None:
    cls.path = fixture.write_tiny_checkpoint(FIXTURE_DIR)
    cls.lora = fixture.lora_config()
    cls.bridge, cls.chunks = fixture.megatron_model(cls.path, lora=cls.lora)
    cls.model = cls.chunks[0]
    cls.model.eval()
    cls.adapters_set = fixture.randomize_adapters(cls.model)
    torch.manual_seed(5)
    cls.input_ids = torch.randint(0, fixture.VOCAB, (1, 100), device="cuda")
    with torch.no_grad():
      cls.with_lora = fixture.megatron_logits(cls.model, cls.input_ids)
    cls.adapter_dir = tempfile.mkdtemp(prefix="qwen35-adapter-")
    # The worker's call, argument for argument (write_adapter).
    cls.bridge.save_hf_adapter(cls.chunks, cls.adapter_dir, peft_config=cls.lora, base_model_name_or_path=cls.path, show_progress=False)

  def test_the_adapter_is_not_a_no_op(self) -> None:
    # LoRA zero-initialises B; without randomize_adapters every test below
    # would pass against the plain base model.
    self.assertGreater(self.adapters_set, 0)
    with torch.no_grad():
      _, base_chunks = fixture.megatron_model(self.path)
      base = fixture.megatron_logits(base_chunks[0].eval(), self.input_ids)
    self.assertGreater(relative_error(self.with_lora, base), 1e-2)

  def test_base_plus_exported_adapter_matches_megatron_with_lora(self) -> None:
    from peft import PeftModel

    reference = PeftModel.from_pretrained(fixture.reference_model(self.path), self.adapter_dir)
    reference.eval()
    with torch.no_grad():
      expected = reference(input_ids=self.input_ids).logits
    self.assertLess(relative_error(self.with_lora, expected), FP32_TOLERANCE)

  def test_every_adapter_tensor_names_a_module_of_the_base_checkpoint(self) -> None:
    # The worker's own guard, run against this export. Names vLLM cannot match
    # are a 200 OK and no adapter, which is the failure mode this exists for.
    from training.megatron_worker import MegatronTrainingWorker

    MegatronTrainingWorker.assert_adapter_targets_base(types.SimpleNamespace(bridge=self.bridge), self.adapter_dir)

  def test_the_gated_deltanet_adapters_are_split_the_way_vllm_packs_them(self) -> None:
    tensors = fixture.saved_tensors(self.adapter_dir)
    gdn = [i for i, kind in enumerate(fixture.LAYER_TYPES) if kind == "linear_attention"]
    for i in gdn:
      prefix = f"base_model.model.model.language_model.layers.{i}.linear_attn."
      siblings = [f"{prefix}in_proj_{s}" for s in ("qkv", "z", "b", "a")]
      for module in siblings + [prefix + "out_proj"]:
        self.assertIn(module + ".lora_A.weight", tensors)
        self.assertIn(module + ".lora_B.weight", tensors)
      # One fused in_proj adapter becomes four: they share A and slice B, which
      # is what makes the split equal to the fused product.
      first = tensors[siblings[0] + ".lora_A.weight"]
      for module in siblings[1:]:
        self.assertTrue(torch.equal(tensors[module + ".lora_A.weight"], first), module)


def assert_peft_initialisation(test: unittest.TestCase, adapter_dir: str) -> None:
  """Every exported lora_A is drawn as PEFT draws it, and B is zero.

  PEFT's kaiming_uniform(a=sqrt(5)) on a [r, in] matrix is U(-1/sqrt(in),
  1/sqrt(in)), std 1/sqrt(3 in). The fixture's smallest adapter has 8 x 64
  entries, for which the sample std scatters by about 3%; 20% is far outside
  that and well inside the sqrt(6) and sqrt(12) factors being ruled out.
  """
  import json

  tensors = fixture.saved_tensors(adapter_dir)
  with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
    config = json.load(f)
  test.assertEqual(config["lora_alpha"], fixture.LORA_ALPHA)
  test.assertEqual(config["r"], fixture.LORA_DIM)
  checked = 0
  for name, t in sorted(tensors.items()):
    if name.endswith("lora_B.weight"):
      test.assertEqual(float(t.abs().max()), 0.0, name)
      continue
    test.assertTrue(name.endswith("lora_A.weight"), name)
    fan_in = t.shape[1]
    expected = 1.0 / math.sqrt(3 * fan_in)
    ratio = float(t.float().std()) / expected
    test.assertLess(abs(ratio - 1.0), 0.2, f"{name}: std is {ratio:.2f}x PEFT's for fan_in {fan_in}")
    test.assertLessEqual(float(t.abs().max()), 1.0 / math.sqrt(fan_in) * 1.001, f"{name}: outside the uniform bound")
    checked += 1
  test.assertGreater(checked, 0)


@unittest.skipUnless(HAVE_STACK, "needs a GPU with megatron-bridge and fla-core")
class AdapterInitialisationTest(unittest.TestCase):
  """Invariant 8: a fresh Megatron adapter is a fresh PEFT adapter.

  Same rank, same lr, different backend was not the same experiment. The
  bridge draws A xavier-normal on the local TP shard where PEFT draws
  kaiming-uniform on the full fan-in, and its alpha default is 32 where the
  FSDP worker's is 16. B starts at zero on both, so the first steps move W by
  scale * |A| * lr: a factor 2 * sqrt(6) ~ 4.9 on the effective learning rate,
  which is the 5x gradient-norm gap between run42 and run19 and the reason
  every Megatron lr on record diverged where the FSDP one held.
  """

  @classmethod
  def setUpClass(cls) -> None:
    cls.path = fixture.write_tiny_checkpoint(FIXTURE_DIR)
    cls.lora = fixture.lora_config()
    cls.bridge, cls.chunks = fixture.megatron_model(cls.path, lora=cls.lora)
    cls.adapter_dir = tempfile.mkdtemp(prefix="qwen35-fresh-adapter-")
    cls.bridge.save_hf_adapter(cls.chunks, cls.adapter_dir, peft_config=cls.lora, base_model_name_or_path=cls.path, show_progress=False)

  def test_a_fresh_adapter_starts_where_pefts_would(self) -> None:
    assert_peft_initialisation(self, self.adapter_dir)

  def test_the_formula_is_pefts_initialiser(self) -> None:
    # Pins the analytic target used above to torch's own kaiming_uniform.
    w = torch.empty(64, 1024)
    torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    self.assertLess(abs(float(w.std()) / (1.0 / math.sqrt(3 * 1024)) - 1.0), 0.05)

  def test_the_alpha_default_is_the_fsdp_workers(self) -> None:
    from training import megatron_worker
    from training.lora_trainer_worker import LoraConfig

    self.assertEqual(megatron_worker.MEGATRON_LORA_ALPHA, LoraConfig.model_fields["lora_alpha"].default)


@unittest.skipUnless(HAVE_STACK, "needs a GPU with megatron-bridge and fla-core")
class CheckpointRoundTripTest(unittest.TestCase):
  """Invariant 4: save_checkpoint writes the model that was trained, whole.

  This is the invariant run42 broke. The save is what an eval job loads and
  what load_from_state resumes from, so it has to contain every tensor the
  base checkpoint had -- vision tower included, because the saved config.json
  still advertises one; the MTP head is the one deliberate exception, dropped
  by the bridge itself -- with the adapters merged into exactly the weights
  they adapted and nothing else changed.
  """

  @classmethod
  def setUpClass(cls) -> None:
    cls.path = fixture.write_tiny_checkpoint(FIXTURE_DIR)
    cls.base = fixture.saved_tensors(cls.path)
    cls.lora = fixture.lora_config()
    cls.bridge, cls.chunks = fixture.megatron_model(cls.path, lora=cls.lora)
    cls.model = cls.chunks[0]
    cls.model.eval()
    fixture.randomize_adapters(cls.model)
    torch.manual_seed(5)
    cls.input_ids = torch.randint(0, fixture.VOCAB, (1, 100), device="cuda")
    with torch.no_grad():
      cls.with_lora = fixture.megatron_logits(cls.model, cls.input_ids)
    cls.adapter_dir = tempfile.mkdtemp(prefix="qwen35-adapter-")
    cls.bridge.save_hf_adapter(cls.chunks, cls.adapter_dir, peft_config=cls.lora, base_model_name_or_path=cls.path, show_progress=False)
    cls.save_dir = tempfile.mkdtemp(prefix="qwen35-save-")
    # save_checkpoint's sequence: the modelopt guard, then the save inside the
    # context manager. Without the guard the bridge's unguarded import kills
    # the save before it writes anything, exactly as it did in run34.
    from training.megatron_worker import ensure_modelopt_importable

    ensure_modelopt_importable()
    with qwen35.multimodal_export_passthrough():
      cls.bridge.save_hf_pretrained(cls.chunks, cls.save_dir)
    cls.saved = fixture.saved_tensors(cls.save_dir)

  def test_without_the_passthrough_the_save_fails_the_way_run42_did(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "not written"):
      self.bridge.save_hf_pretrained(self.chunks, tempfile.mkdtemp(prefix="qwen35-nopass-"))

  def expected_written(self) -> list[str]:
    # Everything the base had except the MTP head, which the bridge drops from
    # any export whose provider has no MTP layers (see WRITER_IGNORED_PREFIXES).
    return sorted(n for n in self.base if not n.startswith(qwen35.WRITER_IGNORED_PREFIXES))

  def test_every_tensor_of_the_base_checkpoint_is_written(self) -> None:
    self.assertEqual(sorted(self.saved), self.expected_written())
    self.assertLess(len(self.saved), len(self.base))  # the MTP head, and only it, is absent

  def test_passthrough_tensors_are_byte_identical_to_the_base(self) -> None:
    names = [n for n in self.base if n.startswith(qwen35.MULTIMODAL_SOURCE_PREFIXES)]
    self.assertGreater(len(names), 0)
    for name in names:
      self.assertEqual(self.saved[name].dtype, self.base[name].dtype, name)
      self.assertTrue(torch.equal(self.saved[name], self.base[name]), name)

  def untouched(self) -> list[str]:
    # Every LoRA target on this architecture is a *_proj; the norms, the
    # embedding, the head, conv1d, A_log and dt_bias are not adapted and must
    # come back as they went in.
    skip = qwen35.MULTIMODAL_SOURCE_PREFIXES + qwen35.WRITER_IGNORED_PREFIXES
    return [n for n in self.base if not n.startswith(skip) and "proj" not in n]

  def test_untouched_language_tensors_survive_the_round_trip(self) -> None:
    names = self.untouched()
    self.assertGreater(len(names), 0)
    for name in names:
      # Not bit-identical for the norms: Megatron stores gamma - 1 and the
      # export adds the 1 back, which is one fp32 rounding.
      torch.testing.assert_close(self.saved[name].float(), self.base[name].float(), rtol=0, atol=1e-6, msg=name)

  def test_adapted_tensors_equal_base_plus_scaled_ba(self) -> None:
    adapter = fixture.saved_tensors(self.adapter_dir)
    scale = fixture.LORA_ALPHA / fixture.LORA_DIM
    checked = 0
    for key, a in adapter.items():
      if not key.endswith(".lora_A.weight"):
        continue
      module = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
      b = adapter[key.replace(".lora_A.", ".lora_B.")]
      expected = self.base[module + ".weight"].double() + scale * (b.double() @ a.double())
      self.assertLess(relative_error(self.saved[module + ".weight"], expected), FP32_TOLERANCE, module)
      checked += 1
    self.assertGreater(checked, 0)
    vision = sum(n.startswith(qwen35.MULTIMODAL_SOURCE_PREFIXES) for n in self.base)
    self.assertEqual(checked + len(self.untouched()) + vision, len(self.expected_written()))

  def test_the_saved_checkpoint_computes_the_adapted_model_in_transformers(self) -> None:
    reference = fixture.reference_model(self.save_dir)
    with torch.no_grad():
      expected = reference(input_ids=self.input_ids).logits
    self.assertLess(relative_error(self.with_lora, expected), FP32_TOLERANCE)

  def test_the_saved_checkpoint_loads_back_through_the_bridge(self) -> None:
    # load_from_state's path: our own output read by the same bridge.
    _, chunks = fixture.megatron_model(self.save_dir)
    with torch.no_grad():
      again = fixture.megatron_logits(chunks[0].eval(), self.input_ids)
    self.assertLess(relative_error(again, self.with_lora), FP32_TOLERANCE)


def rank_shard(local: torch.Tensor, full: torch.Tensor, param: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
  """The slice of a full tensor that this rank's parameter shard corresponds to.

  Megatron tags sharded parameters with partition_dim; replicated ones carry
  none and compare whole. The gated-deltanet A_log and dt_bias are sharded by
  value head without the tag, so a shape mismatch with no tag means dim 0.
  """
  if tuple(local.shape) == tuple(full.shape):
    return full
  dim = getattr(param, "partition_dim", -1)
  if dim is None or dim < 0:
    dim = 0
  return full.chunk(world_size, dim)[rank]


def tensor_parallel_worker(rank: int, world_size: int, port: int, results) -> None:
  """One TP rank of TensorParallelTest; the checks mirror the single-GPU tests."""
  os.environ.update(
    {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}
  )
  import torch.distributed as dist

  torch.cuda.set_device(rank)
  dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
  out: dict[str, object] = {}
  try:
    path = FIXTURE_DIR
    torch.manual_seed(13)
    input_ids = torch.randint(0, fixture.VOCAB, (1, 100), device="cuda")

    _, chunks = fixture.megatron_model(path)
    out["tp"] = chunks[0].config.tensor_model_parallel_size
    out["sequence_parallel"] = chunks[0].config.sequence_parallel
    truth_model = fixture.reference_model(path)
    with torch.no_grad():
      out["forward"] = relative_error(fixture.megatron_logits(chunks[0].eval(), input_ids), truth_model(input_ids=input_ids).logits)

    pairs = fixture.identity_mapped_parameters()

    def reference_grads(dtype):
      model = fixture.reference_model(path, dtype=dtype)
      model.train()
      params = dict(model.named_parameters())
      loss = fixture.next_token_nll(model(input_ids=input_ids).logits, input_ids)
      return [g.detach() for g in torch.autograd.grad(loss, [params[h] for _, h in pairs])]

    truth, yardstick = reference_grads(torch.float64), reference_grads(torch.bfloat16)
    _, chunks = fixture.megatron_model(path, dtype=torch.bfloat16, recompute=True)
    model = chunks[0]
    model.train()
    fixture.next_token_nll(fixture.megatron_logits(model, input_ids), input_ids).backward()
    params = fixture.parameters_by_name(model)
    for (m, _), t, y in zip(pairs, truth, yardstick):
      p = params[m]
      if p.grad is None:
        out[f"grad:{m}"] = ("missing", 0.0)
        continue
      g = p.grad.detach().clone()
      if "q_layernorm" in m or "k_layernorm" in m:
        # Replicated, but applied to this rank's heads only, so the local
        # gradient is a partial sum. finalize_model_grads sum-reduces exactly
        # these two across TP (megatron/core/distributed/finalize_model_grads.py,
        # the qk_layernorm branch); the worker's LoRA runs never train them.
        dist.all_reduce(g)
      t_s = rank_shard(g, t, p, rank, world_size)
      y_s = rank_shard(g, y, p, rank, world_size)
      g = g[: t_s.shape[0]]
      out[f"grad:{m}"] = (relative_error(g, t_s, scale=t), relative_error(y_s, t_s, scale=t))
  except Exception as e:  # noqa: BLE001 -- reported to the parent, which fails the test
    out["error"] = f"{type(e).__name__}: {e}"
  finally:
    results[rank] = out
    dist.destroy_process_group()


@unittest.skipUnless(HAVE_STACK and torch.cuda.device_count() >= 2, "needs two GPUs with megatron-bridge and fla-core")
class TensorParallelTest(unittest.TestCase):
  """Invariant 6: TP=2 with sequence parallelism off is the same model.

  run42 runs TP=2 because 163,840 tokens do not fit one card, and it runs
  with sequence parallelism off because gated-deltanet asserts against it.
  Neither topology had ever been exercised for this architecture. Two ranks
  are spawned, each builds its shard, and each checks its own view against
  the same fp64 reference the single-GPU tests use: gathered logits at fp32
  precision, and the bf16 gradients of its parameter shards against the
  matching slice of the reference gradient, within the calibrated tolerance.
  """

  WORLD_SIZE = 2

  def test_forward_and_backward_match_the_reference_on_every_rank(self) -> None:
    fixture.write_tiny_checkpoint(FIXTURE_DIR)
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
      results = manager.dict()
      procs = [ctx.Process(target=tensor_parallel_worker, args=(r, self.WORLD_SIZE, 29533, results)) for r in range(self.WORLD_SIZE)]
      for p in procs:
        p.start()
      for p in procs:
        p.join(timeout=900)
      for r, p in enumerate(procs):
        self.assertEqual(p.exitcode, 0, f"rank {r} exited {p.exitcode}: {dict(results).get(r)}")
      results = {r: dict(v) for r, v in results.items()}

    failures = []
    for r in range(self.WORLD_SIZE):
      out = results[r]
      self.assertNotIn("error", out, f"rank {r}: {out.get('error')}")
      self.assertEqual(out["tp"], self.WORLD_SIZE)
      self.assertFalse(out["sequence_parallel"])
      if not out["forward"] < FP32_TOLERANCE:
        failures.append(f"rank {r} forward {out['forward']:.1e}")
      for key, value in out.items():
        if not key.startswith("grad:"):
          continue
        ours, own = value
        if ours == "missing" or not ours < max(4 * own, 2 * 2**-8):
          failures.append(f"rank {r} {key}: ours {ours} vs reference-bf16 {own:.1e}")
    self.assertFalse(failures, "\n".join(failures))


def fill_adapters(model) -> int:
  """Give every adapter tensor a nonzero, bf16-exact value that depends on its name.

  bf16-exact so an fp32 model and a bf16 model filled this way hold the same
  adapter to the bit, and deterministic so every rank of a tensor-parallel
  model fills the same way. Ranks of a sharded tensor end up holding equal
  shards, which is a perfectly good adapter; nothing here needs to know how
  the tensor is sharded. Returns how many tensors were set.
  """
  import zlib

  count = 0
  with torch.no_grad():
    for name, p in fixture.parameters_by_name(model).items():
      if ".adapter." not in name:
        continue
      g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
      p.copy_((torch.randn(tuple(p.shape), generator=g) * 0.05).to(torch.bfloat16).to(p.device, p.dtype))
      count += 1
  return count


# HF modules that share one Megatron adapter, because Megatron fuses their
# base projections. One A feeds all of them, so its gradient is the sum of the
# per-module gradients a framework with a separate A per module computes.
FUSED_SIBLINGS = (("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"), ("gate_proj", "up_proj"), ("q_proj", "k_proj", "v_proj"))


def sibling_group(module: str) -> tuple[str, ...]:
  leaf = module.rsplit(".", 1)[-1]
  for group in FUSED_SIBLINGS:
    if leaf in group:
      return tuple(module[: -len(leaf)] + other for other in group)
  return (module,)


def peft_gradients(base, adapter_dir: str, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """Logits and adapter gradients of an HF model wearing the exported adapter."""
  from peft import PeftModel

  # Frozen unless asked: from_pretrained defaults to is_trainable=False.
  model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
  model.train()
  logits = model(input_ids=input_ids).logits
  fixture.next_token_nll(logits, input_ids).backward()
  grads = {
    n.replace(".default.", "."): p.grad.detach().double().cpu()
    for n, p in model.named_parameters()
    if p.grad is not None and ".lora_" in n
  }
  return logits.detach(), grads


def adapter_worker(rank: int, world_size: int, port: int, dirs: dict[str, str], results) -> None:
  os.environ.update(
    {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}
  )
  import torch.distributed as dist

  torch.cuda.set_device(rank)
  dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
  out: dict[str, object] = {}
  try:
    input_ids = torch.randint(0, fixture.VOCAB, (1, 100), generator=torch.Generator().manual_seed(17)).cuda()
    lora = fixture.lora_config()

    # fp32 for the forward and the adapter's values; export is write_adapter's call.
    bridge, chunks = fixture.megatron_model(FIXTURE_DIR, lora=lora)
    # The fresh adapter first, before fill_adapters overwrites the initialisation.
    bridge.save_hf_adapter(chunks, dirs["init"], peft_config=lora, base_model_name_or_path=FIXTURE_DIR, show_progress=False)
    from megatron.bridge.peft.utils import ParallelLinearAdapter

    first = next(m for m in chunks[0].modules() if isinstance(m, ParallelLinearAdapter))
    local = first.linear_in.weight.detach().float().contiguous()
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    out["ranks_drew_alike"] = bool(torch.equal(gathered[0], gathered[1]))
    fill_adapters(chunks[0])
    with torch.no_grad():
      out["logits"] = fixture.megatron_logits(chunks[0].eval(), input_ids).float().cpu()
    bridge.save_hf_adapter(chunks, dirs["values"], peft_config=lora, base_model_name_or_path=FIXTURE_DIR, show_progress=False)

    # bf16 under recompute for the backward, as the run does; the adapter is
    # bit-identical to the fp32 one. The gradients leave through the same
    # export, by standing in for the parameters: the export gathers the
    # tensor-parallel shards and splits the fused projections exactly as it
    # does for the values, which is the only layout both sides agree on.
    bridge, chunks = fixture.megatron_model(FIXTURE_DIR, dtype=torch.bfloat16, lora=lora, recompute=True)
    model = chunks[0]
    model.train()
    fill_adapters(model)
    fixture.next_token_nll(fixture.megatron_logits(model, input_ids), input_ids).backward()
    adapters = {n: p for n, p in fixture.parameters_by_name(model).items() if ".adapter." in n}
    missing = [n for n, p in adapters.items() if p.grad is None]
    out["no_gradient"] = missing
    with torch.no_grad():
      for p in adapters.values():
        if p.grad is not None:
          p.copy_(p.grad)
    bridge.save_hf_adapter(chunks, dirs["grads"], peft_config=lora, base_model_name_or_path=FIXTURE_DIR, show_progress=False)
  except Exception as e:  # noqa: BLE001
    out["error"] = f"{type(e).__name__}: {e}"
  finally:
    results[rank] = out
    dist.destroy_process_group()


@unittest.skipUnless(HAVE_STACK and torch.cuda.device_count() >= 2, "needs two GPUs with megatron-bridge and fla-core")
class AdapterTensorParallelTest(unittest.TestCase):
  """Invariant 7: at TP=2, the adapter vLLM gets and the gradient the optimizer
  gets are both the ones an independent implementation computes.

  run42 trains at TP=2 and publishes through write_adapter, and neither had
  been exercised for this architecture beyond a single GPU. The adapter is
  exported from a two-rank model through save_hf_adapter, loaded onto the fp64
  transformers model with peft, and compared twice: logits, at fp32 precision,
  which checks the tensor-parallel gather and the fused-projection split of
  the values; and the adapter gradients, which are exported through the same
  path and checked against peft's autograd within the calibrated bf16
  tolerance. Megatron shares one A across the HF modules it fuses, so by the
  chain rule its exported A-gradient must equal the sum of peft's per-module
  A-gradients, and the exported copies under each sibling must be identical.
  """

  WORLD_SIZE = 2

  @classmethod
  def setUpClass(cls) -> None:
    cls.path = fixture.write_tiny_checkpoint(FIXTURE_DIR)
    cls.dirs = {k: tempfile.mkdtemp(prefix=f"qwen35-tp-adapter-{k}-") for k in ("init", "values", "grads")}
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
      results = manager.dict()
      procs = [ctx.Process(target=adapter_worker, args=(r, cls.WORLD_SIZE, 29551, cls.dirs, results)) for r in range(cls.WORLD_SIZE)]
      for p in procs:
        p.start()
      for p in procs:
        p.join(timeout=900)
      cls.exit_codes = [p.exitcode for p in procs]
      cls.results = {r: dict(v) for r, v in results.items()}
    cls.input_ids = torch.randint(0, fixture.VOCAB, (1, 100), generator=torch.Generator().manual_seed(17)).cuda()

  def setUp(self) -> None:
    self.assertEqual(self.exit_codes, [0] * self.WORLD_SIZE, self.results)
    for r in range(self.WORLD_SIZE):
      self.assertNotIn("error", self.results[r], self.results[r].get("error"))
      self.assertEqual(self.results[r]["no_gradient"], [])

  def test_a_fresh_adapter_starts_where_pefts_would_on_two_ranks(self) -> None:
    # The row-parallel targets are the ones the bridge draws from the local
    # shard's fan-in; at TP=2 that is sqrt(2) too wide without the re-draw.
    assert_peft_initialisation(self, self.dirs["init"])

  def test_the_two_ranks_draw_different_adapter_shards(self) -> None:
    # Identical draws would collapse a rank-r adapter to rank r/TP.
    self.assertFalse(self.results[0]["ranks_drew_alike"])

  def test_the_exported_adapter_reproduces_the_two_rank_model(self) -> None:
    logits, _ = peft_gradients(fixture.reference_model(self.path), self.dirs["values"], self.input_ids)
    self.assertLess(relative_error(self.results[0]["logits"].cuda(), logits), FP32_TOLERANCE)

  def test_the_exported_gradient_is_pefts_gradient(self) -> None:
    _, truth = peft_gradients(fixture.reference_model(self.path), self.dirs["values"], self.input_ids)
    _, yardstick = peft_gradients(fixture.reference_model(self.path, dtype=torch.bfloat16), self.dirs["values"], self.input_ids)
    exported = {k: v.double() for k, v in fixture.saved_tensors(self.dirs["grads"]).items()}
    self.assertEqual(sorted(exported), sorted(truth))

    failures, report = [], []
    for key, ours in exported.items():
      module, kind = key.rsplit(".lora_", 1)
      if kind.startswith("A"):
        siblings = [f"{m}.lora_A.weight" for m in sibling_group(module)]
        for other in siblings:
          self.assertTrue(torch.equal(exported[other], ours), f"{key} and {other} should be one Megatron tensor")
        expected = sum(truth[k] for k in siblings)
        own = relative_error(sum(yardstick[k] for k in siblings), expected)
      else:
        expected, own = truth[key], relative_error(yardstick[key], truth[key])
      err = relative_error(ours, expected)
      allowed = max(4 * own, 2 * 2**-8)
      report.append(f"{key}: ours {err:.1e} vs peft-bf16 {own:.1e} (allowed {allowed:.1e})")
      if not err < allowed:
        failures.append(report[-1])
    self.assertFalse(failures, "\n" + "\n".join(failures) + "\n--- all ---\n" + "\n".join(report))


def nine_b_snapshot() -> str | None:
  import glob

  hits = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*/model.safetensors.index.json"))
  return os.path.dirname(hits[0]) if hits else None


def nine_b_worker(rank: int, world_size: int, port: int, out_dir: str, results) -> None:
  os.environ.update(
    {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)}
  )
  import torch.distributed as dist

  from training.megatron_worker import ensure_modelopt_importable

  torch.cuda.set_device(rank)
  dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
  try:
    bridge, chunks = fixture.megatron_model(nine_b_snapshot(), dtype=torch.bfloat16, lora=fixture.lora_config())
    fixture.randomize_adapters(chunks[0])
    ensure_modelopt_importable()
    with qwen35.multimodal_export_passthrough():
      bridge.save_hf_pretrained(chunks, out_dir)
    results[rank] = "ok"
  except Exception as e:  # noqa: BLE001
    results[rank] = f"{type(e).__name__}: {e}"
  finally:
    dist.destroy_process_group()


@unittest.skipUnless(
  HAVE_STACK and torch.cuda.device_count() >= 2 and nine_b_snapshot() and os.getenv("QWEN35_9B_SAVE_TEST") == "1",
  "opt in with QWEN35_9B_SAVE_TEST=1; needs two GPUs, ~20 GB of disk and the 9B in the HF cache",
)
class NineBillionSaveTest(unittest.TestCase):
  """run42's failure, at run42's scale: save the real 9B at TP=2 and audit it.

  The fixture proves the mechanism; this proves it on the checkpoint that
  actually died -- four real shards, the real interleave, the real MTP prefix
  logic on the real config -- so a relaunch does not rediscover it at step 10.
  Slow (minutes) and heavy, so opt-in.
  """

  @classmethod
  def setUpClass(cls) -> None:
    import shutil

    cls.out_dir = tempfile.mkdtemp(prefix="qwen35-9b-save-", dir=os.path.expanduser("~"))
    cls.addClassCleanup(shutil.rmtree, cls.out_dir, ignore_errors=True)
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
      results = manager.dict()
      procs = [ctx.Process(target=nine_b_worker, args=(r, 2, 29571, cls.out_dir, results)) for r in range(2)]
      for p in procs:
        p.start()
      for p in procs:
        p.join(timeout=2400)
      cls.results = dict(results)
      cls.exit_codes = [p.exitcode for p in procs]

  def test_every_tensor_but_the_mtp_head_is_written_and_vision_is_untouched(self) -> None:
    import json

    from safetensors import safe_open

    self.assertEqual(self.exit_codes, [0, 0], self.results)
    self.assertEqual(self.results, {0: "ok", 1: "ok"})
    base_dir = nine_b_snapshot()
    with open(os.path.join(base_dir, "model.safetensors.index.json")) as f:
      base_map = json.load(f)["weight_map"]
    expected = sorted(k for k in base_map if not k.startswith(qwen35.WRITER_IGNORED_PREFIXES))
    self.assertEqual(len(base_map), 775)
    self.assertEqual(len(expected), 760)

    saved_map: dict[str, str] = {}
    for name in os.listdir(self.out_dir):
      if name.endswith(".safetensors"):
        with safe_open(os.path.join(self.out_dir, name), framework="pt") as h:
          saved_map.update({k: name for k in h.keys()})
    self.assertEqual(sorted(saved_map), expected)

    vision = [k for k in expected if k.startswith(qwen35.MULTIMODAL_SOURCE_PREFIXES)]
    self.assertEqual(len(vision), 333)
    handles: dict[str, object] = {}

    def read(directory, mapping, key):
      path = os.path.join(directory, mapping[key])
      if path not in handles:
        handles[path] = safe_open(path, framework="pt").__enter__()
      return handles[path].get_tensor(key)

    try:
      for key in vision:
        self.assertTrue(torch.equal(read(self.out_dir, saved_map, key), read(base_dir, base_map, key)), key)
    finally:
      for h in handles.values():
        h.__exit__(None, None, None)


class FakeSource:
  def __init__(self, tensors: dict[str, torch.Tensor]):
    self.tensors = tensors

  def get_all_keys(self) -> list[str]:
    return list(self.tensors)

  def load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
    return {k: self.tensors[k] for k in keys}


def pretrained_over(source) -> types.SimpleNamespace:
  return types.SimpleNamespace(state=types.SimpleNamespace(source=source))


class PassthroughTest(unittest.TestCase):
  """Invariant 5: the passthrough completes a save and cannot corrupt one.

  The shard writer emits a shard only once every key mapped to it has been
  yielded, and the 9B interleaves its 333 vision tensors across all four of
  its shards, so a text-only export that yields nothing for them loses every
  language tensor too. That is how run42 died at its first save. The fix is to
  yield the base checkpoint's copies of exactly those tensors -- and only
  those: a language tensor the export failed to produce must stay an error,
  because filling it in would ship untrained weights under a trained name,
  and the MTP head must be left to the writer, which drops it on its own and
  rejects it if offered.
  """

  LANGUAGE = ["model.language_model.layers.0.linear_attn.out_proj.weight", "lm_head.weight"]
  VISION = ["model.visual.blocks.0.attn.qkv.weight", "model.visual.merger.norm.weight"]
  MTP = ["mtp.fc.weight", "mtp.layers.0.input_layernorm.weight"]

  def setUp(self) -> None:
    self.tensors = {
      k: torch.full((2, 2), float(i), dtype=torch.float32) for i, k in enumerate(self.LANGUAGE + self.VISION + self.MTP)
    }
    self.pretrained = pretrained_over(FakeSource(self.tensors))

  def test_off_by_default_so_the_sampler_sync_never_sees_cpu_tensors(self) -> None:
    self.assertEqual(list(qwen35.passthrough_tensors(set(self.LANGUAGE), self.pretrained)), [])

  def test_supplies_exactly_the_vision_tensors_the_text_only_export_skipped(self) -> None:
    with qwen35.multimodal_export_passthrough():
      out = dict(qwen35.passthrough_tensors(set(self.LANGUAGE), self.pretrained, weight_dtype=torch.bfloat16))
    self.assertEqual(sorted(out), sorted(self.VISION))
    for name, tensor in out.items():
      self.assertEqual(tensor.dtype, torch.bfloat16)
      torch.testing.assert_close(tensor.float(), self.tensors[name])

  def test_the_mtp_head_is_neither_supplied_nor_an_error(self) -> None:
    # The writer drops mtp.* from its expected map itself and raises if handed
    # one; the fixture test below proves that against the real writer. Here:
    # an export missing every mtp key is complete as far as we are concerned.
    with qwen35.multimodal_export_passthrough():
      out = dict(qwen35.passthrough_tensors(set(self.LANGUAGE + self.VISION), self.pretrained))
    self.assertEqual(out, {})

  def test_a_complete_export_gets_nothing_added(self) -> None:
    with qwen35.multimodal_export_passthrough():
      self.assertEqual(list(qwen35.passthrough_tensors(set(self.tensors), self.pretrained)), [])

  def test_refuses_to_fill_in_a_language_tensor(self) -> None:
    exported = set(self.LANGUAGE) - {"lm_head.weight"}
    with qwen35.multimodal_export_passthrough():
      with self.assertRaisesRegex(RuntimeError, "lm_head.weight"):
        list(qwen35.passthrough_tensors(exported, self.pretrained))

  def test_a_bridge_without_a_readable_source_is_left_alone(self) -> None:
    with qwen35.multimodal_export_passthrough():
      self.assertEqual(list(qwen35.passthrough_tensors(set(), types.SimpleNamespace())), [])
      self.assertEqual(list(qwen35.passthrough_tensors(set(), pretrained_over(None))), [])

  def test_the_flag_is_scoped_to_the_block_and_restores_what_it_found(self) -> None:
    self.assertFalse(getattr(qwen35.passthrough_state, "enabled", False))
    with qwen35.multimodal_export_passthrough():
      self.assertTrue(qwen35.passthrough_state.enabled)
      with qwen35.multimodal_export_passthrough():
        pass
      self.assertTrue(qwen35.passthrough_state.enabled)
    self.assertFalse(qwen35.passthrough_state.enabled)

  def test_lm_head_is_a_language_tensor(self) -> None:
    # The bridge maps output_layer.weight to a bare lm_head.weight, with no
    # model.language_model. prefix. A prefix list that let it through would
    # silently replace the trained head with the base one.
    self.assertFalse(any("lm_head.weight".startswith(p) for p in qwen35.MULTIMODAL_SOURCE_PREFIXES))


if __name__ == "__main__":
  unittest.main()
