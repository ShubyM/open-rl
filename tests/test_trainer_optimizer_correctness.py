import asyncio
import importlib
import os
import sys
import tempfile
import types
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

import torch


def _load_trainer_modules():
  stubs = {
    "peft": types.SimpleNamespace(
      LoraConfig=object,
      PeftModelForCausalLM=object,
      get_peft_model=lambda *_args, **_kwargs: None,
    ),
    "transformers": types.SimpleNamespace(
      AutoConfig=object,
      AutoModelForCausalLM=object,
      AutoTokenizer=object,
      PreTrainedModel=object,
      PreTrainedTokenizerBase=object,
    ),
  }
  with patch.dict(sys.modules, stubs):
    for module_name in list(sys.modules):
      if module_name == "training" or module_name.startswith("training."):
        del sys.modules[module_name]
    from training import fft_trainer_worker, lora_trainer_worker, losses, model_loading, trainer_worker

  return trainer_worker, lora_trainer_worker, fft_trainer_worker, losses, model_loading


def _load_training_requests_processor_module():
  stubs = {
    "peft": types.SimpleNamespace(
      LoraConfig=object,
      PeftModelForCausalLM=object,
      get_peft_model=lambda *_args, **_kwargs: None,
    ),
    "transformers": types.SimpleNamespace(
      AutoConfig=object,
      AutoModelForCausalLM=object,
      AutoTokenizer=object,
      PreTrainedModel=object,
      PreTrainedTokenizerBase=object,
    ),
  }
  env = {
    "OPEN_RL_ENABLE_FFT": "true",
    "REDIS_URL": "redis://localhost:6379",
  }
  with patch.dict(sys.modules, stubs), patch.dict(os.environ, env):
    for module_name in list(sys.modules):
      if module_name == "server.training_requests_processor":
        del sys.modules[module_name]
    training_requests_processor = importlib.import_module("server.training_requests_processor")
  return training_requests_processor


trainer_worker_module, lora_trainer_worker_module, fft_trainer_worker_module, losses_module, model_loading_module = _load_trainer_modules()
training_requests_processor_module = _load_training_requests_processor_module()
BaseTrainerWorker = trainer_worker_module.BaseTrainerWorker
FFTTrainingWorker = fft_trainer_worker_module.FFTTrainingWorker
LoraTrainingWorker = lora_trainer_worker_module.LoraTrainingWorker


class _PeftModelStub:
  def __init__(self, adapter_params):
    self.adapter_params = adapter_params
    self.active_adapter = None

  def set_adapter(self, adapter_id):
    self.active_adapter = adapter_id
    for param in self.parameters():
      param.requires_grad_(False)
    for param in self.adapter_params[adapter_id]:
      param.requires_grad_(True)

  def parameters(self):
    for params in self.adapter_params.values():
      yield from params

  def save_pretrained(self, save_directory, *_args, selected_adapters=None, **_kwargs):
    # Mirror peft's on-disk contract: each selected adapter gets a
    # subdirectory with its config (save_adapter renames it into place).
    for adapter in selected_adapters or [self.active_adapter]:
      adapter_dir = os.path.join(save_directory, adapter)
      os.makedirs(adapter_dir, exist_ok=True)
      with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        f.write("{}")


class _TokenizerStub:
  pad_token_id = 0


class _LogitModelStub:
  def __init__(self, vocab_size: int = 17):
    self.vocab_size = vocab_size
    self.calls = []
    self.config = types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace(_attn_implementation="sdpa"))

  def train(self):
    return None

  def __call__(self, input_tensor, attention_mask=None, **_kwargs):
    if attention_mask is not None:
      self.calls.append((input_tensor.detach().clone(), attention_mask.detach().clone()))
    vocab = torch.arange(self.vocab_size, dtype=torch.float32, device=input_tensor.device).view(1, 1, -1)
    positions = torch.arange(input_tensor.shape[1], dtype=torch.float32, device=input_tensor.device).view(1, -1, 1)
    logits = torch.cos(input_tensor.float().unsqueeze(-1) * 0.11 + positions * 0.07 + vocab * 0.13)
    logits.requires_grad_()
    return types.SimpleNamespace(logits=logits)


class _FullModelStub:
  def __init__(self, params):
    self.params = params

  def train(self):
    return None

  def parameters(self):
    yield from self.params

  def to(self, device):
    for param in self.params:
      param.data = param.data.to(device)
      if param.grad is not None:
        param.grad.data = param.grad.data.to(device)
    return self


class _RecordingFullWorker(training_requests_processor_module.FFTTrainingWorker):
  def __init__(self):
    super().__init__()
    self.base_model_name = None
    self.loaded_base_models = []
    self.created_models = []
    self.saved_states = []

  def load_base_model(self, base_model_name):
    self.base_model_name = base_model_name
    self.loaded_base_models.append(base_model_name)

  def create_model(self, base_model_name, model_id, config):
    self.created_models.append((base_model_name, model_id, config))

  def forward_backward(self, data, loss_fn, loss_config=None, model_id=None, forward_only=False):
    return {"model_id": model_id, "loss_fn": loss_fn, "loss_config": loss_config, "data": data, "forward_only": forward_only}

  def save_state(self, model_id, state_path, include_optimizer=False, kind="state"):
    self.saved_states.append((model_id, state_path, include_optimizer, kind))
    return {"path": state_path}


class _RecordingLoraWorker(training_requests_processor_module.LoraTrainingWorker):
  def __init__(self):
    super().__init__()
    self.loaded_base_models = []
    self.created_models = []

  def load_base_model(self, base_model_name):
    self.loaded_base_models.append(base_model_name)

  def create_model(self, base_model_name, model_id, config):
    self.created_models.append((base_model_name, model_id, config))


class _FutureStoreStub:
  def __init__(self, events=None):
    self.results = {}
    self.events = events

  async def set_future(self, req_id, result):
    if self.events is not None:
      self.events.append(("set_future", req_id))
    self.results[req_id] = result


class _TrainingRequestsStoreStub(_FutureStoreStub):
  def __init__(self, batches, events=None):
    super().__init__(events=events)
    self.batches = list(batches)
    self.queried_model_ids = []

  async def get_requests_for_model(self, model_id):
    self.queried_model_ids.append(model_id)
    if self.batches:
      return self.batches.pop(0)
    raise asyncio.CancelledError()

  async def get_value(self, key: str) -> str | None:
    return None

  async def record_accel_usage_event(self, claim_id: str, event_data: dict) -> None:
    pass

  def get_value_sync(self, key: str) -> str | None:
    return None


class _TimeSlicerStub:
  def __init__(self, events=None):
    self.events = events if events is not None else []

  async def register(self, workload):
    self.events.append(("register", workload))
    return {"ok": True}

  @asynccontextmanager
  async def acquire(self, workload):
    self.events.append(("acquire", workload))
    try:
      yield
    finally:
      self.events.append(("release", workload))

  async def unregister(self, workload):
    self.events.append(("unregister", workload))
    return {"ok": True}

  async def close(self):
    self.events.append(("close",))


class TestTextOnlyModelLoading(unittest.TestCase):
  def test_wide_flex_attention_uses_low_resource_kernel(self) -> None:
    text_config = types.SimpleNamespace(_attn_implementation="flex_attention", head_dim=256, global_head_dim=512)
    config = types.SimpleNamespace(get_text_config=lambda: text_config)

    self.assertEqual(
      trainer_worker_module.attention_forward_kwargs(config),
      {
        "kernel_options": {
          "fwd_BLOCK_M": 16,
          "fwd_BLOCK_N": 16,
          "fwd_num_stages": 1,
          "bwd_BLOCK_M1": 16,
          "bwd_BLOCK_N1": 16,
          "bwd_BLOCK_M2": 16,
          "bwd_BLOCK_N2": 16,
          "bwd_num_stages": 1,
        }
      },
    )

  def test_standard_attention_does_not_override_kernel(self) -> None:
    text_config = types.SimpleNamespace(_attn_implementation="sdpa", head_dim=512)
    config = types.SimpleNamespace(get_text_config=lambda: text_config)

    self.assertEqual(trainer_worker_module.attention_forward_kwargs(config), {})

  def test_gemma4_loads_nested_language_model_directly(self) -> None:
    text_config = types.SimpleNamespace(model_type="gemma4_text")
    config = types.SimpleNamespace(model_type="gemma4", text_config=text_config)
    loaded = object()

    with (
      patch("transformers.AutoConfig.from_pretrained", return_value=config),
      patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=(loaded, {"missing_keys": []})) as from_pretrained,
    ):
      result = model_loading_module.load_text_causal_lm("google/gemma-4-E4B-it", dtype=torch.bfloat16)

    self.assertIs(result, loaded)
    from_pretrained.assert_called_once_with(
      "google/gemma-4-E4B-it",
      output_loading_info=True,
      dtype=torch.bfloat16,
      attn_implementation="flex_attention",
      config=text_config,
      key_mapping={r"^model\.language_model\.": "model."},
    )

  def test_regular_causal_model_load_is_unchanged(self) -> None:
    config = types.SimpleNamespace(model_type="qwen3")
    with (
      patch("transformers.AutoConfig.from_pretrained", return_value=config),
      patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=(object(), {"missing_keys": []})) as from_pretrained,
    ):
      model_loading_module.load_text_causal_lm("Qwen/Qwen3-8B", dtype=torch.bfloat16)

    from_pretrained.assert_called_once_with("Qwen/Qwen3-8B", output_loading_info=True, dtype=torch.bfloat16, attn_implementation="sdpa")

  def test_saved_gemma_text_checkpoint_loads_both_key_layouts(self) -> None:
    """Trainer-saved text checkpoints carry hub-layout keys (transformers
    reverses the load-time key_mapping when saving); the text-only branch must
    pass the mapping too or every parameter is silently reinitialized."""
    config = types.SimpleNamespace(model_type="gemma4_text")
    with (
      patch("transformers.AutoConfig.from_pretrained", return_value=config),
      patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=(object(), {"missing_keys": []})) as from_pretrained,
    ):
      model_loading_module.load_text_causal_lm("/checkpoints/step-4")

    from_pretrained.assert_called_once_with(
      "/checkpoints/step-4",
      output_loading_info=True,
      attn_implementation="flex_attention",
      key_mapping={r"^model\.language_model\.": "model."},
    )

  def test_missing_keys_raise_instead_of_training_a_random_model(self) -> None:
    config = types.SimpleNamespace(model_type="gemma4_text")
    with (
      patch("transformers.AutoConfig.from_pretrained", return_value=config),
      patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=(object(), {"missing_keys": ["model.layers.0.self_attn.q_proj.weight"]}),
      ),self.assertRaisesRegex(RuntimeError, "uninitialized")
    ):
      model_loading_module.load_text_causal_lm("/checkpoints/step-4")


def _datum(model_input, target_tokens, *, weights=None, logprobs=None, advantages=None):
  loss_fn_inputs = {"target_tokens": trainer_worker_module.TensorData(data=target_tokens)}
  if weights is not None:
    loss_fn_inputs["weights"] = trainer_worker_module.TensorData(data=weights)
  if logprobs is not None:
    loss_fn_inputs["logprobs"] = trainer_worker_module.TensorData(data=logprobs)
  if advantages is not None:
    loss_fn_inputs["advantages"] = trainer_worker_module.TensorData(data=advantages)
  return trainer_worker_module.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)


class TestTrainerOptimizerCorrectness(unittest.TestCase):
  def test_lora_create_model_loads_base_then_creates_adapter(self) -> None:
    worker = LoraTrainingWorker()
    config = lora_trainer_worker_module.LoraConfig(rank=2, seed=123)
    calls = []

    worker.load_base_model = lambda base_model_name: calls.append(("load", base_model_name))
    worker.create_adapter = lambda model_id, adapter_config: calls.append(("adapter", model_id, adapter_config))

    worker.create_model("base-model", "adapter-a", config)

    self.assertEqual(calls[0], ("load", "base-model"))
    self.assertEqual(calls[1][0], "adapter")
    self.assertEqual(calls[1][1], "adapter-a")
    self.assertIs(calls[1][2], config)

  def test_save_adapter_selects_adapter_it_saves(self) -> None:
    adapter_a_param = torch.nn.Parameter(torch.tensor([1.0]))
    adapter_b_param = torch.nn.Parameter(torch.tensor([1.0]))
    worker = LoraTrainingWorker()
    worker.peft_model = _PeftModelStub(
      {
        "adapter-a": [adapter_a_param],
        "adapter-b": [adapter_b_param],
      }
    )
    worker.peft_model.set_adapter("adapter-b")

    with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": tmp_dir}):
      worker.save_adapter("adapter-a")
      self.assertTrue(os.path.exists(os.path.join(tmp_dir, "peft", "adapter-a", "metadata.json")))

    self.assertEqual(worker.peft_model.active_adapter, "adapter-a")

  def test_fft_create_model_loads_base_then_prepares_model(self) -> None:
    worker = FFTTrainingWorker()
    config = fft_trainer_worker_module.FFTConfig(seed=123)
    calls = []

    worker.load_base_model = lambda base_model_name: calls.append(("load", base_model_name))
    worker.prepare_model_for_training = lambda: calls.append(("prepare", None))

    worker.create_model("base-model", "model-a", config)

    self.assertEqual(calls, [("load", "base-model"), ("prepare", None)])

  def test_optim_step_only_updates_active_adapter_params(self) -> None:
    active_param = torch.nn.Parameter(torch.tensor([1.0]))
    other_param = torch.nn.Parameter(torch.tensor([1.0]))
    active_param.grad = torch.tensor([1.0])
    other_param.grad = torch.tensor([10.0])

    worker = LoraTrainingWorker()
    worker.peft_model = _PeftModelStub(
      {
        "adapter-a": [active_param],
        "adapter-b": [other_param],
      }
    )
    worker.adapter_states = {
      "adapter-a": {"trainable_params": lora_trainer_worker_module.active_adapter_parameters(worker.peft_model, "adapter-a"), "optimizer": None}
    }
    worker.save_adapter = lambda *_args, **_kwargs: None

    result = worker.optim_step(
      {
        "learning_rate": 0.1,
        "beta1": 0.0,
        "beta2": 0.0,
        "eps": 1e-8,
        "weight_decay": 0.0,
      },
      "adapter-a",
    )

    self.assertEqual(worker.peft_model.active_adapter, "adapter-a")
    self.assertAlmostEqual(result["metrics"]["grad_norm:mean"], 1.0)
    self.assertFalse(torch.allclose(active_param.detach(), torch.tensor([1.0])))
    self.assertTrue(torch.allclose(other_param.detach(), torch.tensor([1.0])))
    if active_param.grad is not None:
      self.assertTrue(torch.allclose(active_param.grad, torch.zeros_like(active_param.grad)))
    self.assertIsNotNone(other_param.grad)

  def test_fft_optim_step_updates_full_model_trainable_params(self) -> None:
    trainable_param = torch.nn.Parameter(torch.tensor([1.0]))
    frozen_param = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    trainable_param.grad = torch.tensor([1.0])
    frozen_param.grad = torch.tensor([10.0])

    worker = FFTTrainingWorker()
    worker.model = _FullModelStub([trainable_param, frozen_param])
    worker.trainable_params = fft_trainer_worker_module.trainable_model_parameters(worker.model)

    result = worker.optim_step(
      {
        "learning_rate": 0.1,
        "beta1": 0.0,
        "beta2": 0.0,
        "eps": 1e-8,
        "weight_decay": 0.0,
      }
    )

    self.assertAlmostEqual(result["metrics"]["grad_norm:mean"], 1.0)
    self.assertFalse(torch.allclose(trainable_param.detach(), torch.tensor([1.0])))
    self.assertTrue(torch.allclose(frozen_param.detach(), torch.tensor([1.0])))
    if trainable_param.grad is not None:
      self.assertTrue(torch.allclose(trainable_param.grad, torch.zeros_like(trainable_param.grad)))
    self.assertIsNotNone(frozen_param.grad)

  def test_fft_optim_step_on_cpu_keeps_optimizer_state_on_host(self) -> None:
    trainable_param = torch.nn.Parameter(torch.tensor([1.0]))
    trainable_param.grad = torch.tensor([1.0])

    worker = FFTTrainingWorker()
    worker.model = _FullModelStub([trainable_param])
    worker.trainable_params = fft_trainer_worker_module.trainable_model_parameters(worker.model)

    with patch.dict(os.environ, {"OPEN_RL_OPTIM_CPU_STEP": "1"}):
      result = worker.optim_step(
        {
          "learning_rate": 0.1,
          "beta1": 0.0,
          "beta2": 0.0,
          "eps": 1e-8,
          "weight_decay": 0.0,
        }
      )

    self.assertAlmostEqual(result["metrics"]["grad_norm:mean"], 1.0)
    # The step moves the model back to the accelerator; compare on the host.
    self.assertFalse(torch.allclose(trainable_param.detach().cpu(), torch.tensor([1.0])))
    for state in worker.optimizer.state.values():
      for value in state.values():
        if isinstance(value, torch.Tensor):
          self.assertEqual(value.device.type, "cpu")


class TestTrainingRequestsProcessorFullMode(unittest.IsolatedAsyncioTestCase):
  async def test_importing_training_requests_processor_does_not_create_worker(self) -> None:
    self.assertFalse(hasattr(training_requests_processor_module, "worker"))

  async def test_lora_processor_create_model_uses_worker_create_model(self) -> None:
    worker = _RecordingLoraWorker()
    store = _FutureStoreStub()
    processor = training_requests_processor_module.LoraTrainingRequestsProcessor(store, worker)

    await processor.process_request(
      {
        "request_id": "req-a",
        "model_id": "adapter-a",
        "op": "create_model",
        "payload": {
          "base_model": "base-model",
          "lora_config": {"seed": 123, "rank": 2},
        },
      },
      "adapter-a",
    )

    self.assertEqual(worker.loaded_base_models, [])
    base_model, model_id, config = worker.created_models[0]
    self.assertEqual(base_model, "base-model")
    self.assertEqual(model_id, "adapter-a")
    self.assertEqual(config.seed, 123)
    self.assertEqual(config.rank, 2)
    result = store.results["req-a"]
    self.assertEqual(result["model_id"], "adapter-a")
    self.assertEqual(result["rank"], 2)
    self.assertEqual(result["fine_tuning_type"], "lora")
    self.assertEqual(result["type"], "model_created")

  def test_parse_datum_flattens_chunked_model_input(self) -> None:
    datum = training_requests_processor_module.parse_datum(
      {
        "model_input": {"chunks": [{"tokens": [1, 2]}, {"tokens": [3]}]},
        "loss_fn_inputs": {
          "target_tokens": [2, 3, 4],
          "weights": {"data": [1.0, 0.5, 0.25]},
        },
      }
    )

    self.assertEqual(datum.model_input, [1, 2, 3])
    self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [2, 3, 4])
    self.assertEqual(datum.loss_fn_inputs["weights"].data, [1.0, 0.5, 0.25])

  async def test_full_processor_create_model_uses_model_worker(self) -> None:
    worker = _RecordingFullWorker()
    store = _FutureStoreStub()
    time_slicer = _TimeSlicerStub()

    with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
      processor = training_requests_processor_module.FFTTrainingRequestsProcessor(store, worker, "model-a", time_slicer=time_slicer)
      await processor.process_request(
        {
          "request_id": "req-a",
          "model_id": "model-a",
          "op": "create_model",
          "payload": {
            "base_model": "base-model",
            "full_config": {"seed": 123, "rank": 8},
          },
        },
        "model-a",
      )

    self.assertEqual(worker.loaded_base_models, [])
    base_model, model_id, config = worker.created_models[0]
    self.assertEqual(base_model, "base-model")
    self.assertEqual(model_id, "model-a")
    self.assertEqual(config.seed, 123)
    result = store.results["req-a"]
    self.assertEqual(result["model_id"], "model-a")
    self.assertEqual(result["base_model"], "base-model")
    self.assertEqual(result["fine_tuning_type"], "full")
    self.assertEqual(result["type"], "model_created")

  async def test_full_processor_saves_sampler_checkpoint_as_full_state(self) -> None:
    worker = _RecordingFullWorker()
    store = _FutureStoreStub()
    time_slicer = _TimeSlicerStub()

    with patch.dict(os.environ, {"OPEN_RL_TMP_DIR": "/tmp/open-rl-test", "REDIS_URL": "redis://localhost:6379"}):
      processor = training_requests_processor_module.FFTTrainingRequestsProcessor(store, worker, "model-a", time_slicer=time_slicer)
      await processor.process_request(
        {
          "request_id": "req-a",
          "model_id": "model-a",
          "op": "save_weights_for_sampler",
          "payload": {
            "path": "tinker://model-a/sampler_weights/final",
            "sampling_session_id": "tinker://model-a/sampler_weights/sampler-7",
          },
        },
        "model-a",
      )

    self.assertEqual(
      worker.saved_states,
      [("model-a", "/tmp/open-rl-test/sampler_full/model-a/sampler_weights/final", False, "sampler")],
    )
    self.assertEqual(
      store.results["req-a"],
      {
        "path": "tinker://model-a/sampler_weights/final",
        "sampling_session_id": "tinker://model-a/sampler_weights/sampler-7",
        "type": "sampler_weights_saved",
      },
    )

  async def test_full_processor_requires_redis(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      await training_requests_processor_module.run_training_requests_processor(_RecordingFullWorker(), "model-a")

  async def test_full_processor_uses_default_time_slicer_client(self) -> None:
    store = _TrainingRequestsStoreStub([])
    time_slicer = _TimeSlicerStub()

    with (
      patch.dict(
        os.environ,
        {
          "OPEN_RL_ENABLE_FFT": "true",
          "REDIS_URL": "redis://localhost:6379",
        },
        clear=True,
      ),
      patch.object(training_requests_processor_module, "get_store", return_value=store),
      patch.object(training_requests_processor_module, "time_slicer_client_from_env", return_value=time_slicer) as time_slicer_client_from_env,
    ):
      await training_requests_processor_module.run_training_requests_processor(_RecordingFullWorker(), "model-a")

    time_slicer_client_from_env.assert_called_once_with()
    self.assertEqual([event[0] for event in time_slicer.events], ["register", "unregister", "close"])

  async def test_full_processor_uses_injected_time_slicer(self) -> None:
    worker = _RecordingFullWorker()
    store = _TrainingRequestsStoreStub(
      [
        [
          {
            "request_id": "req-a",
            "model_id": "model-a",
            "op": "create_model",
            "payload": {
              "base_model": "base-model",
              "full_config": {"seed": 123},
            },
          }
        ]
      ]
    )
    time_slicer = _TimeSlicerStub()

    with (
      patch.dict(
        os.environ,
        {
          "OPEN_RL_ENABLE_FFT": "true",
          "REDIS_URL": "redis://localhost:6379",
        },
      ),
      patch.object(training_requests_processor_module, "get_store", return_value=store),
    ):
      await training_requests_processor_module.run_training_requests_processor(worker, "model-a", time_slicer=time_slicer)

    self.assertEqual(store.queried_model_ids, ["model-a", "model-a"])
    self.assertEqual([event[0] for event in time_slicer.events], ["register", "acquire", "release", "unregister", "close"])
    for event in time_slicer.events:
      if len(event) >= 2 and event[0] != "close":
        self.assertEqual(event[1].job_id, "trainer-model-a")
        self.assertEqual(event[1].group, "trainers")
    self.assertEqual(worker.created_models[0][0], "base-model")
    self.assertEqual(store.results["req-a"]["model_id"], "model-a")

  async def test_full_processor_publishes_result_after_release(self) -> None:
    events = []
    worker = _RecordingFullWorker()
    store = _TrainingRequestsStoreStub(
      [
        [
          {
            "request_id": "req-a",
            "model_id": "model-a",
            "op": "create_model",
            "payload": {
              "base_model": "base-model",
              "full_config": {"seed": 123},
            },
          }
        ]
      ],
      events=events,
    )
    time_slicer = _TimeSlicerStub(events=events)

    with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
      processor = training_requests_processor_module.FFTTrainingRequestsProcessor(store, worker, "model-a", time_slicer=time_slicer)
      await processor.run_once()

    self.assertEqual([event[0] for event in events], ["acquire", "release", "set_future"])
    self.assertEqual(store.results["req-a"]["type"], "model_created")


class TestTrainerPaddedBatchingMath(unittest.TestCase):
  def setUp(self) -> None:
    patcher = patch.dict(os.environ, {"OPEN_RL_FUSED_LOGPROB": "0"})
    patcher.start()
    self.addCleanup(patcher.stop)

  def _worker(self) -> BaseTrainerWorker:
    worker = BaseTrainerWorker()
    worker.device = torch.device("cpu")
    worker.tokenizer = _TokenizerStub()
    return worker

  def _data(self):
    return [
      _datum(
        [3, 4, 5, 6],
        [1, 2, 3, 4],
        weights=[1.0, 0.5, 0.25, 2.0],
        logprobs=[-0.1, -0.2, -0.3, -0.4],
        advantages=[1.0, -0.5, 2.0, 0.25],
      ),
      _datum(
        [7, 8],
        [2, 3],
        logprobs=[-0.7, -0.8],
        advantages=[0.75, 1.25],
      ),
      _datum(
        [9, 10, 11],
        [5, 6, 7, 8],
        weights=[0.2, 0.4, 0.6, 0.8],
        logprobs=[-0.9, -1.0, -1.1, -1.2],
        advantages=[-1.0, 0.3, 0.9, 1.7],
      ),
    ]

  def training_tensors(self, worker, model, data):
    input_ids, attention_mask, input_lengths = worker.pad_model_inputs(data)
    target_token_ids, weights, lengths = worker.pad_targets_and_weights(data, input_lengths)
    logprobs = worker.compute_target_logprobs(model, input_ids, attention_mask, target_token_ids)
    old_logprobs = worker.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in data], lengths, torch.float32)
    advantages = worker.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in data], lengths, torch.float32)
    return logprobs, weights, old_logprobs, advantages, lengths

  def test_padded_batch_logprobs_and_losses_match_per_example_math(self) -> None:
    worker = self._worker()
    model = _LogitModelStub()
    data = self._data()

    batch_logprobs, batch_weights, batch_old_logprobs, batch_advantages, batch_lengths = self.training_tensors(worker, model, data)
    single_results = [self.training_tensors(worker, model, [datum]) for datum in data]

    for row, (single_logprobs, single_weights, single_old_logprobs, single_advantages, single_lengths) in enumerate(single_results):
      length = batch_lengths[row]
      self.assertEqual(length, single_lengths[0])
      torch.testing.assert_close(batch_logprobs[row, :length], single_logprobs[0, :length])
      torch.testing.assert_close(batch_weights[row, :length], single_weights[0, :length])
      torch.testing.assert_close(batch_weights[row, length:], torch.zeros_like(batch_weights[row, length:]))
      torch.testing.assert_close(batch_old_logprobs[row, :length], single_old_logprobs[0, :length])
      torch.testing.assert_close(batch_old_logprobs[row, length:], torch.zeros_like(batch_old_logprobs[row, length:]))
      torch.testing.assert_close(batch_advantages[row, :length], single_advantages[0, :length])
      torch.testing.assert_close(batch_advantages[row, length:], torch.zeros_like(batch_advantages[row, length:]))

    def single_sum(fn):
      losses = [fn(logprobs, weights, old_logprobs, advantages).sum() for logprobs, weights, old_logprobs, advantages, _lengths in single_results]
      return torch.stack(losses).sum()

    torch.testing.assert_close(
      losses_module.cross_entropy_loss(batch_logprobs, batch_weights).sum(),
      single_sum(lambda logprobs, weights, _old_logprobs, _advantages: losses_module.cross_entropy_loss(logprobs, weights)),
    )
    torch.testing.assert_close(
      losses_module.importance_sampling_loss(
        batch_logprobs,
        batch_weights,
        batch_old_logprobs,
        batch_advantages,
      ).sum(),
      single_sum(
        lambda logprobs, weights, old_logprobs, advantages: losses_module.importance_sampling_loss(
          logprobs,
          weights,
          old_logprobs,
          advantages,
        )
      ),
    )
    ppo_config = {"clip_range": 0.2, "kl_coeff": 0.03}
    torch.testing.assert_close(
      losses_module.ppo_loss(
        batch_logprobs,
        batch_weights,
        batch_old_logprobs,
        batch_advantages,
        ppo_config,
      ).sum(),
      single_sum(
        lambda logprobs, weights, old_logprobs, advantages: losses_module.ppo_loss(
          logprobs,
          weights,
          old_logprobs,
          advantages,
          ppo_config,
        )
      ),
    )

  def test_token_budget_batches_preserve_examples(self) -> None:
    worker = self._worker()
    data = self._data()
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "6"}):
      batches = worker.make_training_batches(data)

    seen = [idx for batch in batches for idx, _datum in batch]
    self.assertCountEqual(seen, range(len(data)))
    for batch in batches:
      padded_tokens = max(len(datum.model_input) for _idx, datum in batch) * len(batch)
      self.assertTrue(len(batch) == 1 or padded_tokens <= 6)

  def test_forward_backward_padded_batches_preserve_client_output_shape(self) -> None:
    worker = self._worker()
    model = _LogitModelStub()
    data = self._data()

    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "12"}):
      result = worker.forward_backward(model, data, "cross_entropy")

    self.assertEqual(len(result["loss_fn_outputs"]), len(data))
    self.assertGreater(len(model.calls), 0)
    self.assertTrue(any(call[0].shape[0] > 1 for call in model.calls))
    for datum, output in zip(data, result["loss_fn_outputs"], strict=True):
      logprobs = output["logprobs"]
      self.assertEqual(logprobs["shape"], [min(len(datum.model_input), len(datum.loss_fn_inputs["target_tokens"].data))])

  def test_zero_effective_advantages_skip_policy_backward(self) -> None:
    for loss_fn in ("importance_sampling", "ppo"):
      with self.subTest(loss_fn=loss_fn):
        worker = self._worker()
        parameter = torch.nn.Parameter(torch.tensor(0.25))
        model = _FullModelStub([parameter])
        data = [
          _datum(
            [3, 4],
            [1, 2],
            weights=[1.0, 0.0],
            logprobs=[-0.1, -0.2],
            advantages=[0.0, 2.0],
          )
        ]

        with patch.object(
          worker,
          "compute_target_logprobs",
          side_effect=lambda _model, _inputs, _mask, targets, parameter=parameter: parameter.expand_as(targets),
        ):
          result = worker.forward_backward(model, data, loss_fn)

        self.assertIsNone(parameter.grad)
        self.assertEqual(result["metrics"], {"loss:mean": 0.0, "loss:sum": 0.0})
        self.assertEqual(result["loss_fn_outputs"][0]["logprobs"]["shape"], [2])

  def test_ppo_kl_penalty_keeps_backward_for_zero_advantages(self) -> None:
    worker = self._worker()
    parameter = torch.nn.Parameter(torch.tensor(0.25))
    model = _FullModelStub([parameter])
    data = [_datum([3], [1], weights=[1.0], logprobs=[-0.1], advantages=[0.0])]

    with patch.object(
      worker,
      "compute_target_logprobs",
      side_effect=lambda _model, _inputs, _mask, targets: parameter.expand_as(targets),
    ):
      worker.forward_backward(model, data, "ppo", {"kl_coeff": 0.1})

    self.assertIsNotNone(parameter.grad)
    self.assertNotEqual(parameter.grad.item(), 0.0)

  def test_fft_forward_backward_uses_single_process_model(self) -> None:
    worker = FFTTrainingWorker()
    worker.device = torch.device("cpu")
    worker.tokenizer = _TokenizerStub()
    worker.model = _LogitModelStub()
    data = self._data()

    result = worker.forward_backward(data, "cross_entropy")

    self.assertEqual(len(result["loss_fn_outputs"]), len(data))
    self.assertGreater(len(worker.model.calls), 0)


class TestDataParallelForwardBackward(unittest.TestCase):
  """Datum sharding must reproduce single-process gradients under FSDP's per-backward averaging."""

  def _worker(self) -> BaseTrainerWorker:
    worker = BaseTrainerWorker()
    worker.device = torch.device("cpu")
    worker.tokenizer = _TokenizerStub()
    return worker

  def _data(self):
    return [
      _datum([3, 4, 5], [1, 2, 3], weights=[1.0, 0.5, 0.25]),
      _datum([7, 8], [2, 3], weights=[2.0, 0.75]),
      _datum([9], [4], weights=[1.5]),
    ]

  def _run_forward_backward(self, data, *, loss_fn="cross_entropy", loss_config=None, env=None, fakes=None):
    worker = self._worker()
    parameter = torch.nn.Parameter(torch.tensor(0.25))
    model = _FullModelStub([parameter])
    patches = [patch.object(trainer_worker_module, name, fake) for name, fake in (fakes or {}).items()]
    with patch.dict(os.environ, env or {}, clear=False), patch.object(
      worker,
      "compute_target_logprobs",
      side_effect=lambda _model, _inputs, _mask, targets, parameter=parameter: parameter.expand_as(targets),
    ) as compute_calls:
      for p in patches:
        p.start()
      try:
        result = worker.forward_backward(model, data, loss_fn, loss_config)
      finally:
        for p in patches:
          p.stop()
    return result, parameter, compute_calls

  def test_shard_datum_indices_round_robin(self) -> None:
    self.assertEqual(trainer_worker_module.shard_datum_indices(5, 0, 2), [0, 2, 4])
    self.assertEqual(trainer_worker_module.shard_datum_indices(5, 1, 2), [1, 3])
    self.assertEqual(trainer_worker_module.shard_datum_indices(1, 1, 2), [])
    self.assertEqual(trainer_worker_module.shard_datum_indices(3, 0, 1), [0, 1, 2])

  def test_sharded_gradients_average_to_single_process_gradient(self) -> None:
    data = self._data()
    reference, reference_param, _calls = self._run_forward_backward(data)
    placeholder = {"logprobs": {"data": [], "dtype": "float32", "shape": [0]}}

    def gather_with_placeholders(part):
      missing = {idx: placeholder for idx in range(len(data)) if idx not in part}
      return [part, missing]

    rank_grads = []
    rank_totals = []
    rank_parts = {}
    for rank_id in ("0", "1"):
      captured = {}
      fakes = {
        "all_reduce_max": lambda passes: 2,
        "all_reduce_sum": lambda value: captured.setdefault("total", value),
        "all_gather_object": lambda part: (captured.setdefault("part", part), gather_with_placeholders(part))[1],
      }
      _result, parameter, _calls = self._run_forward_backward(data, env={"WORLD_SIZE": "2", "RANK": rank_id}, fakes=fakes)
      rank_grads.append(parameter.grad)
      rank_totals.append(captured["total"])
      rank_parts.update(captured["part"])

    # FSDP averages each backward across ranks; the shard-count loss scaling
    # must make that average equal the single-process gradient sum.
    torch.testing.assert_close((rank_grads[0] + rank_grads[1]) / 2, reference_param.grad)
    self.assertAlmostEqual(sum(rank_totals), reference["metrics"]["loss:sum"], places=5)
    self.assertEqual(sorted(rank_parts), list(range(len(data))))

  def test_short_rank_pads_with_zero_scaled_filler_passes(self) -> None:
    data = self._data()
    placeholder = {"logprobs": {"data": [], "dtype": "float32", "shape": [0]}}
    fakes = {
      "all_reduce_max": lambda passes: 2,
      "all_reduce_sum": lambda value: value,
      "all_gather_object": lambda part: [part, {idx: placeholder for idx in range(len(data)) if idx not in part}],
    }

    result, parameter, compute_calls = self._run_forward_backward(data, env={"WORLD_SIZE": "2", "RANK": "1"}, fakes=fakes)

    # Rank 1 owns one datum but must run two passes; the filler pass leaves
    # gradients and reported loss untouched.
    self.assertEqual(compute_calls.call_count, 2)
    weights_rank1 = 2.0 + 0.75
    torch.testing.assert_close(parameter.grad, torch.tensor(2 * -weights_rank1))
    self.assertAlmostEqual(result["metrics"]["loss:sum"], 0.25 * -weights_rank1, places=5)
    self.assertEqual(result["loss_fn_outputs"][1]["logprobs"]["shape"], [2])

  def test_distributed_ranks_never_skip_zero_advantage_backward(self) -> None:
    data = [_datum([3, 4], [1, 2], weights=[1.0, 0.0], logprobs=[-0.1, -0.2], advantages=[0.0, 2.0])]
    placeholder = {"logprobs": {"data": [], "dtype": "float32", "shape": [0]}}
    fakes = {
      "all_reduce_max": lambda passes: 1,
      "all_reduce_sum": lambda value: value,
      "all_gather_object": lambda part: [part, {idx: placeholder for idx in range(len(data)) if idx not in part}],
    }

    result, parameter, _calls = self._run_forward_backward(data, loss_fn="importance_sampling", env={"WORLD_SIZE": "2", "RANK": "0"}, fakes=fakes)

    # Single-process mode skips this backward entirely (see
    # test_zero_effective_advantages_skip_policy_backward); a distributed rank
    # must still run it so the group's collective counts stay aligned.
    self.assertIsNotNone(parameter.grad)
    torch.testing.assert_close(parameter.grad, torch.tensor(0.0))
    self.assertEqual(result["metrics"]["loss:sum"], 0.0)


class TestFSDPAttentionKernelOptions(unittest.TestCase):
  """The FSDP forward must use the same low-resource FlexAttention tiles as the
  single-process paths: Gemma's 512-dim heads need 256KB shared memory under the
  default tiles while sm_90 GPUs top out at 227KB (Triton "out of resource")."""

  class _BackboneStub:
    def __init__(self):
      self.calls = []

    def __call__(self, **kwargs):
      self.calls.append(kwargs)
      input_ids = kwargs["input_ids"]
      hidden = torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
      return types.SimpleNamespace(last_hidden_state=hidden)

  class _WideHeadCausalLMStub:
    def __init__(self, attn_implementation: str = "flex_attention", head_dim: int = 512):
      self.model = TestFSDPAttentionKernelOptions._BackboneStub()
      text_config = types.SimpleNamespace(_attn_implementation=attn_implementation, head_dim=head_dim, global_head_dim=None)
      self.config = types.SimpleNamespace(get_text_config=lambda: text_config)
      self._head = torch.nn.Linear(4, 7, bias=False)

    def get_output_embeddings(self):
      return self._head

  def _forward(self, causal_lm):
    wrapper = fft_trainer_worker_module.FSDPTargetLogprobModel(causal_lm)
    input_ids = torch.ones(1, 3, dtype=torch.long)
    attention_mask = torch.ones(1, 3, dtype=torch.long)
    targets = torch.tensor([[1, 2, 3]])
    result = wrapper(input_ids, attention_mask, targets)
    self.assertEqual(result.shape, (1, 3))
    return causal_lm.model.calls[0]

  def test_wide_flex_attention_heads_get_low_resource_tiles(self) -> None:
    call_kwargs = self._forward(self._WideHeadCausalLMStub())

    self.assertIn("kernel_options", call_kwargs)
    self.assertEqual(call_kwargs["kernel_options"]["fwd_BLOCK_M"], 16)
    self.assertEqual(call_kwargs["kernel_options"]["fwd_num_stages"], 1)

  def test_narrow_heads_keep_default_tiles(self) -> None:
    call_kwargs = self._forward(self._WideHeadCausalLMStub(head_dim=128))
    self.assertNotIn("kernel_options", call_kwargs)

  def test_sdpa_models_get_no_flex_tiles(self) -> None:
    call_kwargs = self._forward(self._WideHeadCausalLMStub(attn_implementation="sdpa"))
    self.assertNotIn("kernel_options", call_kwargs)

  def test_backbone_forward_consults_activation_offload(self) -> None:
    from contextlib import nullcontext

    offload_calls = []

    def recording_offload(tensor):
      offload_calls.append(tensor.shape)
      return nullcontext()

    with patch.object(fft_trainer_worker_module, "activation_offload_context", recording_offload):
      self._forward(self._WideHeadCausalLMStub())

    # Long-context FSDP runs rely on OPEN_RL_ACTIVATION_CPU_OFFLOAD exactly like
    # single-GPU runs: activations are per-rank and unsharded.
    self.assertEqual(len(offload_calls), 1)


class TestAtomicCheckpointWrites(unittest.TestCase):
  """A save killed mid-write must never leave a loadable mix of old and new shards."""

  class _SavingModelStub:
    def save_pretrained(self, path, state_dict=None):
      with open(os.path.join(path, "model.safetensors"), "w") as f:
        f.write("new-weights")

  class _ExplodingModelStub:
    def save_pretrained(self, path, state_dict=None):
      with open(os.path.join(path, "model-00001-of-00002.safetensors"), "w") as f:
        f.write("partial")
      raise RuntimeError("simulated OOM during save")

  def _worker(self, model):
    worker = FFTTrainingWorker()
    worker.model = model
    worker.tokenizer = None
    worker.base_model_name = "base-model"
    return worker

  def test_overwrite_replaces_stale_shards_atomically(self) -> None:
    worker = self._worker(self._SavingModelStub())
    with tempfile.TemporaryDirectory() as tmp:
      dest = os.path.join(tmp, "ckpt")
      os.makedirs(dest)
      with open(os.path.join(dest, "model-00007-of-00099.safetensors"), "w") as f:
        f.write("stale-shard-from-a-previous-layout")

      result = worker.save_checkpoint(dest, {"kind": "weights"})

      self.assertEqual(result, {"path": dest})
      files = sorted(os.listdir(dest))
      self.assertIn("model.safetensors", files)
      self.assertIn("metadata.json", files)
      self.assertNotIn("model-00007-of-00099.safetensors", files)
      self.assertEqual([entry for entry in os.listdir(tmp) if entry != "ckpt"], [])

  def test_failed_save_leaves_existing_checkpoint_untouched(self) -> None:
    worker = self._worker(self._ExplodingModelStub())
    with tempfile.TemporaryDirectory() as tmp:
      dest = os.path.join(tmp, "ckpt")
      os.makedirs(dest)
      with open(os.path.join(dest, "model.safetensors"), "w") as f:
        f.write("known-good-weights")

      with self.assertRaisesRegex(RuntimeError, "simulated OOM"):
        worker.save_checkpoint(dest, {"kind": "weights"})

      with open(os.path.join(dest, "model.safetensors")) as f:
        self.assertEqual(f.read(), "known-good-weights")
      self.assertNotIn("metadata.json", os.listdir(dest))


class TestLoraUnembedTargets(unittest.TestCase):
  """The tinker SDK defaults train_unembed=True, but vLLM cannot apply
  lm_head adapters for our model families — the trainer must not emit them
  unless explicitly overridden."""

  def _worker(self, tied: bool):
    import torch.nn as nn

    class Attn(nn.Module):
      def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)

    class Base(nn.Module):
      def __init__(self):
        super().__init__()
        self.self_attn = Attn()
        self.lm_head = nn.Linear(4, 8)

    base = Base()
    base.config = types.SimpleNamespace(tie_word_embeddings=tied)
    return types.SimpleNamespace(base_model=base, base_model_name="test-model", lora_target_modules={}, linear_module_names=None)

  def _targets(self, worker, **kwargs) -> list[str]:
    config = lora_trainer_worker_module.LoraConfig(train_mlp=False, train_unembed=True, **kwargs)
    return lora_trainer_worker_module.LoraTrainingWorker.target_lora_modules(worker, config)

  def test_lm_head_skipped_by_default_for_vllm_loadability(self) -> None:
    with patch.dict(os.environ, {}, clear=False):
      os.environ.pop("OPEN_RL_LORA_TRAIN_UNEMBED", None)
      targets = self._targets(self._worker(tied=False))
    self.assertIn("self_attn.q_proj", targets)
    self.assertNotIn("lm_head", targets)

  def test_lm_head_trained_with_explicit_override(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_LORA_TRAIN_UNEMBED": "1"}):
      targets = self._targets(self._worker(tied=False))
    self.assertIn("lm_head", targets)

  def test_lm_head_never_trained_on_tied_embeddings(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_LORA_TRAIN_UNEMBED": "1"}):
      targets = self._targets(self._worker(tied=True))
    self.assertNotIn("lm_head", targets)


class TestLoraHybridAttentionTargets(unittest.TestCase):
  """Qwen3.5/3.6 checkpoints implement most layers as gated-deltanet linear
  attention: train_attn must cover their in_proj_*/out_proj projections, not
  only the q/k/v/o of the few full-attention layers."""

  def _worker(self):
    import torch.nn as nn

    class FullAttn(nn.Module):
      def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)

    class LinearAttn(nn.Module):
      def __init__(self):
        super().__init__()
        self.in_proj_qkv = nn.Linear(4, 12)
        self.in_proj_z = nn.Linear(4, 4)
        self.in_proj_b = nn.Linear(4, 2)
        self.in_proj_a = nn.Linear(4, 2)
        self.out_proj = nn.Linear(4, 4)
        self.conv1d = nn.Conv1d(12, 12, 3)

    class Base(nn.Module):
      def __init__(self):
        super().__init__()
        self.self_attn = FullAttn()
        self.linear_attn = LinearAttn()

    base = Base()
    base.config = types.SimpleNamespace(tie_word_embeddings=False)
    return types.SimpleNamespace(base_model=base, base_model_name="test-model", lora_target_modules={}, linear_module_names=None)

  def test_train_attn_targets_gdn_projections(self) -> None:
    config = lora_trainer_worker_module.LoraConfig(train_attn=True, train_mlp=False, train_unembed=False)
    targets = lora_trainer_worker_module.LoraTrainingWorker.target_lora_modules(self._worker(), config)
    self.assertIn("self_attn.q_proj", targets)
    for suffix in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
      self.assertIn(f"linear_attn.{suffix}", targets)
    # The GDN conv1d is not an nn.Linear and must never receive an adapter.
    self.assertNotIn("linear_attn.conv1d", targets)


class TestAdapterHubLayoutRemap(unittest.TestCase):
  """Adapters trained on the text-only view of a multimodal checkpoint must be
  saved with hub-layout keys (model.language_model.*) or the vLLM sampler
  silently applies no adapter at all."""

  def _write_adapter(self, tmp: str, keys: list[str]) -> str:
    from safetensors.torch import save_file

    weights_file = os.path.join(tmp, "adapter_model.safetensors")
    save_file({k: torch.zeros(2, 2) for k in keys}, weights_file, metadata={"format": "pt"})
    return weights_file

  def _remap(self, tmp: str, multimodal: bool) -> list[str]:
    from safetensors.torch import load_file

    worker = types.SimpleNamespace(base_is_multimodal=multimodal)
    lora_trainer_worker_module.LoraTrainingWorker._remap_adapter_to_hub_layout(worker, tmp)
    return sorted(load_file(os.path.join(tmp, "adapter_model.safetensors")))

  def test_text_layout_keys_gain_language_model_segment(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      self._write_adapter(
        tmp,
        ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", "base_model.model.lm_head.lora_A.weight"],
      )
      keys = self._remap(tmp, multimodal=True)
      self.assertEqual(
        keys,
        [
          "base_model.model.lm_head.lora_A.weight",
          "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight",
        ],
      )

  def test_hub_layout_and_text_only_bases_left_alone(self) -> None:
    text_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    hub_key = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
    with tempfile.TemporaryDirectory() as tmp:
      self._write_adapter(tmp, [hub_key])
      self.assertEqual(self._remap(tmp, multimodal=True), [hub_key])
    with tempfile.TemporaryDirectory() as tmp:
      self._write_adapter(tmp, [text_key])
      self.assertEqual(self._remap(tmp, multimodal=False), [text_key])


if __name__ == "__main__":
  unittest.main()


class TestAdapterAliasResolution(unittest.TestCase):
  """Alias-named sampler refs (tinker://<id>/sampler_weights/final) resolve to
  peft/<id>/<alias>; the adapter lives in the snapshot dir, so the alias must
  be a link to it — and pruning must never delete a snapshot an alias targets."""

  def _worker(self):
    worker = LoraTrainingWorker()
    worker.peft_model = _PeftModelStub({"m1": [torch.nn.Parameter(torch.tensor([1.0]))]})
    worker.peft_model.set_adapter("m1")
    return worker

  def test_alias_symlink_resolves_to_snapshot_dir(self) -> None:
    worker = self._worker()
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": tmp}):
      worker.save_adapter("m1", alias="final", session_label="sampler-3")
      root = os.path.join(tmp, "peft", "m1")
      self.assertTrue(os.path.isdir(os.path.join(root, "sampler-3")))
      alias_path = os.path.join(root, "final")
      self.assertTrue(os.path.islink(alias_path))
      self.assertTrue(os.path.isfile(os.path.join(alias_path, "adapter_config.json")))

  def test_alias_follows_the_latest_save(self) -> None:
    worker = self._worker()
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": tmp}):
      worker.save_adapter("m1", alias="final", session_label="sampler-1")
      worker.save_adapter("m1", alias="final", session_label="sampler-2")
      alias_path = os.path.join(tmp, "peft", "m1", "final")
      self.assertEqual(os.path.basename(os.path.realpath(alias_path)), "sampler-2")

  def test_prune_keeps_alias_target_alive(self) -> None:
    worker = self._worker()
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": tmp}):
      root = os.path.join(tmp, "peft", "m1")
      worker.save_adapter("m1", alias="final", session_label="sampler-1")
      for seq in range(2, 8):
        worker.save_adapter("m1", session_label=f"sampler-{seq}")
        os.utime(os.path.join(root, f"sampler-{seq}"), (seq * 1000, seq * 1000))
      remaining = {name for name in os.listdir(root) if name.startswith("sampler-")}
      self.assertIn("sampler-1", remaining)  # alias target survives pruning
      self.assertNotIn("sampler-2", remaining)  # oldest non-aliased is pruned
      self.assertTrue(os.path.isfile(os.path.join(root, "final", "adapter_config.json")))


class _GradTrackingModelStub:
  """Logits depend on a real parameter so backward leaves observable grads."""

  def __init__(self):
    self.w = torch.nn.Parameter(torch.zeros(()))
    self.config = types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace(_attn_implementation="eager"))

  def train(self):
    return None

  def parameters(self):
    yield self.w

  def __call__(self, input_ids, attention_mask=None, **_kwargs):
    base = torch.cos(input_ids.float().unsqueeze(-1) * 0.11 + torch.arange(7, dtype=torch.float32).view(1, 1, -1) * 0.13)
    return types.SimpleNamespace(logits=base + self.w)


class TestForwardOnly(unittest.TestCase):
  """TrainingClient.forward() must not accumulate gradients: the SDK's
  custom-loss path (DPO) sends real weights on the forward pass and the true
  gradient arrives as a separate linearized forward_backward afterwards."""

  def _worker(self) -> BaseTrainerWorker:
    worker = BaseTrainerWorker()
    worker.device = torch.device("cpu")
    worker.tokenizer = _TokenizerStub()
    return worker

  def _data(self):
    return [_datum([1, 2, 3], [2, 3, 4], weights=[1.0, 1.0, 1.0])]

  def test_forward_only_leaves_no_gradients(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_FUSED_LOGPROB": "0"}):
      model = _GradTrackingModelStub()
      worker = self._worker()
      result = worker.forward_backward(model, self._data(), "cross_entropy", forward_only=True)
      self.assertIsNone(model.w.grad)
      self.assertEqual(len(result["loss_fn_outputs"]), 1)

      model_trained = _GradTrackingModelStub()
      trained = worker.forward_backward(model_trained, self._data(), "cross_entropy")
      self.assertIsNotNone(model_trained.w.grad)
      self.assertEqual(result["loss_fn_outputs"][0]["logprobs"]["data"], trained["loss_fn_outputs"][0]["logprobs"]["data"])


class TestLoraDropoutDefault(unittest.TestCase):
  def test_dropout_defaults_off_for_unbiased_logprobs(self) -> None:
    self.assertEqual(lora_trainer_worker_module.LoraConfig().lora_dropout, 0.0)


class TestPreWrapTargetDiscovery(unittest.TestCase):
  """get_peft_model wraps modules in place; target discovery must use the
  module names captured at load time or a second adapter with a different
  config silently loses every already-wrapped projection."""

  def test_targets_come_from_captured_names_after_wrapping(self) -> None:
    base = torch.nn.Module()
    worker = types.SimpleNamespace(
      base_model=base,
      base_model_name="test-model",
      lora_target_modules={},
      linear_module_names=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj"],
    )
    config = lora_trainer_worker_module.LoraConfig(train_attn=True, train_mlp=True, train_unembed=False)
    targets = lora_trainer_worker_module.LoraTrainingWorker.target_lora_modules(worker, config)
    self.assertIn("self_attn.q_proj", targets)
    self.assertIn("mlp.gate_proj", targets)


class TestFusedHeadAdapterGuard(unittest.TestCase):
  """The fused path multiplies by head.weight directly; when lm_head carries
  an adapter the full-logits path must run so the adapter applies."""

  def _worker(self) -> BaseTrainerWorker:
    worker = BaseTrainerWorker()
    worker.device = torch.device("cpu")
    worker.tokenizer = _TokenizerStub()
    return worker

  def _data(self):
    return [_datum([1, 2, 3], [2, 3, 4], weights=[1.0, 1.0, 1.0])]

  def test_adapted_head_uses_full_logits_path(self) -> None:
    worker = self._worker()
    worker.output_head_is_adapted = True
    result = worker.forward_backward(_GradTrackingModelStub(), self._data(), "cross_entropy")
    self.assertEqual(len(result["loss_fn_outputs"]), 1)

  def test_unresolvable_backbone_errors_instead_of_silent_fallback(self) -> None:
    worker = self._worker()
    with self.assertRaisesRegex(RuntimeError, "OPEN_RL_FUSED_LOGPROB"):
      worker.forward_backward(_GradTrackingModelStub(), self._data(), "cross_entropy")
