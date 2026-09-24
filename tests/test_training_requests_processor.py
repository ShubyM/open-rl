"""The processor over a real worker and a tiny Llama, with only the model download replaced."""

import asyncio
import json
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from server import training_requests_processor as trp
from server.store import InMemoryStore
from training.trainer_worker import TrainerWorker


def tiny_llama(source: str = "tiny", device: torch.device | None = None):
  torch.manual_seed(0)
  config = LlamaConfig(
    vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64
  )
  return LlamaForCausalLM(config), None


def tiny_worker() -> TrainerWorker:
  worker = TrainerWorker()
  worker.device = torch.device("cpu")
  return worker


def create_model(model_id: str, fine_tuning_type: str = "lora", **config) -> dict:
  payload = {"base_model": "tiny", "fine_tuning_type": fine_tuning_type}
  payload["lora_config" if fine_tuning_type == "lora" else "full_config"] = config
  return {"request_id": f"create-{model_id}", "model_id": model_id, "op": "create_model", "payload": payload}


def forward_backward(model_id: str) -> dict:
  data = [{"model_input": {"chunks": [{"tokens": [3, 4, 5, 6]}]}, "loss_fn_inputs": {"target_tokens": [1, 2, 3, 4], "weights": [1.0, 1.0, 1.0, 1.0]}}]
  return {"request_id": f"fb-{model_id}", "model_id": model_id, "op": "forward_backward", "payload": {"data": data, "loss_fn": "cross_entropy"}}


def optim_step(model_id: str) -> dict:
  return {"request_id": f"os-{model_id}", "model_id": model_id, "op": "optim_step", "payload": {"adam_params": {"learning_rate": 1e-2}}}


class ScriptedStore(InMemoryStore):
  """Hands out the given batches, then ends the processor loop."""

  def __init__(self, batches, events=None):
    super().__init__()
    self.batches = list(batches)
    self.events = events
    self.results = {}
    self.queried_model_ids = []

  async def get_requests_for_model(self, model_id):
    self.queried_model_ids.append(model_id)
    return await self.get_requests()

  async def get_requests(self, active_set_id=None):
    if self.batches:
      return self.batches.pop(0)
    raise asyncio.CancelledError()

  async def set_future(self, request_id, result):
    if self.events is not None:
      self.events.append(("set_future", request_id))
    self.results[request_id] = result


class SlicerStub:
  faulted = None

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


class ProcessorTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.tmp = tempfile.TemporaryDirectory()
    self.env = patch.dict(
      os.environ, {"OPEN_RL_TMP_DIR": self.tmp.name, "REDIS_URL": "redis://localhost:6379", "OPEN_RL_WEIGHT_SYNC_STRATEGY": "full"}
    )
    self.env.start()
    self.loader = patch.object(trp.TrainerWorker, "load_base_model", autospec=True, side_effect=self.install_tiny)
    self.loader.start()

  def tearDown(self) -> None:
    self.loader.stop()
    self.env.stop()
    self.tmp.cleanup()

  @staticmethod
  def install_tiny(worker: TrainerWorker, base_model_name: str) -> None:
    if worker.base is None:
      worker.base, worker.tokenizer = tiny_llama()
      worker.base_name = base_model_name

  async def test_lora_create_model_makes_an_adapter_and_exports_it_for_the_sampler(self) -> None:
    worker = tiny_worker()
    store = ScriptedStore([])
    processor = trp.TrainingRequestsProcessor(store, worker)

    await processor.process_request(create_model("adapter-a", seed=123, rank=2))

    model = worker.models["adapter-a"]
    self.assertTrue(model.is_lora)
    self.assertTrue(all(param.shape[0] == 2 or param.shape[1] == 2 for param in model.params.values()))
    self.assertEqual(
      store.results["create-adapter-a"],
      {"base_model": "tiny", "model_id": "adapter-a", "fine_tuning_type": "lora", "rank": 2, "type": "model_created"},
    )
    with open(os.path.join(self.tmp.name, "peft", "adapter-a", "adapter-a", "adapter_config.json")) as f:
      self.assertEqual(json.load(f)["r"], 2)

  async def test_every_lora_step_lands_in_the_sampler_directory(self) -> None:
    worker = tiny_worker()
    store = ScriptedStore([])
    processor = trp.TrainingRequestsProcessor(store, worker)
    await processor.process_request(create_model("adapter-a", rank=2))
    adapter_file = os.path.join(self.tmp.name, "peft", "adapter-a", "adapter-a", "adapter_model.safetensors")
    before = os.path.getmtime(adapter_file)
    os.utime(adapter_file, (before - 10, before - 10))

    await processor.process_request(forward_backward("adapter-a"))
    await processor.process_request(optim_step("adapter-a"))

    self.assertEqual(store.results["os-adapter-a"]["type"], "optim_step_completed")
    self.assertGreater(os.path.getmtime(adapter_file), before - 10)

  async def test_full_create_model_runs_under_the_lease_and_takes_the_offload_setting(self) -> None:
    worker = tiny_worker()
    events = []
    store = ScriptedStore([[create_model("model-a", "full", seed=123, cpu_offload=False)]], events=events)
    slicer = SlicerStub(events)

    await trp.run_training_requests_processor(worker, "model-a", time_slicer=slicer, store=store)

    self.assertEqual([event[0] for event in events], ["register", "acquire", "release", "set_future", "unregister", "close"])
    for event in events:
      if len(event) >= 2 and event[0] != "set_future":
        self.assertEqual(event[1].name, "trainer-model-a")
        self.assertEqual(event[1].claim, "trainers")
    self.assertFalse(worker.models["model-a"].is_lora)
    self.assertFalse(worker.cpu_offload)
    self.assertEqual(store.results["create-model-a"]["type"], "model_created")
    self.assertEqual(store.queried_model_ids, ["model-a", "model-a"])

  async def test_full_sampler_save_writes_a_versioned_checkpoint(self) -> None:
    worker = tiny_worker()
    store = ScriptedStore([])
    processor = trp.TrainingRequestsProcessor(store, worker, "model-a", time_slicer=SlicerStub())
    await processor.process_request(create_model("model-a", "full"))

    await processor.process_request(
      {
        "request_id": "save-a",
        "model_id": "model-a",
        "op": "save_weights_for_sampler",
        "payload": {"path": "tinker://model-a/sampler_weights/final", "sampling_session_id": "tinker://model-a/sampler_weights/sampler-7"},
      }
    )

    version = os.path.join(self.tmp.name, "sampler_full", "model-a", "sampler_weights", "final")
    self.assertTrue(os.path.exists(os.path.join(version, "model.safetensors")))
    with open(os.path.join(version, "metadata.json")) as f:
      self.assertEqual(json.load(f)["kind"], "sampler")
    self.assertEqual(
      store.results["save-a"],
      {
        "path": "tinker://model-a/sampler_weights/final",
        "sampling_session_id": "tinker://model-a/sampler_weights/sampler-7",
        "type": "sampler_weights_saved",
      },
    )

  async def test_full_sampler_save_under_the_delta_strategy_writes_what_changed(self) -> None:
    worker = tiny_worker()
    store = ScriptedStore([])
    processor = trp.TrainingRequestsProcessor(store, worker, "model-a", time_slicer=SlicerStub())
    await processor.process_request(create_model("model-a", "full", weight_sync_strategy="delta"))
    await processor.process_request(forward_backward("model-a"))
    await processor.process_request(optim_step("model-a"))

    await processor.process_request(
      {
        "request_id": "save-a",
        "model_id": "model-a",
        "op": "save_weights_for_sampler",
        "payload": {"path": "tinker://model-a/sampler_weights/step-1"},
      }
    )

    version = os.path.join(self.tmp.name, "sampler_full", "model-a", "sampler_weights", "step-1")
    self.assertTrue(os.path.exists(os.path.join(version, "delta.safetensors")))
    with open(os.path.join(version, "metadata.json")) as f:
      metadata = json.load(f)
    self.assertEqual(metadata["format"], "sparse_delta")
    self.assertGreater(metadata["changed_elements"], 0)

  async def test_an_unknown_weight_sync_strategy_fails_the_request(self) -> None:
    store = ScriptedStore([])
    processor = trp.TrainingRequestsProcessor(store, tiny_worker(), "model-a", time_slicer=SlicerStub())
    await processor.process_request(create_model("model-a", "full", weight_sync_strategy="bogus"))
    self.assertEqual(store.results["create-model-a"]["type"], "RequestFailedResponse")
    self.assertIn("Invalid weight_sync_strategy", store.results["create-model-a"]["error_message"])

  async def test_sampling_from_the_trainer_is_refused(self) -> None:
    store = ScriptedStore([])
    processor = trp.TrainingRequestsProcessor(store, tiny_worker())
    await processor.process_request(create_model("adapter-a", rank=2))
    await processor.process_request(
      {"request_id": "s-1", "model_id": "adapter-a", "op": "sample", "payload": {"prompt_tokens": [1, 2, 3], "max_tokens": 4}}
    )
    self.assertEqual(store.results["s-1"]["type"], "RequestFailedResponse")
    self.assertIn("vLLM sampler", store.results["s-1"]["error_message"])

  async def test_a_time_sliced_processor_requires_redis(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.tmp.name}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      await trp.run_training_requests_processor(tiny_worker(), "model-a", time_sliced=True, store=ScriptedStore([]))

  async def test_a_time_sliced_processor_uses_the_default_slicer_client(self) -> None:
    store = ScriptedStore([])
    slicer = SlicerStub()
    with patch.object(trp, "time_slicer_client_from_env", return_value=slicer) as from_env:
      await trp.run_training_requests_processor(tiny_worker(), "model-a", time_sliced=True, store=store)
    from_env.assert_called_once_with()
    self.assertEqual([event[0] for event in slicer.events], ["register", "unregister", "close"])

  async def test_a_shared_processor_gets_no_slicer(self) -> None:
    store = ScriptedStore([])
    with patch.object(trp, "time_slicer_client_from_env") as from_env, self.assertRaises(asyncio.CancelledError):
      await trp.TrainingRequestsProcessor(store, tiny_worker()).run_once()
    from_env.assert_not_called()


if __name__ == "__main__":
  unittest.main()
