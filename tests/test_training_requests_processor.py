"""Request dispatch through the training requests processors.

These tests check plumbing: that a queued request reaches the right worker
method with the right arguments, that the result has the shape the gateway
translates, and that the time-slicer window brackets the work. The workers
are fakes on purpose. Training behavior lives in test_trainer_worker.py.
"""

import asyncio
import os
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

from server import training_requests_processor as processor_module
from training.fft_trainer_worker import FFTTrainingWorker


class FakeFFTWorker(FFTTrainingWorker):
  """Subclasses the real worker only because run_training_requests_processor picks a processor by isinstance."""

  def __init__(self):
    self.created_models = []
    self.saved_states = []

  def create_model(self, base_model_name, model_id, config):
    self.created_models.append((base_model_name, model_id, config))

  def save_state(self, model_id, state_path, include_optimizer=False, kind="state"):
    self.saved_states.append((model_id, state_path, include_optimizer, kind))
    return {"path": state_path}

  # The processor brackets every batch with wake_up and sleep.
  def wake_up(self):
    pass

  def sleep(self):
    pass


class FakeLoraWorker:
  def __init__(self):
    self.created_models = []

  def create_model(self, base_model_name, model_id, config):
    self.created_models.append((base_model_name, model_id, config))


class FakeStore:
  def __init__(self, batches=(), events=None):
    self.results = {}
    self.values = {}
    self.events = events
    self.batches = list(batches)
    self.queried_model_ids = []

  async def set_future(self, req_id, result):
    if self.events is not None:
      self.events.append(("set_future", req_id))
    self.results[req_id] = result

  async def get_requests_for_model(self, model_id):
    self.queried_model_ids.append(model_id)
    if self.batches:
      return self.batches.pop(0)
    raise asyncio.CancelledError()

  async def set_value(self, key, value):
    self.values[key] = value

  async def get_value(self, key):
    return self.values.get(key)

  def get_value_sync(self, key):
    return self.values.get(key)

  async def record_accel_usage_event(self, claim_id, event_data):
    pass


class FakeTimeSlicer:
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


def create_model_request(model_id: str, **payload) -> dict:
  return {"request_id": "req-a", "model_id": model_id, "op": "create_model", "payload": payload}


REDIS_ENV = {"REDIS_URL": "redis://localhost:6379"}


class ParseDatumTest(unittest.TestCase):
  def test_chunked_model_input_is_flattened_and_loss_inputs_are_wrapped(self):
    d = processor_module.parse_datum(
      {
        "model_input": {"chunks": [{"tokens": [1, 2]}, {"tokens": [3]}]},
        "loss_fn_inputs": {"target_tokens": [2, 3, 4], "weights": {"data": [1.0, 0.5, 0.25]}},
      }
    )

    self.assertEqual(d.model_input, [1, 2, 3])
    self.assertEqual(d.loss_fn_inputs["target_tokens"].data, [2, 3, 4])
    self.assertEqual(d.loss_fn_inputs["weights"].data, [1.0, 0.5, 0.25])


class LoraProcessorTest(unittest.IsolatedAsyncioTestCase):
  async def test_create_model_passes_a_typed_lora_config_and_reports_the_rank(self):
    worker = FakeLoraWorker()
    store = FakeStore()
    processor = processor_module.LoraTrainingRequestsProcessor(store, worker)

    await processor.process_request(create_model_request("adapter-a", base_model="base-model", lora_config={"seed": 123, "rank": 2}), "adapter-a")

    base_model, model_id, config = worker.created_models[0]
    self.assertEqual((base_model, model_id, config.seed, config.rank), ("base-model", "adapter-a", 123, 2))
    self.assertEqual(store.results["req-a"]["type"], "model_created")
    self.assertEqual(store.results["req-a"]["rank"], 2)
    self.assertEqual(store.results["req-a"]["fine_tuning_type"], "lora")


class FFTProcessorTest(unittest.IsolatedAsyncioTestCase):
  def processor(self, store, worker, time_slicer=None):
    with patch.dict(os.environ, REDIS_ENV):
      return processor_module.FFTTrainingRequestsProcessor(store, worker, "model-a", time_slicer=time_slicer or FakeTimeSlicer())

  async def test_create_model_passes_a_typed_full_config(self):
    worker = FakeFFTWorker()
    store = FakeStore()

    await self.processor(store, worker).process_request(
      create_model_request("model-a", base_model="base-model", full_config={"seed": 123}), "model-a"
    )

    base_model, model_id, config = worker.created_models[0]
    self.assertEqual((base_model, model_id, config.seed), ("base-model", "model-a", 123))
    self.assertEqual(store.results["req-a"]["type"], "model_created")
    self.assertEqual(store.results["req-a"]["fine_tuning_type"], "full")

  async def test_sampler_weights_are_saved_under_the_tmp_dir_from_the_tinker_path(self):
    worker = FakeFFTWorker()
    store = FakeStore()
    request = {
      "request_id": "req-a",
      "model_id": "model-a",
      "op": "save_weights_for_sampler",
      "payload": {"path": "tinker://model-a/sampler_weights/final", "sampling_session_id": "tinker://model-a/sampler_weights/sampler-7"},
    }

    with patch.dict(os.environ, {"OPEN_RL_TMP_DIR": "/tmp/open-rl-test"}):
      await self.processor(store, worker).process_request(request, "model-a")

    self.assertEqual(worker.saved_states, [("model-a", "/tmp/open-rl-test/sampler_full/model-a/sampler_weights/final", False, "sampler")])
    self.assertEqual(store.results["req-a"]["type"], "sampler_weights_saved")
    self.assertEqual(store.results["req-a"]["path"], "tinker://model-a/sampler_weights/final")

  async def test_result_is_published_only_after_the_accelerator_is_released(self):
    events = []
    worker = FakeFFTWorker()
    store = FakeStore(batches=[[create_model_request("model-a", base_model="base-model", full_config={"seed": 123})]], events=events)

    await self.processor(store, worker, FakeTimeSlicer(events)).run_once()

    self.assertEqual([event[0] for event in events], ["acquire", "release", "set_future"])
    self.assertEqual(store.results["req-a"]["type"], "model_created")

  async def test_full_mode_requires_redis(self):
    with patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      await processor_module.run_training_requests_processor(FakeFFTWorker(), "model-a")

  async def test_run_registers_the_trainer_workload_for_the_whole_run(self):
    worker = FakeFFTWorker()
    store = FakeStore(batches=[[create_model_request("model-a", base_model="base-model", full_config={"seed": 123})]])
    time_slicer = FakeTimeSlicer()

    with (
      patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true", **REDIS_ENV}),
      patch.object(processor_module, "get_store", return_value=store),
    ):
      await processor_module.run_training_requests_processor(worker, "model-a", time_slicer=time_slicer)

    self.assertEqual([event[0] for event in time_slicer.events], ["register", "acquire", "release", "unregister", "close"])
    workloads = {event[1] for event in time_slicer.events if len(event) > 1}
    self.assertEqual({(w.name, w.claim) for w in workloads}, {("trainer-model-a", "trainers")})
    self.assertEqual(store.results["req-a"]["model_id"], "model-a")

  async def test_run_builds_the_time_slicer_client_from_env_when_none_is_given(self):
    time_slicer = FakeTimeSlicer()

    with (
      patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true", **REDIS_ENV}, clear=True),
      patch.object(processor_module, "get_store", return_value=FakeStore()),
      patch.object(processor_module, "time_slicer_client_from_env", return_value=time_slicer) as from_env,
    ):
      await processor_module.run_training_requests_processor(FakeFFTWorker(), "model-a")

    from_env.assert_called_once_with()
    self.assertEqual([event[0] for event in time_slicer.events], ["register", "unregister", "close"])
