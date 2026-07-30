import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

from server import gateway
from server.store import InMemoryStore


class GetInfoTest(unittest.TestCase):
  def setUp(self) -> None:
    patcher = patch.object(gateway, "store", InMemoryStore())
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_get_info_uses_base_model_env(self) -> None:
    with patch.dict(os.environ, {"BASE_MODEL": "env-model"}, clear=True):
      info = asyncio.run(gateway.get_info({"model_id": "model-a"}))

    self.assertEqual(info["model_name"], "env-model")
    self.assertEqual(info["model_data"]["tokenizer_id"], "env-model")
    self.assertEqual(info["model_id"], "model-a")

  def test_get_info_404s_without_base_model_env(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      response = asyncio.run(gateway.get_info({"model_id": "model-a"}))
    self.assertEqual(response.status_code, 404)

  def test_create_model_requires_base_model_payload(self) -> None:
    response = asyncio.run(gateway.create_model({}))
    self.assertEqual(response.status_code, 400)

  def test_create_model_accepts_base_model_payload(self) -> None:
    created = asyncio.run(gateway.create_model({"base_model": "my-model"}))
    model_id = created["request_id"]
    queued = asyncio.run(gateway.store.get_requests())
    self.assertEqual(queued[0]["model_id"], model_id)
    self.assertEqual(queued[0]["payload"]["base_model"], "my-model")


class GatewayPathTest(unittest.TestCase):
  def test_checkpoint_state_paths_are_model_scoped(self) -> None:
    old_tmp_dir = gateway.TMP_DIR
    with tempfile.TemporaryDirectory() as tmp_dir:
      gateway.TMP_DIR = tmp_dir
      self.addCleanup(setattr, gateway, "TMP_DIR", old_tmp_dir)

      self.assertEqual(
        gateway.checkpoint_state_path("job-a", "final"),
        os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "final"),
      )
      self.assertEqual(
        gateway.checkpoint_state_path("job-b", "final"),
        os.path.join(tmp_dir, "checkpoints", "job-b", "weights", "final"),
      )

  def test_checkpoint_state_paths_accept_explicit_output_directories(self) -> None:
    self.assertEqual(gateway.checkpoint_state_path("job-a", "/mnt/checkpoints/final"), "/mnt/checkpoints/final")

  def test_sampler_adapter_paths_are_per_snapshot(self) -> None:
    old_tmp_dir = gateway.TMP_DIR
    gateway.TMP_DIR = "/tmp/orl-test"
    self.addCleanup(setattr, gateway, "TMP_DIR", old_tmp_dir)

    self.assertEqual(
      gateway.sampler_adapter_path("tinker://model-1/sampler_weights/sampler-42"),
      os.path.join("/tmp/orl-test", "peft", "model-1", "sampler-42"),
    )
    # Distinct snapshots of the same model must never share a directory —
    # in-place overwrites let vLLM read a half-written adapter
    # ("<dir> doesn't contain tensors").
    self.assertNotEqual(
      gateway.sampler_adapter_path("tinker://model-1/sampler_weights/sampler-42"),
      gateway.sampler_adapter_path("tinker://model-1/sampler_weights/sampler-43"),
    )


class StatePathRoundTripTest(unittest.TestCase):
  """save_weights returns tinker:// paths and resume must resolve them back."""

  def _tmp_dir(self):
    old_tmp_dir = gateway.TMP_DIR
    tmp = tempfile.mkdtemp()
    gateway.TMP_DIR = tmp
    self.addCleanup(setattr, gateway, "TMP_DIR", old_tmp_dir)
    return tmp

  def test_tinker_state_path_forms(self) -> None:
    self.assertEqual(gateway.tinker_state_path("job-a", "final"), "tinker://job-a/weights/final")
    self.assertEqual(gateway.tinker_state_path("job-a", "/abs/path"), "/abs/path")
    self.assertEqual(gateway.tinker_state_path("job-a", "tinker://job-a/weights/x"), "tinker://job-a/weights/x")

  def test_resolve_state_ref_round_trips_saved_paths(self) -> None:
    tmp = self._tmp_dir()
    public = gateway.tinker_state_path("job-a", "step-42")
    self.assertEqual(
      gateway.resolve_state_ref(public),
      os.path.join(tmp, "checkpoints", "job-a", "weights", "step-42"),
    )

  def test_resolve_state_ref_ignores_non_state_refs(self) -> None:
    self.assertIsNone(gateway.resolve_state_ref("tinker://job-a/sampler_weights/sampler-1"))
    self.assertIsNone(gateway.resolve_state_ref("/abs/path"))
    self.assertIsNone(gateway.resolve_state_ref(None))

  def test_checkpoint_state_path_accepts_tinker_refs(self) -> None:
    tmp = self._tmp_dir()
    self.assertEqual(
      gateway.checkpoint_state_path("ignored-model", "tinker://job-a/weights/final"),
      os.path.join(tmp, "checkpoints", "job-a", "weights", "final"),
    )

  def test_create_model_from_state_resolves_tinker_refs(self) -> None:
    tmp = self._tmp_dir()

    class StoreStub:
      def __init__(self):
        self.requests = []
        self.futures = {}

      async def put_request(self, req_data):
        self.requests.append(req_data)

      async def set_future(self, req_id, result):
        self.futures[req_id] = result

    store = StoreStub()
    with patch.object(gateway, "store", store), patch.dict(os.environ, {}, clear=True):
      asyncio.run(gateway.create_model_from_state({"state_path": "tinker://job-a/weights/final"}))

    self.assertEqual(
      store.requests[0]["payload"]["state_path"],
      os.path.join(tmp, "checkpoints", "job-a", "weights", "final"),
    )

  def test_save_weights_payload_carries_round_trippable_public_path(self) -> None:
    self._tmp_dir()

    class StoreStub:
      def __init__(self):
        self.requests = []
        self.futures = {}

      async def put_request(self, req_data):
        self.requests.append(req_data)

      async def set_future(self, req_id, result):
        self.futures[req_id] = result

    store = StoreStub()
    with patch.object(gateway, "store", store), patch.dict(os.environ, {}, clear=True):
      asyncio.run(gateway.save_weights({"model_id": "job-a", "path": "step-7"}))

    payload = store.requests[0]["payload"]
    self.assertEqual(payload["public_path"], "tinker://job-a/weights/step-7")
    self.assertEqual(gateway.resolve_state_ref(payload["public_path"]), payload["state_path"])


class TrainingRunMetadataTest(unittest.TestCase):
  """The REST surface tinker-cookbook uses to verify renderer metadata on resume."""

  def test_user_metadata_round_trips_through_training_runs_endpoint(self) -> None:
    with patch.object(gateway, "store", InMemoryStore()), patch.dict(os.environ, {}, clear=True):
      resp = asyncio.run(gateway.create_model({"base_model": "base-model", "user_metadata": {"renderer_name": "gemma4"}}))
      run = asyncio.run(gateway.get_training_run(resp["request_id"]))

    self.assertEqual(run["training_run_id"], resp["request_id"])
    self.assertEqual(run["base_model"], "base-model")
    self.assertEqual(run["user_metadata"], {"renderer_name": "gemma4"})
    self.assertIn("last_request_time", run)

  def test_unknown_training_run_404s(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      response = asyncio.run(gateway.get_training_run("no-such-model"))
    self.assertEqual(response.status_code, 404)


class LoadWeightsWorkerResurrectionTest(unittest.TestCase):
  def test_fft_load_weights_relaunches_dead_trainer_worker(self) -> None:
    class StoreStub:
      def __init__(self):
        self.requests = []
        self.futures = {}

      async def put_request(self, req_data):
        self.requests.append(req_data)

      async def set_future(self, req_id, result):
        self.futures[req_id] = result

    class ManagerStub:
      def __init__(self):
        self.launched = []

      def launch_trainer(self, model_id, base_model=None):
        self.launched.append((model_id, base_model))

    store = StoreStub()
    manager = ManagerStub()
    with (
      patch.object(gateway, "store", store),
      patch.object(gateway, "fft_worker_manager", manager),
      patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}, clear=True),
    ):
      asyncio.run(gateway.load_weights({"model_id": "model-a", "path": "tinker://model-a/weights/final"}))

    self.assertEqual(manager.launched, [("model-a", None)])
    self.assertEqual(store.requests[0]["op"], "load_weights")

  def test_lora_load_weights_does_not_touch_worker_manager(self) -> None:
    class StoreStub:
      def __init__(self):
        self.requests = []
        self.futures = {}

      async def put_request(self, req_data):
        self.requests.append(req_data)

      async def set_future(self, req_id, result):
        self.futures[req_id] = result

    store = StoreStub()
    with patch.object(gateway, "store", store), patch.dict(os.environ, {}, clear=True):
      asyncio.run(gateway.load_weights({"model_id": "model-a", "path": "final"}))

    self.assertEqual(store.requests[0]["op"], "load_weights")


if __name__ == "__main__":
  unittest.main()


class RetryDedupeTest(unittest.TestCase):
  """The tinker SDK auto-retries mutating POSTs on timeouts/5xx; a replayed
  enqueue would apply gradients twice. The gateway must return the original
  request_id for a (op, model_id, seq_id) it has already enqueued."""

  def setUp(self) -> None:
    patcher = patch.object(gateway, "store", InMemoryStore())
    patcher.start()
    self.addCleanup(patcher.stop)
    gateway.enqueued_requests.clear()
    self.addCleanup(gateway.enqueued_requests.clear)

  def _fb_request(self, seq_id: int | None) -> dict:
    req: dict = {"model_id": "model-a", "forward_backward_input": {"data": [], "loss_fn": "cross_entropy"}}
    if seq_id is not None:
      req["seq_id"] = seq_id
    return req

  def test_retried_forward_backward_returns_original_request_and_enqueues_once(self) -> None:
    first = asyncio.run(gateway.forward_backward(self._fb_request(seq_id=7)))
    replay = asyncio.run(gateway.forward_backward(self._fb_request(seq_id=7)))
    self.assertEqual(first["request_id"], replay["request_id"])
    queued = asyncio.run(gateway.store.get_requests())
    self.assertEqual(len(queued), 1)

  def test_distinct_seq_ids_enqueue_separately(self) -> None:
    first = asyncio.run(gateway.forward_backward(self._fb_request(seq_id=7)))
    second = asyncio.run(gateway.forward_backward(self._fb_request(seq_id=8)))
    self.assertNotEqual(first["request_id"], second["request_id"])
    self.assertEqual(len(asyncio.run(gateway.store.get_requests())), 2)

  def test_same_seq_id_different_ops_do_not_collide(self) -> None:
    fb = asyncio.run(gateway.forward_backward(self._fb_request(seq_id=7)))
    optim = asyncio.run(gateway.optim_step({"model_id": "model-a", "seq_id": 7, "adam_params": {"learning_rate": 1e-4}}))
    self.assertNotEqual(fb["request_id"], optim["request_id"])

  def test_requests_without_seq_id_are_never_deduped(self) -> None:
    first = asyncio.run(gateway.forward_backward(self._fb_request(seq_id=None)))
    second = asyncio.run(gateway.forward_backward(self._fb_request(seq_id=None)))
    self.assertNotEqual(first["request_id"], second["request_id"])

  def test_retried_optim_step_is_absorbed(self) -> None:
    req = {"model_id": "model-a", "seq_id": 3, "adam_params": {"learning_rate": 1e-4}}
    first = asyncio.run(gateway.optim_step(req))
    replay = asyncio.run(gateway.optim_step(req))
    self.assertEqual(first["request_id"], replay["request_id"])
    self.assertEqual(len(asyncio.run(gateway.store.get_requests())), 1)

  def test_expired_entries_are_not_replayed(self) -> None:
    first = asyncio.run(gateway.forward_backward(self._fb_request(seq_id=7)))
    key = ("forward_backward", "model-a", 7)
    request_id, stamp = gateway.enqueued_requests[key]
    gateway.enqueued_requests[key] = (request_id, stamp - gateway.RETRY_DEDUPE_TTL_SECONDS - 1)
    second = asyncio.run(gateway.forward_backward(self._fb_request(seq_id=7)))
    self.assertNotEqual(first["request_id"], second["request_id"])

  def test_dedupe_map_is_bounded(self) -> None:
    for seq in range(gateway.RETRY_DEDUPE_MAX_ENTRIES + 10):
      gateway.remember_request("forward_backward", {"model_id": "m", "seq_id": seq}, f"req-{seq}")
    self.assertEqual(len(gateway.enqueued_requests), gateway.RETRY_DEDUPE_MAX_ENTRIES)
    self.assertNotIn(("forward_backward", "m", 0), gateway.enqueued_requests)


class ForwardOnlyEndpointTest(unittest.TestCase):
  """TrainingClient.forward() must enqueue a gradient-free pass; the SDK's
  custom-loss path sends real weights on it and would otherwise train twice."""

  def setUp(self) -> None:
    patcher = patch.object(gateway, "store", InMemoryStore())
    patcher.start()
    self.addCleanup(patcher.stop)
    gateway.enqueued_requests.clear()
    self.addCleanup(gateway.enqueued_requests.clear)

  def test_forward_endpoint_marks_forward_only(self) -> None:
    asyncio.run(gateway.forward({"model_id": "m", "forward_input": {"data": [], "loss_fn": "cross_entropy"}}))
    queued = asyncio.run(gateway.store.get_requests())
    self.assertTrue(queued[0]["payload"]["forward_only"])

  def test_forward_backward_endpoint_does_not(self) -> None:
    asyncio.run(gateway.forward_backward({"model_id": "m", "forward_backward_input": {"data": []}}))
    queued = asyncio.run(gateway.store.get_requests())
    self.assertNotIn("forward_only", queued[0]["payload"])
