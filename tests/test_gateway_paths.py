import asyncio
import json
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

  def test_get_info_prefers_the_models_own_base_model(self) -> None:
    meta = json.dumps({"base_model": "google/gemma-4-e2b", "fine_tuning_type": "full"})
    asyncio.run(gateway.store.set_value("open_rl:model_meta:model-g", meta))
    with patch.dict(os.environ, {"BASE_MODEL": "Qwen/Qwen2.5-0.5B"}, clear=True):
      info = asyncio.run(gateway.get_info({"model_id": "model-g"}))
      via_sampler_ref = asyncio.run(gateway.get_info({"model_id": "tinker://model-g/sampler_weights/sampler-1"}))
      unknown = asyncio.run(gateway.get_info({"model_id": "model-unknown"}))

    # The client loads its tokenizer from this name, so it must be the job's model.
    self.assertEqual(info["model_name"], "google/gemma-4-e2b")
    self.assertEqual(info["model_data"]["tokenizer_id"], "google/gemma-4-e2b")
    self.assertEqual(via_sampler_ref["model_name"], "google/gemma-4-e2b")
    self.assertEqual(unknown["model_name"], "Qwen/Qwen2.5-0.5B")

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
    self.assertEqual(queued[0]["payload"], {})
    meta = json.loads(gateway.store.get_value_sync(f"open_rl:model_meta:{model_id}"))
    self.assertEqual(meta["base_model"], "my-model")


class SaveSeqIdZeroTest(unittest.TestCase):
  def setUp(self) -> None:
    patcher = patch.object(gateway, "store", InMemoryStore())
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_the_first_saves_zero_seq_id_is_kept(self) -> None:
    # The client's counter is 0-based; 0 must not fall back to a timestamp id.
    asyncio.run(gateway.save_weights_for_sampler({"model_id": "job-a", "sampling_session_seq_id": 0}))
    asyncio.run(gateway.save_weights({"model_id": "job-a", "seq_id": 0}))
    queued = asyncio.run(gateway.store.get_requests())
    self.assertEqual(queued[0]["payload"]["sampling_session_id"], "tinker://job-a/sampler_weights/sampler-0")
    self.assertTrue(queued[1]["payload"]["state_path"].endswith("job-a-samp-0"))


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

  def test_a_tinker_path_names_the_model_that_saved_it(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(gateway, "TMP_DIR", tmp_dir):
      state_dir = os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "step-5")
      self.assertEqual(gateway.tinker_state_path(state_dir), "tinker://job-a/weights/step-5")
      # A resuming job passes the dead job's path under its own model id.
      self.assertEqual(gateway.checkpoint_state_path("job-b", "tinker://job-a/weights/step-5"), state_dir)
      self.assertEqual(gateway.tinker_state_path("/elsewhere/final"), "/elsewhere/final")
      # Only weights paths are checkpoints. A sampler path is refused, not resolved under the caller.
      self.assertIsNone(gateway.tinker_checkpoint_dir("tinker://job-a/sampler_weights/sampler-3"))
      refused = asyncio.run(gateway.load_weights({"model_id": "job-b", "path": "tinker://job-a/sampler_weights/sampler-3"}))
      self.assertEqual(refused.status_code, 400)

  def test_save_state_keeps_the_optimizer_and_answers_with_a_tinker_path(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(gateway, "TMP_DIR", tmp_dir):
      asyncio.run(gateway.save_weights({"model_id": "job-a", "path": "step-5"}))
      queued = asyncio.run(gateway.store.get_requests())
      self.assertEqual(queued[0]["op"], "save_state")
      self.assertTrue(queued[0]["payload"]["include_optimizer"])
      saved = gateway.translate_future_result({"type": "state_saved", "path": queued[0]["payload"]["state_path"]})
    self.assertEqual(saved, {"type": "save_weights", "path": "tinker://job-a/weights/step-5"})

  def test_weights_info_reads_the_checkpoint_on_disk(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir, patch.object(gateway, "TMP_DIR", tmp_dir):
      state_dir = os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "step-5")
      os.makedirs(os.path.join(state_dir, "job-a"))
      with open(os.path.join(state_dir, "metadata.json"), "w") as f:
        json.dump({"base_model": "google/gemma-4-e2b", "model_id": "job-a", "has_optimizer": True}, f)
      with open(os.path.join(state_dir, "job-a", "adapter_config.json"), "w") as f:
        json.dump({"r": 8}, f)
      info = asyncio.run(gateway.weights_info({"tinker_path": "tinker://job-a/weights/step-5"}))
      missing = asyncio.run(gateway.weights_info({"tinker_path": "tinker://job-a/weights/never"}))
    self.assertEqual(info["base_model"], "google/gemma-4-e2b")
    self.assertTrue(info["is_lora"])
    self.assertEqual(info["lora_rank"], 8)
    self.assertEqual(missing.status_code, 404)

  def test_checkpoint_state_paths_accept_explicit_output_directories(self) -> None:
    self.assertEqual(gateway.checkpoint_state_path("job-a", "/mnt/checkpoints/final"), "/mnt/checkpoints/final")


if __name__ == "__main__":
  unittest.main()


class ProtobufWireTest(unittest.TestCase):
  """Tinker SDK >= 0.25 sends forward_backward as protobuf and only reads
  forward_backward and sample results as protobuf."""

  FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "fwdbwd_request_tinker_0.29.0.pb")

  def setUp(self) -> None:
    from fastapi.testclient import TestClient

    patcher = patch.object(gateway, "store", InMemoryStore())
    patcher.start()
    self.addCleanup(patcher.stop)
    self.client = TestClient(gateway.app)

  def _queued(self) -> list[dict]:
    return asyncio.run(gateway.store.get_requests())

  def test_protobuf_and_json_forward_backward_queue_the_same_request(self) -> None:
    with open(self.FIXTURE, "rb") as fh:
      body = fh.read()
    proto = self.client.post("/api/v1/forward_backward", content=body, headers={"Content-Type": "application/x-protobuf"})
    self.assertEqual(proto.status_code, 200, proto.text)
    from_proto = self._queued()[0]

    with open(self.FIXTURE[:-3] + ".json") as fh:
      json_body = json.load(fh)
    as_json = self.client.post("/api/v1/forward_backward", json=json_body)
    self.assertEqual(as_json.status_code, 200, as_json.text)
    from_json = self._queued()[0]

    self.assertEqual(from_proto["op"], "forward_backward")
    self.assertEqual(from_proto["model_id"], "model-abc")
    self.assertEqual(from_proto["payload"], from_json["payload"])
    self.assertEqual(from_proto["payload"]["loss_fn"], "importance_sampling")
    self.assertEqual(from_proto["payload"]["loss_config"], {"clip_range": 0.2, "kl_coeff": 0.01, "mode": "token"})

  def test_forward_only_protobuf_goes_to_the_same_op_as_the_json_forward_route(self) -> None:
    from server.proto import tinker_public_pb2 as pb

    msg = pb.ForwardBackwardRequest(model_id="model-abc", seq_id=1, loss_fn="cross_entropy", forward_only=True)
    response = self.client.post("/api/v1/forward_backward", content=msg.SerializeToString(), headers={"Content-Type": "application/x-protobuf"})
    self.assertEqual(response.status_code, 200, response.text)
    self.assertEqual(self._queued()[0]["op"], "forward_backward")

  def test_bad_bodies_are_client_errors_not_500s(self) -> None:
    garbage = self.client.post("/api/v1/forward_backward", content=b"\xff\xfe not proto", headers={"Content-Type": "application/x-protobuf"})
    self.assertEqual(garbage.status_code, 400, garbage.text)
    compressed = self.client.post(
      "/api/v1/forward_backward", content=b"x", headers={"Content-Type": "application/x-protobuf", "Content-Encoding": "zstd"}
    )
    self.assertEqual(compressed.status_code, 415, compressed.text)
    other = self.client.post("/api/v1/forward_backward", content=b"x", headers={"Content-Type": "text/plain"})
    self.assertEqual(other.status_code, 415, other.text)
    # A non-JSON body on a plain `req: dict` route used to crash FastAPI's 422
    # handler while it JSON-encoded the raw bytes, turning it into a 500.
    binary_to_dict_route = self.client.post("/api/v1/optim_step", content=b"\x8a\xff", headers={"Content-Type": "application/x-protobuf"})
    self.assertEqual(binary_to_dict_route.status_code, 422, binary_to_dict_route.text)
    self.assertNotIn("\x8a", binary_to_dict_route.text)

  def test_retrieve_future_answers_protobuf_only_when_asked_and_only_for_proto_types(self) -> None:
    from server.proto import tinker_public_pb2 as pb

    asyncio.run(
      gateway.store.set_future(
        "samp-1", {"type": "sample_completed", "sequences": [{"tokens": [1, 2], "logprobs": [-0.5, -1.0], "stop_reason": "stop"}]}
      )
    )
    asyncio.run(gateway.store.set_future("optim-1", {"type": "optim_step_completed", "metrics": {"grad_norm:mean": 0.0}}))

    as_json = self.client.post("/api/v1/retrieve_future", json={"request_id": "samp-1"})
    self.assertEqual(as_json.status_code, 200)
    self.assertTrue(as_json.headers["content-type"].startswith("application/json"))
    self.assertEqual(as_json.json()["type"], "sample")

    as_proto = self.client.post("/api/v1/retrieve_future", json={"request_id": "samp-1"}, headers={"Accept": "application/x-protobuf"})
    self.assertEqual(as_proto.status_code, 200)
    self.assertEqual(as_proto.headers["content-type"], "application/x-protobuf")
    msg = pb.SampleResponse()
    msg.ParseFromString(as_proto.content)
    self.assertEqual(msg.sequences[0].stop_reason, pb.STOP_REASON_STOP)

    optim = self.client.post("/api/v1/retrieve_future", json={"request_id": "optim-1"}, headers={"Accept": "application/x-protobuf"})
    self.assertTrue(optim.headers["content-type"].startswith("application/json"))
    self.assertEqual(optim.json()["type"], "optim_step")


class SampleSequenceIdsTest(unittest.TestCase):
  """Tinker SDK >= 0.25 asserts that every asample promise carries one
  sequence id per requested sample."""

  def setUp(self) -> None:
    patcher = patch.object(gateway, "store", InMemoryStore())
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_asample_promise_carries_one_id_per_sample(self) -> None:
    with patch.object(gateway, "get_sampler_backend", return_value="torch"):
      promise = asyncio.run(gateway.asample({"model_id": "job-a", "prompt": {"chunks": [{"tokens": [1, 2]}]}, "num_samples": 3}))
    self.assertEqual(len(promise["sample_sequence_ids"]), 3)
    self.assertEqual(len(set(promise["sample_sequence_ids"])), 3)
    self.assertTrue(all(sid.startswith(promise["request_id"]) for sid in promise["sample_sequence_ids"]))

  def test_asample_defaults_to_a_single_sample(self) -> None:
    with patch.object(gateway, "get_sampler_backend", return_value="torch"):
      promise = asyncio.run(gateway.asample({"model_id": "job-a", "prompt": {"chunks": [{"tokens": [1]}]}}))
    self.assertEqual(len(promise["sample_sequence_ids"]), 1)
