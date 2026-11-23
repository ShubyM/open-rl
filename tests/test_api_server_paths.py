import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient

from server import api_server
from server.api import paths, wire
from server.proto import tinker_public_pb2 as pb
from server.store import InMemoryStateStore, InMemoryStore
from server.worker_manager import LocalWorkerManager
from tests.test_session_lifecycle import RuntimeManager


class ApiServerTest(unittest.TestCase):
  """Requests go through a fresh `create_app(...)` ASGI app (no module-global monkeypatching)."""

  def _make_app(self, **kwargs):
    self.store = kwargs.pop("store", InMemoryStore())
    self.state = kwargs.pop("state", InMemoryStateStore())
    self.app = api_server.create_app(store=self.store, state=self.state, **kwargs)
    self.gw = self.app.state.server
    self.client = TestClient(self.app)

  def setUp(self) -> None:
    self._make_app()

  def post(self, path: str, body: dict, **kwargs):
    return self.client.post(f"/api/v1/{path}", json=body, **kwargs)

  def post_bytes(self, path: str, body: bytes, media_type: str):
    return self.client.post(f"/api/v1/{path}", content=body, headers={"Content-Type": media_type})

  def queued(self) -> list[dict]:
    return asyncio.run(self.store.get_requests())


class GetInfoTest(ApiServerTest):
  def test_get_info_uses_base_model_env(self) -> None:
    with patch.dict(os.environ, {"BASE_MODEL": "env-model"}, clear=True):
      info = self.post("get_info", {"model_id": "model-a"}).json()
    self.assertEqual((info["model_name"], info["model_data"]["tokenizer_id"], info["model_id"]), ("env-model", "env-model", "model-a"))

  def test_get_info_prefers_the_models_own_base_model(self) -> None:
    meta = json.dumps({"base_model": "google/gemma-4-e2b", "fine_tuning_type": "full"})
    asyncio.run(self.state.set_value("open_rl:model_meta:model-g", meta))
    with patch.dict(os.environ, {"BASE_MODEL": "Qwen/Qwen2.5-0.5B"}, clear=True):
      info = self.post("get_info", {"model_id": "model-g"}).json()
      via_ref = self.post("get_info", {"model_id": "tinker://model-g/sampler_weights/sampler-1"}).json()
      unknown = self.post("get_info", {"model_id": "model-unknown"}).json()
    self.assertEqual((info["model_name"], info["model_data"]["tokenizer_id"]), ("google/gemma-4-e2b", "google/gemma-4-e2b"))
    self.assertEqual((via_ref["model_name"], unknown["model_name"]), ("google/gemma-4-e2b", "Qwen/Qwen2.5-0.5B"))

  def test_get_info_reports_the_lora_rank_the_job_was_created_with(self) -> None:
    asyncio.run(self.state.set_value("open_rl:model_meta:model-r32", json.dumps({"base_model": "g", "lora_config": {"rank": 32}})))
    full_meta = json.dumps({"base_model": "g", "fine_tuning_type": "full", "lora_config": {"rank": 32}})
    asyncio.run(self.state.set_value("open_rl:model_meta:model-full", full_meta))
    with patch.dict(os.environ, {"BASE_MODEL": "Qwen/Qwen2.5-0.5B"}, clear=True):
      ranks = [
        self.post("get_info", {"model_id": mid}).json()["lora_rank"]
        for mid in ("model-r32", "tinker://model-r32/sampler_weights/sampler-1", "model-full", "model-unknown")
      ]
    self.assertEqual(ranks, [32, 32, 16, 16])

  def test_get_info_404s_without_base_model_env(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      response = self.post("get_info", {"model_id": "model-a"})
    self.assertEqual((response.status_code, response.json()), (404, {"error": "No base model is configured"}))

  def test_create_model_requires_base_model_payload(self) -> None:
    response = self.post("create_model", {})
    self.assertEqual(response.status_code, 422)
    self.assertEqual(response.json()["detail"][0]["loc"], ["body", "base_model"])

  def test_a_missing_model_id_is_a_422_that_names_it(self) -> None:
    for path in ("delete_model", "optim_step", "save_weights", "save_weights_for_sampler", "load_weights"):
      with self.subTest(path=path):
        response = self.post(path, {})
        self.assertEqual(response.status_code, 422)
        self.assertIn(["body", "model_id"], [error["loc"] for error in response.json()["detail"]])

  def test_create_model_accepts_base_model_payload(self) -> None:
    model_id = self.post("create_model", {"base_model": "my-model"}).json()["request_id"]
    queued = self.queued()
    self.assertEqual((queued[0]["model_id"], queued[0]["payload"]["base_model"]), (model_id, "my-model"))
    self.assertEqual(json.loads(self.state.get_value_sync(f"open_rl:model_meta:{model_id}"))["base_model"], "my-model")


class ErrorShapeTest(ApiServerTest):
  def test_an_unknown_route_answers_the_shared_error_shape(self) -> None:
    response = self.client.get("/api/v1/no_such_endpoint")
    self.assertEqual((response.status_code, response.json()), (404, {"error": "Not Found"}))

  def test_a_body_of_the_wrong_shape_is_a_422_that_names_the_field(self) -> None:
    response = self.post("get_info", {"model_id": ["not", "a", "string"]})
    body = response.json()
    self.assertEqual((response.status_code, body["error"]), (422, "invalid request"))
    self.assertEqual((body["detail"][0]["loc"], body["detail"][0]["input"]), (["body", "model_id"], ["not", "a", "string"]))

  def test_a_binary_body_on_a_json_route_is_a_422_not_a_500(self) -> None:
    response = self.post_bytes("optim_step", b"\x8a\xff", wire.PROTO)
    self.assertEqual((response.status_code, response.json()["detail"][0]["input"]), (422, "<2 bytes>"))

  def test_http_errors_keep_their_headers_and_body_rules(self) -> None:
    throttled = asyncio.run(api_server.http_error(None, HTTPException(status_code=429, detail="slow down", headers={"Retry-After": "3"})))
    self.assertEqual((throttled.status_code, throttled.headers["retry-after"], json.loads(throttled.body)), (429, "3", {"error": "slow down"}))
    unchanged = asyncio.run(api_server.http_error(None, HTTPException(status_code=304)))
    self.assertEqual((unchanged.status_code, unchanged.body), (304, b""))


class SaveSeqIdZeroTest(ApiServerTest):
  def test_the_first_saves_zero_seq_id_is_kept(self) -> None:
    self.post("save_weights_for_sampler", {"model_id": "job-a", "sampling_session_seq_id": 0})
    self.post("save_weights", {"model_id": "job-a", "seq_id": 0})
    queued = self.queued()
    self.assertEqual(queued[0]["payload"]["sampling_session_id"], "tinker://job-a/sampler_weights/sampler-0")
    self.assertTrue(queued[1]["payload"]["state_path"].endswith("job-a-samp-0"))


class ApiServerPathTest(ApiServerTest):
  def test_checkpoint_state_paths_are_model_scoped_and_tinker_uri_roundtrip(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      self._make_app(tmp_dir=tmp_dir)
      dir_a = os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "step-5")
      self.assertEqual(paths.checkpoint_state_path("job-a", "step-5", base_dir=tmp_dir), dir_a)
      self.assertEqual(paths.tinker_state_path(dir_a, base_dir=tmp_dir), "tinker://job-a/weights/step-5")
      self.assertEqual(paths.checkpoint_state_path("job-b", "tinker://job-a/weights/step-5", base_dir=tmp_dir), dir_a)
      self.assertEqual(paths.tinker_state_path("/elsewhere/final", base_dir=tmp_dir), "/elsewhere/final")
      self.assertEqual(paths.checkpoint_state_path("job-a", "/mnt/checkpoints/final", base_dir=tmp_dir), "/mnt/checkpoints/final")
      self.assertIsNone(paths.tinker_checkpoint_dir("tinker://job-a/sampler_weights/sampler-3", base_dir=tmp_dir))

      refused = self.post("load_weights", {"model_id": "job-b", "path": "tinker://job-a/sampler_weights/sampler-3"})
      self.assertEqual(refused.status_code, 400)
      self.assertIn("is not a tinker://<model>/weights/<name> path", refused.json()["error"])

      self.post("save_weights", {"model_id": "job-a", "path": "step-5"})
      queued = self.queued()[0]
      self.assertEqual((queued["op"], queued["payload"]["include_optimizer"]), ("save_state", True))
      self.assertEqual(paths.tinker_state_path(queued["payload"]["state_path"], base_dir=tmp_dir), "tinker://job-a/weights/step-5")

  def test_weights_info_reads_the_checkpoint_on_disk(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      self._make_app(tmp_dir=tmp_dir)
      state_dir = os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "step-5")
      os.makedirs(os.path.join(state_dir, "job-a"))
      with open(os.path.join(state_dir, "metadata.json"), "w") as f:
        json.dump({"base_model": "google/gemma-4-e2b", "model_id": "job-a", "has_optimizer": True}, f)
      with open(os.path.join(state_dir, "job-a", "adapter_config.json"), "w") as f:
        json.dump({"r": 8}, f)
      info = self.post("weights_info", {"tinker_path": "tinker://job-a/weights/step-5"}).json()
      missing = self.post("weights_info", {"tinker_path": "tinker://job-a/weights/never"})
    self.assertEqual((info["base_model"], info["is_lora"], info["lora_rank"]), ("google/gemma-4-e2b", True, 8))
    self.assertEqual((missing.status_code, missing.json()), (404, {"error": "No checkpoint at tinker://job-a/weights/never"}))


class ProtobufWireTest(ApiServerTest):
  FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "fwdbwd_request_tinker_0.29.0.pb")

  def test_protobuf_and_json_forward_backward_queue_equivalent_proto_bytes(self) -> None:
    with open(self.FIXTURE, "rb") as fh:
      body = fh.read()
    proto = self.client.post("/api/v1/forward_backward", content=body, headers={"Content-Type": wire.PROTO})
    self.assertEqual(proto.status_code, 200, proto.text)
    from_proto = self.queued()[0]
    self.assertEqual(from_proto["payload"]["proto_bytes"], body)

    with open(self.FIXTURE[:-3] + ".json") as fh:
      json_body = json.load(fh)
    as_json = self.client.post("/api/v1/forward_backward", json=json_body)
    self.assertEqual(as_json.status_code, 200, as_json.text)
    from_json = self.queued()[0]

    self.assertEqual((from_proto["op"], from_proto["model_id"]), ("forward_backward", "model-abc"))
    self.assertEqual(from_proto["payload"]["loss_fn"], "importance_sampling")
    self.assertEqual(from_proto["payload"]["loss_config"], {"clip_range": 0.2, "kl_coeff": 0.01, "mode": "token"})
    d_proto, d_json = wire.datums(from_proto["payload"]["proto_bytes"]), wire.datums(from_json["payload"]["proto_bytes"])
    self.assertEqual(len(d_proto), len(d_json))
    for a, b in zip(d_proto, d_json):
      np.testing.assert_array_equal(a.tokens, b.tokens)
      for k in a.inputs:
        np.testing.assert_array_equal(a.inputs[k], b.inputs[k])

  def test_forward_only_reaches_the_worker_from_both_routes(self) -> None:
    msg = pb.ForwardBackwardRequest(model_id="model-abc", seq_id=1, loss_fn="cross_entropy", forward_only=True)
    resp = self.client.post("/api/v1/forward_backward", content=msg.SerializeToString(), headers={"Content-Type": wire.PROTO})
    self.assertEqual(resp.status_code, 200, resp.text)
    self.assertTrue(self.queued()[0]["payload"]["forward_only"])

    legacy = self.client.post("/api/v1/forward", json={"model_id": "model-abc", "forward_input": {"data": [], "loss_fn": "cross_entropy"}})
    self.assertEqual(legacy.status_code, 200, legacy.text)
    self.assertTrue(self.queued()[0]["payload"]["forward_only"])

    fb_body = {"model_id": "model-abc", "forward_backward_input": {"data": [], "loss_fn": "cross_entropy"}}
    train = self.client.post("/api/v1/forward_backward", json=fb_body)
    self.assertEqual(train.status_code, 200, train.text)
    self.assertFalse(self.queued()[0]["payload"]["forward_only"])

  def test_bad_bodies_are_client_errors_not_500s(self) -> None:
    self.assertEqual(self.client.post("/api/v1/forward_backward", content=b"\xff\xfe bad", headers={"Content-Type": wire.PROTO}).status_code, 400)
    self.assertEqual(
      self.client.post("/api/v1/forward_backward", content=b"x", headers={"Content-Type": wire.PROTO, "Content-Encoding": "zstd"}).status_code, 415
    )
    self.assertEqual(self.client.post("/api/v1/forward_backward", content=b"x", headers={"Content-Type": "text/plain"}).status_code, 415)
    bad = self.client.post("/api/v1/optim_step", content=b"\x8a\xff", headers={"Content-Type": wire.PROTO})
    self.assertEqual(bad.status_code, 422)
    self.assertNotIn("\x8a", bad.text)

  def test_retrieve_future_passes_proto_bytes_and_decodes_json_at_edge(self) -> None:
    samp_raw = wire.sample_output([wire.SampleOut(stop=True, tokens=[1, 2], logprobs=[-0.5, -1.0])])
    asyncio.run(self.store.set_result("samp-1", wire.proto_result("sample", samp_raw)))
    asyncio.run(self.store.set_result("optim-1", {"type": "optim_step", "metrics": {"grad_norm:mean": 0.0}}))

    as_json = self.client.post("/api/v1/retrieve_future", json={"request_id": "samp-1"})
    self.assertTrue(as_json.headers["content-type"].startswith("application/json"))
    self.assertEqual(as_json.json()["type"], "sample")
    self.assertEqual(as_json.json()["sequences"][0]["tokens"], [1, 2])

    as_proto = self.client.post("/api/v1/retrieve_future", json={"request_id": "samp-1"}, headers={"Accept": wire.PROTO})
    self.assertEqual((as_proto.status_code, as_proto.headers["content-type"], as_proto.content), (200, wire.PROTO, samp_raw))

    optim = self.client.post("/api/v1/retrieve_future", json={"request_id": "optim-1"}, headers={"Accept": wire.PROTO})
    self.assertTrue(optim.headers["content-type"].startswith("application/json"))
    self.assertEqual(optim.json()["type"], "optim_step")


class SampleSequenceIdsTest(ApiServerTest):
  def setUp(self) -> None:
    self._make_app(sampler_backend="torch")

  def test_asample_promise_carries_one_id_per_sample(self) -> None:
    promise = self.post("asample", {"model_id": "job-a", "prompt": {"chunks": [{"tokens": [1, 2]}]}, "num_samples": 3}).json()
    self.assertEqual((len(promise["sample_sequence_ids"]), len(set(promise["sample_sequence_ids"]))), (3, 3))
    self.assertTrue(all(sid.startswith(promise["request_id"]) for sid in promise["sample_sequence_ids"]))

  def test_asample_defaults_to_a_single_sample(self) -> None:
    promise = self.post("asample", {"model_id": "job-a", "prompt": {"chunks": [{"tokens": [1]}]}}).json()
    self.assertEqual(len(promise["sample_sequence_ids"]), 1)


class InputBoundaryTest(ApiServerTest):
  def test_invalid_training_datum_is_a_validation_error(self) -> None:
    resp = self.post("forward_backward", {"model_id": "m", "forward_backward_input": {"data": [{"model_input": {"chunks": [{"tokens": ["bad"]}]}}]}})
    self.assertEqual((resp.status_code, self.store.queues), (422, {}))

  def test_null_configs_and_sampling_defaults_preserve_zero(self) -> None:
    self._make_app(sampler_backend="torch")
    self.assertEqual(self.post("create_model", {"base_model": "base", "lora_config": None, "full_config": None}).status_code, 200)
    self.assertEqual(self.queued()[0]["payload"]["lora_config"]["rank"], 16)
    for value in (None, 0):
      with self.subTest(value=value):
        body = {"model_id": "base", "prompt": {"chunks": [{"tokens": [1]}]}, "sampling_params": {"temperature": value, "max_tokens": value}}
        self.assertEqual(self.post("asample", body).status_code, 200)
        q = self.queued()[0]["payload"]
        self.assertEqual((q["temperature"], q["max_tokens"]), (1.0 if value is None else 0, 20 if value is None else 0))


class RestoreRoutingTest(ApiServerTest):
  def test_restore_uses_checkpoint_identity(self):
    for kind in ("lora", "full"):
      with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}):
        with open(os.path.join(directory, "metadata.json"), "w") as f:
          json.dump({"base_model": "checkpoint-base"}, f)
        if kind == "lora":
          with open(os.path.join(directory, "adapter_config.json"), "w") as f:
            json.dump({"r": 8}, f)
        resp = self.post("create_model_from_state", {"state_path": directory})
        self.assertEqual(resp.status_code, 200)
        mid = resp.json()["request_id"]
        meta = json.loads(self.state.get_value_sync(f"open_rl:model_meta:{mid}"))
        queued_ft = self.queued()[0]["payload"]["fine_tuning_type"]
        self.assertEqual((meta["base_model"], meta["fine_tuning_type"], queued_ft), ("checkpoint-base", kind, kind))

  def test_restore_refuses_a_fine_tuning_type_the_checkpoint_does_not_have(self):
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}):
      with open(os.path.join(directory, "metadata.json"), "w") as f:
        json.dump({"base_model": "checkpoint-base"}, f)
      with open(os.path.join(directory, "adapter_config.json"), "w") as f:
        json.dump({"r": 8}, f)
      resp = self.post("create_model_from_state", {"state_path": directory, "user_metadata": {"openrl.fine_tuning_type": "full"}})
    self.assertEqual(resp.status_code, 400)
    self.assertIn("does not match the lora checkpoint", resp.json()["error"])


class FineTuningSettingsTest(ApiServerTest):
  def setUp(self) -> None:
    super().setUp()
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}))

  def metadata(self, model_id: str) -> dict:
    return json.loads(self.state.get_value_sync(f"open_rl:model_meta:{model_id}"))

  def test_tags_choose_full_fine_tuning_and_its_weight_sync(self) -> None:
    tags = ["openrl.fine_tuning_type=full", "openrl.weight_sync=full", "openrl.fft_cpu_offload=false"]
    sid = self.post("create_session", {"tags": tags}).json()["session_id"]
    meta = self.metadata(self.post("create_model", {"base_model": "m", "session_id": sid}).json()["request_id"])
    self.assertEqual((meta["fine_tuning_type"], meta["weight_sync_config"]["strategy"]), ("full", "full"))
    self.assertEqual(meta["full_config"], {"seed": None, "cpu_offload": False, "weight_sync_strategy": "full"})

  def test_the_client_log_id_tag_lands_on_the_model(self) -> None:
    sid = self.post("create_session", {"tags": ["openrl.client_log_id=gsm8k-7"]}).json()["session_id"]
    mid = self.post("create_model", {"base_model": "m", "session_id": sid}).json()["request_id"]
    self.assertEqual(self.metadata(mid)["client_log_id"], "gsm8k-7")

  def test_lora_is_the_default_and_refuses_fft_options_or_removed_fields(self) -> None:
    self.assertEqual(self.metadata(self.post("create_model", {"base_model": "m"}).json()["request_id"])["fine_tuning_type"], "lora")
    r1 = self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.weight_sync": "full"}})
    self.assertEqual(r1.status_code, 400)
    self.assertIn("openrl.weight_sync applies to fine_tuning_type=full only", r1.json()["error"])
    r2 = self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.fine_tuning_type": "qlora"}})
    self.assertEqual(r2.status_code, 400)
    r3 = self.post("create_model", {"base_model": "m"}, headers={"x-open-rl-fine-tuning-type": "full"})
    self.assertEqual(r3.status_code, 400)
    self.assertIn("set openrl.fine_tuning_type", r3.json()["error"])
    r4 = self.post("create_model", {"base_model": "m", "full_config": {"seed": 1}})
    self.assertEqual(r4.status_code, 400)
    self.assertIn("openrl.fft_seed", r4.json()["error"])


class ExclusiveAndTrainerBackendTest(ApiServerTest):
  def setUp(self) -> None:
    self._make_app(worker_manager=RuntimeManager())
    self.enterContext(patch("server.worker_manager.get_state_store", return_value=self.state))

  def metadata(self, model_id: str) -> dict:
    return json.loads(self.state.get_value_sync(f"open_rl:model_meta:{model_id}"))

  def active_set(self, model_id: str) -> str | None:
    return asyncio.run(self.gw.resolve_active_set_id(model_id))

  def test_exclusive_session_inheritance_overrides_and_tags(self) -> None:
    m1 = self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.exclusive": "true"}}).json()["request_id"]
    self.assertEqual((self.metadata(m1)["exclusive"], self.metadata(m1)["full_config"]["cpu_offload"]), (True, False))
    m_bool = self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.exclusive": True}}).json()["request_id"]
    self.assertTrue(self.metadata(m_bool)["exclusive"])
    m_def = self.post("create_model", {"base_model": "m"}).json()["request_id"]
    self.assertEqual((self.metadata(m_def)["exclusive"], self.metadata(m_def)["full_config"]["cpu_offload"]), (False, True))

    sid = self.post("create_session", {"user_metadata": {"openrl.exclusive": "true"}}).json()["session_id"]
    self.post("session_heartbeat", {"session_id": sid})
    self.assertTrue(self.metadata(self.post("create_model", {"base_model": "m", "session_id": sid}).json()["request_id"])["exclusive"])
    own = self.post("create_model", {"base_model": "m", "session_id": sid, "user_metadata": {"openrl.exclusive": "false"}}).json()["request_id"]
    self.assertFalse(self.metadata(own)["exclusive"])

    tags = ["openrl.exclusive=true", "my-label"]
    t_sid = self.post("create_session", {"tags": tags}).json()["session_id"]
    self.assertTrue(self.metadata(self.post("create_model", {"base_model": "m", "session_id": t_sid}).json()["request_id"])["exclusive"])
    o_sid = self.post("create_session", {"tags": tags, "user_metadata": {"openrl.exclusive": "false"}}).json()["session_id"]
    self.assertFalse(self.metadata(self.post("create_model", {"base_model": "m", "session_id": o_sid}).json()["request_id"])["exclusive"])

    for tag, err in (("openrl.exclusiv=true", "openrl.exclusiv: unknown setting"), ("openrl.exclusive", "needs a value")):
      resp = self.post("create_session", {"tags": [tag]})
      self.assertEqual(resp.status_code, 400)
      self.assertIn(err, resp.json()["error"])
    self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": {"wandb_link": "x", "exclusive": "yes"}}).status_code, 200)
    self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.exclusive": "yes"}}).status_code, 400)
    self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.exclusiv": "true"}}).status_code, 400)

  def test_trainer_backends_multi_gpu_cp_and_sampler_replicas(self) -> None:
    um = {"openrl.trainer_backend": "automodel"}
    auto1 = self.post("create_model", {"base_model": "m", "user_metadata": um, "lora_config": {"rank": 16}}).json()["request_id"]
    auto2 = self.post("create_model", {"base_model": "m", "user_metadata": um, "lora_config": {"rank": 32}}).json()["request_id"]
    self.assertEqual((self.active_set(auto1), self.active_set(auto2)), ("automodel-m-1", "automodel-m-1"))
    self.assertEqual(self.active_set(self.post("create_model", {"base_model": "m"}).json()["request_id"]), "m-1")

    img1 = self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.trainer_backend": "ghcr.io/org/t:1"}}).json()["request_id"]
    img2 = self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.trainer_backend": "ghcr.io/org/t:2"}}).json()["request_id"]
    self.assertRegex(self.active_set(img1), r"^image-[0-9a-f]{10}-m-1$")
    self.assertNotEqual(self.active_set(img1), self.active_set(img2))

    # Multi-GPU, trainer_gpus alias, cp validation
    mg_meta = {"openrl.trainer_backend": "automodel", "openrl.trainer_gpus": "4", "openrl.trainer_cp": "4"}
    mg = self.post("create_model", {"base_model": "m", "user_metadata": mg_meta}).json()["request_id"]
    self.assertEqual((self.metadata(mg)["trainer_device_count"], self.metadata(mg)["trainer_cp"]), (4, 4))
    self.assertNotEqual(self.active_set(mg), "automodel-m-1")
    self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.trainer_device_count": "4"}}).status_code, 400)
    bad_cp = {"openrl.trainer_backend": "automodel", "openrl.trainer_device_count": "4", "openrl.trainer_cp": "3"}
    bad_alias = {"openrl.trainer_backend": "automodel", "openrl.trainer_device_count": "2", "openrl.trainer_gpus": "4"}
    self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": bad_cp}).status_code, 400)
    self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": bad_alias}).status_code, 400)
    self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.trainer_backend": "nope"}}).status_code, 400)

    # FFT + sampler_replicas / automodel checks
    with patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}):
      bad_auto = {"openrl.trainer_backend": "automodel", "openrl.fine_tuning_type": "full"}
      bad_rep = {"openrl.fine_tuning_type": "full", "openrl.sampler_replicas": "4"}
      good_rep = {"openrl.fine_tuning_type": "full", "openrl.sampler_replicas": "4", "openrl.exclusive": "true"}
      self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": bad_auto}).status_code, 400)
      self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": bad_rep}).status_code, 400)
      ok_fft = self.post("create_model", {"base_model": "m", "user_metadata": good_rep})
      self.assertEqual(self.metadata(ok_fft.json()["request_id"])["sampler_replicas"], 4)

    # Without worker_manager or with LocalWorkerManager
    self._make_app(worker_manager=None)
    self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.exclusive": "true"}}).status_code, 400)
    self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.trainer_backend": "automodel"}}).status_code, 400)
    self._make_app(worker_manager=object.__new__(LocalWorkerManager))
    self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.trainer_backend": "ghcr.io/org/t:1"}}).status_code, 400)
    self.assertEqual(self.post("create_model", {"base_model": "m", "user_metadata": {"openrl.sampler_replicas": "4"}}).status_code, 400)


if __name__ == "__main__":
  unittest.main()
