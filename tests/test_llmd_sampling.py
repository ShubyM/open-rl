import asyncio
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import api_server, llmd
from server.store import InMemoryStateStore, InMemoryStore

# A real vLLM 0.30.0 /v1/completions response for prompt [9707, 11, 1246, 525, 498, 30],
# n=2, logprobs=1, prompt_logprobs=1, return_token_ids.
COMPLETION = json.loads((Path(__file__).parent / "fixtures" / "vllm_completion_0.30.0.json").read_text())
PROMPT = [9707, 11, 1246, 525, 498, 30]
BASE_MODEL = "Qwen/Qwen3.5-9B"


class Router:
  """An HTTP server standing in for the llm-d router. It replies with the
  recorded vLLM response, after failing the first `failures` requests."""

  def __init__(self, failures: int = 0) -> None:
    self.bodies: list[dict] = []
    self.failures = failures
    router = self

    class Handler(BaseHTTPRequestHandler):
      def do_POST(self) -> None:
        router.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        if router.failures:
          router.failures -= 1
          self.send_response(503)
          self.end_headers()
          return
        payload = json.dumps(COMPLETION).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

      def log_message(self, *args) -> None:
        pass

    self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self.url = f"http://127.0.0.1:{self.server.server_port}"
    threading.Thread(target=self.server.serve_forever, daemon=True).start()

  def close(self) -> None:
    self.server.shutdown()


class LlmdSamplingTest(unittest.TestCase):
  def setUp(self) -> None:
    self.tmp = self.enterContext(tempfile.TemporaryDirectory())
    self.router = Router()
    self.addCleanup(self.router.close)
    self.enterContext(patch.object(api_server, "store", InMemoryStore()))
    self.enterContext(patch.object(api_server, "state", InMemoryStateStore()))
    self.enterContext(patch.object(api_server, "TMP_DIR", self.tmp))
    self.enterContext(patch.object(api_server, "get_sampler_backend", return_value="vllm"))
    self.enterContext(patch.object(api_server, "llmd_client", None))
    self.enterContext(patch.object(api_server, "llmd_versions", llmd.AdapterVersions()))
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_LLMD_ROUTERS": json.dumps({BASE_MODEL: self.router.url})}))
    meta = json.dumps({"base_model": BASE_MODEL, "fine_tuning_type": "lora"})
    asyncio.run(api_server.state.set_value("open_rl:model_meta:job-a", meta))
    self.client = self.enterContext(TestClient(api_server.app))

  def save_adapter(self, weights: str = "v1") -> str:
    """What the trainer does on every save: overwrite the job's one PEFT directory."""
    peft_dir = os.path.join(self.tmp, "peft", "job-a", "job-a")
    os.makedirs(peft_dir, exist_ok=True)
    Path(peft_dir, "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": BASE_MODEL}))
    Path(peft_dir, "adapter_model.safetensors").write_text(weights)
    return peft_dir

  def retrieve_save(self, ref: str) -> None:
    """The client retrieving a save_weights_for_sampler result, as the SDK does after each save."""
    request_id = f"save-{ref}"
    result = {"type": "sampler_weights_saved", "path": None, "sampling_session_id": ref}
    asyncio.run(api_server.store.set_future(request_id, result))
    response = self.client.post("/api/v1/retrieve_future", json={"request_id": request_id})
    self.assertEqual(response.json()["type"], "save_weights_for_sampler")

  def adapter_weights(self, ref: str) -> str | None:
    path = Path(self.tmp, "llmd-adapters", llmd.adapter_name(ref), "adapter_model.safetensors")
    return path.read_text() if path.exists() else None

  def sample(self, model_id: str, **extra) -> dict:
    body = {"model_id": model_id, "prompt": {"chunks": [{"tokens": PROMPT}]}, "num_samples": 2, "sampling_params": {"max_tokens": 4}, **extra}
    promise = self.client.post("/api/v1/asample", json=body).json()
    return self.client.post("/api/v1/retrieve_future", json={"request_id": promise["request_id"]}).json()

  def test_saved_version_samples_under_its_own_adapter_name(self) -> None:
    self.save_adapter()
    ref = "tinker://job-a/sampler_weights/000031"

    result = self.sample(ref)

    name = self.router.bodies[0]["model"]
    self.assertEqual(name, llmd.adapter_name(ref))
    self.assertTrue(name.startswith("job-a-"))
    self.assertEqual(self.adapter_weights(ref), "v1")
    self.assertEqual(self.router.bodies[0]["prompt"], PROMPT)
    self.assertNotEqual(llmd.adapter_name("tinker://job-a/sampler_weights/000032"), name)
    self.assertEqual([seq["tokens"] for seq in result["sequences"]], [[15, 15, 15, 491], [9477, 36234, 498, 53]])
    self.assertAlmostEqual(result["sequences"][0]["logprobs"][0], -4.67476224899292)
    self.assertEqual(asyncio.run(api_server.store.get_sampling_requests_for_model("job-a")), [])

  def test_a_later_save_does_not_change_an_earlier_version(self) -> None:
    first, second = "tinker://job-a/sampler_weights/sampler-1", "tinker://job-a/sampler_weights/sampler-2"
    self.save_adapter("v1")
    self.retrieve_save(first)
    self.save_adapter("v2")
    self.retrieve_save(second)

    self.sample(second)

    self.assertEqual(self.adapter_weights(second), "v2")
    # The first version was superseded with nothing sampling it, so it is gone.
    self.assertIsNone(self.adapter_weights(first))

  def test_deleting_the_job_removes_its_adapters(self) -> None:
    ref = "tinker://job-a/sampler_weights/sampler-1"
    self.save_adapter()
    self.retrieve_save(ref)
    self.client.post("/api/v1/delete_model", json={"model_id": "job-a"})
    self.assertEqual(os.listdir(os.path.join(self.tmp, "llmd-adapters")), [])

  def test_before_the_first_save_samples_the_base_model(self) -> None:
    self.sample("job-a")
    self.assertEqual(self.router.bodies[0]["model"], BASE_MODEL)

  def test_router_errors_are_retried(self) -> None:
    self.router.failures = 2
    with patch.object(llmd.asyncio, "sleep", return_value=None):
      result = self.sample("job-a")
    self.assertEqual(len(self.router.bodies), 3)
    self.assertEqual(len(result["sequences"]), 2)

  def test_sampling_session_does_not_wait_for_a_sampler(self) -> None:
    with patch.object(api_server, "worker_manager", None):
      response = self.client.post("/api/v1/create_sampling_session", json={"session_id": "s", "model_path": "tinker://job-a/sampler_weights/000031"})
    self.assertEqual(response.status_code, 200)


class AdapterVersionsTest(unittest.TestCase):
  def test_a_superseded_version_waits_for_its_requests(self) -> None:
    versions = llmd.AdapterVersions()
    self.assertEqual(versions.add("job", ["v1"]), [])
    versions.start("v1")
    self.assertEqual(versions.add("job", ["v2", "v2-path"]), [])
    self.assertEqual(versions.finish("job", "v1"), ["v1"])
    self.assertTrue(versions.known("v2-path"))

  def test_the_latest_version_stays_until_the_job_is_deleted(self) -> None:
    versions = llmd.AdapterVersions()
    versions.add("job", ["v1", "v1-path"])
    versions.start("v1")
    self.assertEqual(versions.finish("job", "v1"), [])
    self.assertEqual(versions.remove_job("job"), ["v1-path", "v1"])
    self.assertFalse(versions.known("v1"))

  def test_adding_a_known_version_again_changes_nothing(self) -> None:
    versions = llmd.AdapterVersions()
    versions.add("job", ["v1"])
    self.assertEqual(versions.add("job", ["v1"]), [])
    self.assertTrue(versions.known("v1"))


class SampleResponseTest(unittest.TestCase):
  def test_prompt_logprobs_line_up_with_the_prompt(self) -> None:
    result = llmd.sample_response(COMPLETION, PROMPT)
    self.assertIsNone(result["prompt_logprobs"][0])
    self.assertAlmostEqual(result["prompt_logprobs"][1], -6.832459449768066)
    self.assertEqual(len(result["prompt_logprobs"]), len(PROMPT))
    self.assertEqual([seq["stop_reason"] for seq in result["sequences"]], ["length", "length"])

  def test_stop_strings_and_token_ids_go_in_separate_fields(self) -> None:
    body = llmd.completion_body("m", {"prompt_token_ids": [1], "max_tokens": 4, "stop": ["</s>", 248046]})
    self.assertEqual(body["stop"], ["</s>"])
    self.assertEqual(body["stop_token_ids"], [248046])
    self.assertNotIn("prompt_logprobs", body)


if __name__ == "__main__":
  unittest.main()
