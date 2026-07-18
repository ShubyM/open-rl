import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import httpx

from server import control_cli


class ClientStub:
  responses: dict[str, object] = {}
  post_responses: dict[str, object] = {}
  calls: list[tuple[str, str, object]] = []

  def __init__(self, base_url: str, timeout: float = 10.0):
    self.base_url = base_url

  def close(self) -> None:
    pass

  def get(self, path: str, params=None):
    self.calls.append(("GET", path, params))
    response = self.responses.get(path)
    if callable(response):
      response = response(path, params)
    if isinstance(response, Exception):
      raise response
    if response is None:
      raise AssertionError(f"unexpected path {path}")
    return response

  def post(self, path: str, payload=None):
    self.calls.append(("POST", path, payload))
    response = self.post_responses.get(path)
    if callable(response):
      response = response(path, payload)
    if isinstance(response, Exception):
      raise response
    if response is None:
      raise AssertionError(f"unexpected path {path}")
    return response


class ControlCliTest(unittest.TestCase):
  def setUp(self) -> None:
    ClientStub.responses = {}
    ClientStub.post_responses = {}
    ClientStub.calls = []

  def run_cli(self, *args: str):
    stdout, stderr = io.StringIO(), io.StringIO()
    with patch("server.control_cli.ControlClient", ClientStub), redirect_stdout(stdout), redirect_stderr(stderr):
      code = control_cli.main(list(args))
    return code, stdout.getvalue(), stderr.getvalue()

  def test_duration_suffixes(self) -> None:
    self.assertEqual(control_cli.parse_duration("500ms"), 0.5)
    self.assertEqual(control_cli.parse_duration("2m"), 120)
    self.assertEqual(control_cli.parse_duration("1.5h"), 5400)

  def test_control_client_post_uses_control_api_and_json_payload(self) -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
      requests.append(request)
      return httpx.Response(202, json={"status": "accepted"})

    client = control_cli.ControlClient("http://gateway.example/")
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(respond))
    self.addCleanup(client.close)

    result = client.post("/runs/run-a/stop", {"reason": "operator"})

    self.assertEqual(result, {"status": "accepted"})
    self.assertEqual(requests[0].method, "POST")
    self.assertEqual(str(requests[0].url), "http://gateway.example/api/v1/control/runs/run-a/stop")
    self.assertEqual(json.loads(requests[0].content), {"reason": "operator"})

  def test_runs_json_is_machine_readable(self) -> None:
    ClientStub.responses = {
      "/runs": {"runs": [{"id": "run-a", "status": "running", "phase": "sampling"}]},
    }
    code, output, error = self.run_cli("runs", "--json")
    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertEqual(json.loads(output)["runs"][0]["id"], "run-a")
    self.assertEqual(error, "")

  def test_deploy_is_agent_friendly_and_does_not_call_control_api(self) -> None:
    result = {
      "status": "deployed",
      "revision": "abc123",
      "source_bytes": 42,
      "source_path": "/mnt/shared/open-rl/source/abc123",
      "namespace": "training",
      "deployment": "open-rl-gateway",
      "workers_reset": True,
    }
    with patch("server.control_cli.deploy_source", return_value=result) as deploy:
      code, output, error = self.run_cli("deploy", "--namespace", "training", "--reset-workers", "--json")

    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertEqual(json.loads(output), result)
    self.assertEqual(error, "")
    self.assertEqual(ClientStub.calls, [])
    config = deploy.call_args.args[0]
    self.assertEqual(config.source, "src")
    self.assertEqual(config.namespace, "training")
    self.assertTrue(config.reset_workers)

  def test_failed_status_has_nonzero_exit(self) -> None:
    ClientStub.responses = {
      "/runs/run-a": {"id": "run-a", "status": "failed", "phase": "image_pull", "message": "ImagePullBackOff"},
    }
    code, output, error = self.run_cli("status", "run-a", "--json")
    self.assertEqual(code, control_cli.EXIT_UNHEALTHY)
    self.assertEqual(json.loads(output)["message"], "ImagePullBackOff")
    self.assertEqual(error, "")

  def test_not_found_has_stable_exit(self) -> None:
    ClientStub.responses = {"/runs/missing": LookupError("missing")}
    code, output, error = self.run_cli("status", "missing", "--json")
    self.assertEqual(code, control_cli.EXIT_NOT_FOUND)
    self.assertEqual(output, "")
    self.assertIn("not found", error)

  def test_log_payload_shapes(self) -> None:
    self.assertEqual(control_cli.log_lines({"lines": ["one", "two"]}), ["one", "two"])
    self.assertEqual(control_cli.log_lines({"logs": "one\ntwo\n"}), ["one", "two"])

  def test_logs_jsonl_preserves_native_structured_fields(self) -> None:
    ClientStub.responses = {
      "/runs/run-a/logs": {
        "lines": [{"timestamp": "2026-07-11T00:00:00Z", "stream": "stderr", "message": "worker failed"}],
        "error": None,
      }
    }

    code, output, error = self.run_cli("logs", "run-a", "--component", "sampler", "--json")

    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertEqual(
      json.loads(output),
      {
        "run_id": "run-a",
        "component": "sampler",
        "timestamp": "2026-07-11T00:00:00Z",
        "stream": "stderr",
        "message": "worker failed",
      },
    )
    self.assertNotIn("line", json.loads(output))
    self.assertEqual(error, "")

  def test_logs_backend_error_is_nonzero(self) -> None:
    ClientStub.responses = {"/runs/run-a/logs": {"lines": [], "error": "pod logs unavailable"}}

    code, output, error = self.run_cli("logs", "run-a", "--json")

    self.assertEqual(code, control_cli.EXIT_API)
    self.assertEqual(output, "")
    self.assertIn("pod logs unavailable", error)

  def test_cluster_reads_standard_kubernetes_resources(self) -> None:
    ClientStub.responses = {
      "/cluster": {
        "status": "healthy",
        "nodes": [
          {
            "name": "kind-control-plane",
            "status": "ready",
            "allocatable": {"cpu": "8", "memory": "16Gi", "nvidia.com/gpu": "1"},
            "pod_count": 6,
          }
        ],
      }
    }
    code, output, error = self.run_cli("cluster")
    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertIn("kind-control-plane", output)
    self.assertIn("16Gi", output)
    self.assertEqual(error, "")

  def test_degraded_cluster_has_nonzero_exit_for_automation(self) -> None:
    ClientStub.responses = {"/cluster": {"status": "degraded", "nodes": [], "errors": ["time-slicer unavailable"]}}

    code, output, error = self.run_cli("cluster", "--json")

    self.assertEqual(code, control_cli.EXIT_UNHEALTHY)
    self.assertEqual(json.loads(output)["status"], "degraded")
    self.assertEqual(error, "")

  def test_problems_preserves_json_and_signals_unhealthy(self) -> None:
    ClientStub.responses = {
      "/problems": {
        "generated_at": "2026-07-11T00:00:00Z",
        "problems": [
          {
            "id": "pod/pod-a/unschedulable",
            "severity": "warning",
            "code": "pod_unschedulable",
            "summary": "Pod cannot be scheduled",
            "evidence": ["0/1 nodes available"],
            "resources": {"run_id": "run-a", "component": "trainer", "pod_name": "pod-a", "node": None},
            "remediation": "Inspect node capacity",
            "actions": [{"name": "inspect", "allowed": True}],
          }
        ],
      }
    }

    code, output, error = self.run_cli("problems", "--json")

    self.assertEqual(code, control_cli.EXIT_UNHEALTHY)
    self.assertEqual(json.loads(output)["problems"][0]["code"], "pod_unschedulable")
    self.assertEqual(error, "")

    code, output, error = self.run_cli("problems")
    self.assertEqual(code, control_cli.EXIT_UNHEALTHY)
    self.assertIn("run-a", output)
    self.assertIn("trainer", output)
    self.assertIn("pod-a", output)
    self.assertEqual(error, "")

  def test_empty_or_info_only_problems_are_successful(self) -> None:
    for problems in ([], [{"id": "note", "severity": "info", "code": "notice", "summary": "FYI"}]):
      with self.subTest(problems=problems):
        ClientStub.responses = {"/problems": {"problems": problems}}
        code, output, error = self.run_cli("problems", "--json")
        self.assertEqual(code, control_cli.EXIT_OK)
        self.assertEqual(json.loads(output)["problems"], problems)
        self.assertEqual(error, "")

  def test_inspect_resolves_exact_run_before_cluster_resources(self) -> None:
    ClientStub.responses = {"/runs/shared": {"id": "shared", "status": "running", "can_stop": True}}

    code, output, error = self.run_cli("inspect", "shared", "--json")

    result = json.loads(output)
    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertEqual(result["kind"], "run")
    self.assertEqual(result["resource"]["id"], "shared")
    self.assertEqual([call[1] for call in ClientStub.calls], ["/runs/shared"])
    self.assertEqual(error, "")

  def test_inspect_node_includes_its_related_pods(self) -> None:
    ClientStub.responses = {
      "/runs/node-a": LookupError("node-a"),
      "/cluster": {
        "nodes": [{"name": "node-a", "status": "ready", "ready": True}],
        "pods": [
          {"pod_name": "pod-a", "node": "node-a"},
          {"pod_name": "pod-b", "node": "node-b"},
        ],
      },
    }

    code, output, error = self.run_cli("inspect", "node-a", "--json")

    result = json.loads(output)
    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertEqual(result["kind"], "node")
    self.assertEqual([pod["pod_name"] for pod in result["related_pods"]], ["pod-a"])
    self.assertEqual(error, "")

  def test_inspect_exact_pod_and_missing_target(self) -> None:
    cluster = {
      "nodes": [{"name": "node-a", "status": "ready", "ready": True}],
      "pods": [{"pod_name": "pod-a", "node": "node-a", "status": "failed", "model_id": "run-a"}],
    }
    ClientStub.responses = {"/runs/pod-a": LookupError("pod-a"), "/cluster": cluster}

    code, output, error = self.run_cli("inspect", "pod-a", "--json")

    self.assertEqual(code, control_cli.EXIT_UNHEALTHY)
    self.assertEqual(json.loads(output)["kind"], "pod")
    self.assertEqual(error, "")

    ClientStub.responses = {"/runs/missing": LookupError("missing"), "/cluster": cluster}
    code, output, error = self.run_cli("inspect", "missing", "--json")
    self.assertEqual(code, control_cli.EXIT_NOT_FOUND)
    self.assertEqual(output, "")
    self.assertIn("not found: missing", error)

  def test_events_is_one_shot_and_passes_cursor(self) -> None:
    ClientStub.responses = {
      "/runs/run-a/events": {
        "events": [{"cursor": "2-0", "component": "sampler", "phase": "ready"}],
        "next_cursor": "2-0",
      }
    }

    code, output, error = self.run_cli("events", "run-a", "--after", "1-0", "--limit", "7", "--json")

    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertEqual(json.loads(output)["next_cursor"], "2-0")
    self.assertEqual(ClientStub.calls, [("GET", "/runs/run-a/events", {"after": "1-0", "limit": 7})])
    self.assertEqual(error, "")

  def test_stop_accepts_immediate_and_idempotent_responses(self) -> None:
    for operation in ("accepted", "noop"):
      with self.subTest(operation=operation):
        ClientStub.post_responses = {"/runs/run-a/stop": {"status": operation, "run": {"id": "run-a", "status": "stopped", "can_stop": False}}}
        ClientStub.calls = []

        code, output, error = self.run_cli("stop", "run-a", "--wait", "--json")

        self.assertEqual(code, control_cli.EXIT_OK)
        self.assertEqual(json.loads(output)["status"], operation)
        self.assertEqual(ClientStub.calls, [("POST", "/runs/run-a/stop", None)])
        self.assertEqual(error, "")

  def test_stop_wait_polls_until_can_stop_is_false(self) -> None:
    states = iter(
      [
        {"id": "run-a", "status": "running", "can_stop": True},
        {"id": "run-a", "status": "running", "can_stop": False},
      ]
    )
    ClientStub.post_responses = {"/runs/run-a/stop": {"status": "accepted", "run": {"id": "run-a", "status": "running", "can_stop": True}}}
    ClientStub.responses = {"/runs/run-a": lambda path, params: next(states)}

    code, output, error = self.run_cli("stop", "run-a", "--wait", "--timeout", "1s", "--interval", "0s", "--json")

    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertFalse(json.loads(output)["run"]["can_stop"])
    self.assertEqual([call[0] for call in ClientStub.calls], ["POST", "GET", "GET"])
    self.assertEqual(error, "")

  def test_stop_wait_timeout_has_stable_exit(self) -> None:
    ClientStub.post_responses = {"/runs/run-a/stop": {"status": "accepted", "run": {"id": "run-a", "status": "running", "can_stop": True}}}

    code, output, error = self.run_cli("stop", "run-a", "--wait", "--timeout", "0s", "--json")

    self.assertEqual(code, control_cli.EXIT_TIMEOUT)
    self.assertEqual(output, "")
    self.assertIn("timed out", error)

  def test_help_documents_stable_exit_codes(self) -> None:
    help_text = control_cli.format_help()
    self.assertIn("Exit codes: 0 success, 2 invalid arguments, 3 not found", help_text)
    self.assertIn("problem at warning severity or higher", help_text)

  def test_chz_parser_keeps_public_flags_kebab_cased(self) -> None:
    code, output, error = self.run_cli("logs", "run-a", "--request-timeout", "nope")

    self.assertEqual(code, control_cli.EXIT_USAGE)
    self.assertEqual(output, "")
    self.assertIn("request-timeout", error)
    self.assertNotIn("request_timeout", error)

  def test_chz_key_value_syntax_accepts_snake_case(self) -> None:
    ClientStub.responses = {"/runs/run-a/logs": {"lines": ["ready"], "error": None}}

    code, output, error = self.run_cli("logs", "run_id=run-a", "component=sampler", "json=true")

    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertEqual(json.loads(output)["component"], "sampler")
    self.assertEqual(error, "")

  def test_command_help_has_no_internal_underscore_names(self) -> None:
    code, output, error = self.run_cli("logs", "--help")

    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertIn("<run-id>", output)
    self.assertIn("--request-timeout", output)
    self.assertNotIn("run_id", output)
    self.assertEqual(error, "")


if __name__ == "__main__":
  unittest.main()
