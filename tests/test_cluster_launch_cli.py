import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from server import control_cli
from server.cluster_job import ClusterJobError


class ControlClientStub:
  def __init__(self, base_url: str, timeout: float = 10.0):
    self.base_url = base_url

  def close(self) -> None:
    pass


def launch_result(status: str = "running") -> dict[str, object]:
  return {
    "job": "qwen-smoke",
    "namespace": "open-rl-local",
    "pod": "qwen-smoke-pod",
    "gateway_url": "http://open-rl-gateway-service.open-rl-local.svc:8000",
    "status": status,
    "follow_command": ["kubectl", "--namespace", "open-rl-local", "logs", "-f", "job/qwen-smoke"],
    "stop_command": ["kubectl", "--namespace", "open-rl-local", "delete", "job/qwen-smoke"],
  }


class ClusterLaunchCliTest(unittest.TestCase):
  def run_cli(self, *args: str):
    stdout, stderr = io.StringIO(), io.StringIO()
    with patch("server.control_cli.ControlClient", ControlClientStub), redirect_stdout(stdout), redirect_stderr(stderr):
      code = control_cli.main(list(args))
    return code, stdout.getvalue(), stderr.getvalue()

  def test_launch_builds_job_config_from_public_flags_and_emits_json(self) -> None:
    with patch("server.control_cli.launch_job", return_value=launch_result()) as launch:
      code, output, error = self.run_cli(
        "launch",
        "examples/tiny/tiny_rl.py",
        "--args",
        "steps=2 base_model=Qwen/Qwen2.5-0.5B",
        "--image",
        "open-rl-client:dev",
        "--image-pull-policy",
        "Never",
        "--context",
        "kind-open-rl",
        "--namespace",
        "open-rl-local",
        "--gateway-namespace",
        "open-rl-local",
        "--name",
        "qwen-smoke",
        "--detach",
        "--timeout",
        "2m",
        "--json",
      )

    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertEqual(json.loads(output), launch_result())
    self.assertEqual(error, "")
    config = launch.call_args.args[0]
    self.assertEqual(config.source, "examples/tiny/tiny_rl.py")
    self.assertEqual(config.args, "steps=2 base_model=Qwen/Qwen2.5-0.5B")
    self.assertEqual(config.image, "open-rl-client:dev")
    self.assertEqual(config.image_pull_policy, "Never")
    self.assertEqual(config.context, "kind-open-rl")
    self.assertEqual(config.namespace, "open-rl-local")
    self.assertEqual(config.gateway_namespace, "open-rl-local")
    self.assertEqual(config.name, "qwen-smoke")
    self.assertTrue(config.detach)
    self.assertEqual(config.timeout, 120.0)

  def test_launch_accepts_native_snake_case_key_value_arguments(self) -> None:
    with patch("server.control_cli.launch_job", return_value=launch_result()) as launch:
      code, output, error = self.run_cli(
        "launch",
        "source=recipe.py",
        "image=client:dev",
        "image_pull_policy=Never",
        "gateway_namespace=open-rl",
        "request_memory=256Mi",
        "detach=true",
        "json=true",
      )

    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertEqual(json.loads(output)["status"], "running")
    self.assertEqual(error, "")
    config = launch.call_args.args[0]
    self.assertEqual(config.image_pull_policy, "Never")
    self.assertEqual(config.gateway_namespace, "open-rl")
    self.assertEqual(config.request_memory, "256Mi")

  def test_failed_job_has_unhealthy_exit_and_human_stop_hint(self) -> None:
    with patch("server.control_cli.launch_job", return_value=launch_result("failed")):
      code, output, error = self.run_cli("launch", "recipe.py", "--image", "client:dev")

    self.assertEqual(code, control_cli.EXIT_UNHEALTHY)
    self.assertIn("job/qwen-smoke  failed", output)
    self.assertIn("kubectl --namespace open-rl-local delete job/qwen-smoke", output)
    self.assertEqual(error, "")

  def test_launcher_error_is_machine_readable_and_has_api_exit(self) -> None:
    with patch("server.control_cli.launch_job", side_effect=ClusterJobError("client image is not present")):
      code, output, error = self.run_cli("launch", "recipe.py", "--image", "client:missing", "--json")

    self.assertEqual(code, control_cli.EXIT_API)
    self.assertEqual(json.loads(output), {"status": "error", "error": "client image is not present"})
    self.assertEqual(error, "")

  def test_missing_image_and_malformed_recipe_args_are_usage_errors(self) -> None:
    code, output, error = self.run_cli("launch", "examples/tiny/tiny_rl.py", "--json")
    self.assertEqual(code, control_cli.EXIT_USAGE)
    self.assertIn("compatible client image is required", json.loads(output)["error"])
    self.assertEqual(error, "")

    code, output, error = self.run_cli(
      "launch",
      "examples/tiny/tiny_rl.py",
      "--image",
      "client:dev",
      "--args",
      "prompt='unterminated",
      "--json",
    )
    self.assertEqual(code, control_cli.EXIT_USAGE)
    self.assertIn("Invalid recipe args", json.loads(output)["error"])
    self.assertEqual(error, "")

  def test_launch_help_uses_public_kebab_case(self) -> None:
    code, output, error = self.run_cli("launch", "--help")

    self.assertEqual(code, control_cli.EXIT_OK)
    self.assertIn("Usage: openrl launch <source>", output)
    self.assertIn("--image-pull-policy", output)
    self.assertIn("--gateway-namespace", output)
    self.assertIn("--max-source-bytes", output)
    self.assertNotIn("image_pull_policy", output)
    self.assertEqual(error, "")


if __name__ == "__main__":
  unittest.main()
