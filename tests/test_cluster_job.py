import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server.cluster_job import (
  ClusterJobError,
  JobConfig,
  SourceDeployConfig,
  build_job_manifest,
  build_source_archive,
  build_source_directory_archive,
  deploy_source,
  generated_job_name,
  kubectl_command,
  launch_job,
  pod_start_failure,
)

ROOT = Path(__file__).resolve().parents[1]


def archived_files(payload: bytes) -> set[str]:
  with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
    return {member.name for member in archive.getmembers() if member.isfile()}


class ClusterJobArchiveTest(unittest.TestCase):
  def test_single_file_becomes_the_archive_entrypoint(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      source = Path(temp) / "tiny_rl.py"
      source.write_text("print('ready')\n", encoding="utf-8")

      payload, entrypoint, source_bytes = build_source_archive(source)

    self.assertEqual(entrypoint, "tiny_rl.py")
    self.assertEqual(source_bytes, len(b"print('ready')\n"))
    self.assertEqual(archived_files(payload), {"tiny_rl.py"})

  def test_directory_is_rooted_at_workspace_and_excludes_generated_state(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      source = Path(temp) / "recipe"
      (source / "package").mkdir(parents=True)
      (source / "package" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
      (source / "train.py").write_text("from package.helper import VALUE\n", encoding="utf-8")
      for excluded in (".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", "wandb", ".aws", ".kube", ".ssh"):
        (source / excluded).mkdir()
        (source / excluded / "large.bin").write_bytes(b"x" * 1024)
      (source / ".env.local").write_text("TOKEN=secret\n", encoding="utf-8")
      outside = Path(temp) / "outside.py"
      outside.write_text("SECRET = True\n", encoding="utf-8")
      (source / "external.py").symlink_to(outside)

      payload, entrypoint, source_bytes = build_source_archive(source, "train.py")

    expected = {"package/helper.py", "train.py"}
    self.assertEqual(entrypoint, "train.py")
    self.assertEqual(archived_files(payload), expected)
    self.assertEqual(source_bytes, len(b"VALUE = 1\n") + len(b"from package.helper import VALUE\n"))

  def test_archive_limit_counts_only_uploadable_raw_source(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      source = Path(temp)
      (source / "main.py").write_bytes(b"12345678")
      (source / ".venv").mkdir()
      (source / ".venv" / "ignored.bin").write_bytes(b"x" * 1024)

      _, entrypoint, source_bytes = build_source_archive(source, max_source_bytes=8)
      self.assertEqual(entrypoint, "main.py")
      self.assertEqual(source_bytes, 8)

      with self.assertRaises(ClusterJobError):
        build_source_archive(source, max_source_bytes=7)

  def test_include_builds_a_curated_bundle_without_scanning_sibling_state(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      source = Path(temp)
      (source / "scripts").mkdir()
      (source / "scripts" / "run.py").write_text("print('ready')\n", encoding="utf-8")
      (source / "examples").mkdir()
      (source / "examples" / "recipe.py").write_text("VALUE = 1\n", encoding="utf-8")
      (source / "huge-local-state").mkdir()
      (source / "huge-local-state" / "weights.data").write_bytes(b"x" * 1024)

      payload, entrypoint, source_bytes = build_source_archive(
        source,
        "scripts/run.py",
        include=("scripts/run.py", "examples"),
      )

    self.assertEqual(entrypoint, "scripts/run.py")
    self.assertEqual(archived_files(payload), {"scripts/run.py", "examples/recipe.py"})
    self.assertEqual(source_bytes, len(b"print('ready')\n") + len(b"VALUE = 1\n"))

  def test_entrypoint_cannot_escape_or_name_an_excluded_file(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      source = Path(temp) / "recipe"
      source.mkdir()
      (source / "main.py").write_text("pass\n", encoding="utf-8")
      (source / ".venv").mkdir()
      (source / ".venv" / "hidden.py").write_text("pass\n", encoding="utf-8")

      for entrypoint in ("../main.py", ".venv/hidden.py", "missing.py"):
        with self.subTest(entrypoint=entrypoint), self.assertRaises(ClusterJobError):
          build_source_archive(source, entrypoint)

  def test_server_source_revision_only_changes_with_uploadable_content(self) -> None:
    with tempfile.TemporaryDirectory() as temp:
      source = Path(temp) / "src"
      (source / "server").mkdir(parents=True)
      module = source / "server" / "gateway.py"
      module.write_text("VALUE = 1\n", encoding="utf-8")
      _, first_revision, source_bytes = build_source_directory_archive(source)
      (source / "__pycache__").mkdir()
      (source / "__pycache__" / "gateway.pyc").write_bytes(b"ignored")
      _, same_revision, _ = build_source_directory_archive(source)
      module.write_text("VALUE = 2\n", encoding="utf-8")
      payload, changed_revision, _ = build_source_directory_archive(source)

    self.assertEqual(source_bytes, len(b"VALUE = 1\n"))
    self.assertEqual(first_revision, same_revision)
    self.assertNotEqual(first_revision, changed_revision)
    self.assertEqual(archived_files(payload), {"server/gateway.py"})


class ClusterJobManifestTest(unittest.TestCase):
  def test_manifest_is_a_small_unprivileged_source_streaming_job(self) -> None:
    config = JobConfig(
      source="recipe",
      image="example/openrl-client:dev",
      image_pull_policy="Never",
      gateway_namespace="open-rl-system",
      request_cpu="100m",
      request_memory="256Mi",
      limit_memory="1Gi",
      env_secret="wandb-credentials",
      ttl_seconds=300,
    )

    manifest = build_job_manifest(
      config,
      job_name="math-recipe",
      namespace="jobs",
      entrypoint="recipes/train.py",
      recipe_args=["steps=2", "prompt=hello world"],
    )

    self.assertEqual(manifest["apiVersion"], "batch/v1")
    self.assertEqual(manifest["kind"], "Job")
    self.assertEqual(manifest["metadata"]["name"], "math-recipe")
    self.assertEqual(manifest["metadata"]["namespace"], "jobs")
    self.assertEqual(manifest["spec"]["backoffLimit"], 0)
    self.assertEqual(manifest["spec"]["activeDeadlineSeconds"], 86400)
    self.assertEqual(manifest["spec"]["ttlSecondsAfterFinished"], 300)

    pod = manifest["spec"]["template"]["spec"]
    self.assertEqual(pod["restartPolicy"], "Never")
    self.assertFalse(pod["automountServiceAccountToken"])
    self.assertNotIn("serviceAccountName", pod)
    self.assertNotIn("persistentVolumeClaim", json.dumps(pod))
    self.assertEqual(pod["securityContext"]["fsGroup"], 65532)
    self.assertEqual(pod["securityContext"]["seccompProfile"], {"type": "RuntimeDefault"})
    self.assertEqual(pod["volumes"], [{"name": "workspace", "emptyDir": {"sizeLimit": "1Gi"}}])

    container = pod["containers"][0]
    self.assertEqual(container["image"], "example/openrl-client:dev")
    self.assertIn(
      {"name": "OPEN_RL_EXAMPLES_UV_PROJECT_ENVIRONMENT", "value": "/app/examples/.venv"},
      container["env"],
    )
    self.assertEqual(container["imagePullPolicy"], "Never")
    self.assertEqual(container["resources"]["requests"], {"cpu": "100m", "ephemeral-storage": "256Mi", "memory": "256Mi"})
    self.assertEqual(container["resources"]["limits"], {"ephemeral-storage": "2Gi", "memory": "1Gi"})
    self.assertTrue(container["securityContext"]["runAsNonRoot"])
    self.assertFalse(container["securityContext"]["allowPrivilegeEscalation"])
    self.assertEqual(container["envFrom"], [{"secretRef": {"name": "wandb-credentials"}}])
    self.assertIn("steps=2", container["args"])
    self.assertIn("prompt=hello world", container["args"])
    self.assertIn("/workspace/recipes/train.py", container["args"])
    subprocess.run(["sh", "-n", "-c", container["args"][0]], check=True)

    environment = {item["name"]: item["value"] for item in container["env"]}
    gateway_url = "http://open-rl-gateway-service.open-rl-system.svc:8000"
    self.assertEqual(environment["BASE_URL"], gateway_url)
    self.assertEqual(environment["OPENRL_BASE_URL"], gateway_url)
    self.assertEqual(environment["TINKER_BASE_URL"], gateway_url)
    self.assertEqual(environment["TINKER_API_KEY"], "tml-dummy-key")

  def test_explicit_gateway_url_wins(self) -> None:
    config = JobConfig(
      source="recipe.py",
      image="example/client:dev",
      gateway_namespace="ignored",
      gateway_url="http://custom-gateway:7000/",
    )

    manifest = build_job_manifest(config, "recipe", "jobs", "recipe.py", [])

    environment = {item["name"]: item["value"] for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]}
    self.assertEqual(environment["TINKER_BASE_URL"], "http://custom-gateway:7000")

  def test_kubectl_command_has_optional_context_and_explicit_namespace(self) -> None:
    config = JobConfig(source="recipe.py", image="client:dev", context="kind-open-rl", namespace="configured")

    self.assertEqual(kubectl_command(config), ["kubectl", "--context", "kind-open-rl"])
    self.assertEqual(
      kubectl_command(config, "jobs"),
      ["kubectl", "--context", "kind-open-rl", "--namespace", "jobs"],
    )

  def test_explicit_dns_name_can_use_the_full_kubernetes_limit(self) -> None:
    name = f"a{'b' * 61}z"
    self.assertEqual(generated_job_name(JobConfig(source="recipe.py", image="client:dev", name=name)), name)

  def test_start_failure_surfaces_unschedulable_and_terminated_pods(self) -> None:
    unschedulable = {
      "status": {
        "phase": "Pending",
        "conditions": [{"type": "PodScheduled", "status": "False", "reason": "Unschedulable", "message": "insufficient memory"}],
      }
    }
    self.assertEqual(pod_start_failure(unschedulable), "Unschedulable: insufficient memory")

    terminated = {
      "status": {
        "phase": "Running",
        "containerStatuses": [{"state": {"terminated": {"reason": "Error", "exitCode": 17}}}],
      }
    }
    self.assertEqual(pod_start_failure(terminated), "Error: container exited with code 17")


class ClusterJobLaunchTest(unittest.TestCase):
  def test_detached_launch_streams_source_and_returns_agent_commands(self) -> None:
    calls: list[tuple[list[str], object]] = []
    pod = {
      "metadata": {"name": "recipe-pod"},
      "status": {"phase": "Running", "conditions": [{"type": "Ready", "status": "True"}]},
    }

    def run(command, *, input_data=None, capture_output=True):
      calls.append((command, input_data))
      if "get" in command and "pods" in command:
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"items": [pod]}), stderr="")
      return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with tempfile.TemporaryDirectory() as temp:
      source = Path(temp) / "recipe.py"
      source.write_text("print('cluster')\n", encoding="utf-8")
      config = JobConfig(
        source=str(source),
        image="client:dev",
        image_pull_policy="Never",
        context="kind-open-rl",
        namespace="open-rl-local",
        name="recipe",
        detach=True,
      )
      with patch("server.cluster_job.run_kubectl", side_effect=run):
        result = launch_job(config)

    self.assertEqual(
      {key: result[key] for key in ("job", "namespace", "pod", "status")},
      {"job": "recipe", "namespace": "open-rl-local", "pod": "recipe-pod", "status": "running"},
    )
    self.assertEqual(
      result["gateway_url"],
      "http://open-rl-gateway-service.open-rl-local.svc:8000",
    )
    self.assertEqual(result["follow_command"][-3:], ["logs", "-f", "job/recipe"])
    self.assertEqual(result["stop_command"][-2:], ["delete", "job/recipe"])
    self.assertFalse(any("port-forward" in command for command, _ in calls))

    create_call = next((command, data) for command, data in calls if "create" in command)
    self.assertEqual(json.loads(create_call[1])["metadata"]["name"], "recipe")
    upload_call = next((command, data) for command, data in calls if "exec" in command and isinstance(data, bytes))
    self.assertEqual(archived_files(upload_call[1]), {"recipe.py"})


class SourceDeployTest(unittest.TestCase):
  def test_deploy_streams_source_to_pvc_and_rolls_gateway_without_an_image_build(self) -> None:
    calls: list[tuple[list[str], object, bool]] = []
    pod = {
      "metadata": {"name": "open-rl-gateway-abc"},
      "status": {"phase": "Running", "conditions": [{"type": "Ready", "status": "True"}]},
    }

    def run(command, *, input_data=None, capture_output=True):
      calls.append((command, input_data, capture_output))
      if "get" in command and "pods" in command:
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"items": [pod]}), stderr="")
      return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with tempfile.TemporaryDirectory() as temp:
      source = Path(temp) / "src"
      (source / "server").mkdir(parents=True)
      (source / "server" / "gateway.py").write_text("VALUE = 1\n", encoding="utf-8")
      config = SourceDeployConfig(source=str(source), namespace="training", reset_workers=True)
      with patch("server.cluster_job.run_kubectl", side_effect=run):
        result = deploy_source(config)

    self.assertEqual(result["status"], "deployed")
    self.assertTrue(result["workers_reset"])
    upload = next((command, data, _) for command, data, _ in calls if "exec" in command)
    self.assertEqual(archived_files(upload[1]), {"server/gateway.py"})
    compile(upload[0][-1], "<source-deploy-extractor>", "exec")
    set_env = next(command for command, _, _ in calls if "set" in command and "env" in command)
    self.assertIn(f"PYTHONPATH={result['source_path']}", set_env)
    self.assertIn(f"OPEN_RL_SOURCE_REVISION={result['revision']}", set_env)
    self.assertTrue(any("rollout" in command and "status" in command for command, _, _ in calls))
    self.assertTrue(any("delete" in command and "accel-timeslicer=true" in command for command, _, _ in calls))


class ClusterDeploymentWorkflowTest(unittest.TestCase):
  def test_slow_path_is_plain_docker_and_kubectl(self) -> None:
    result = subprocess.run(
      [
        "make",
        "--no-print-directory",
        "-n",
        "push-to-cluster",
        "GCP_PROJECT=test-project",
        "IMAGE_TAG=test-revision",
        "K8S_DIR=k8s/deploy/distributed-fft-timeslice",
      ],
      cwd=ROOT,
      check=True,
      capture_output=True,
      text=True,
    )

    self.assertIn("docker build", result.stdout)
    self.assertIn("docker push gcr.io/test-project/open-rl-server:test-revision", result.stdout)
    self.assertIn("kubectl apply -k k8s/deploy/distributed-fft-timeslice", result.stdout)
    self.assertIn("kubectl set image deployment/open-rl-gateway", result.stdout)
    self.assertIn("OPEN_RL_WORKER_REVISION=test-revision", result.stdout)
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    self.assertNotIn("sha256sum", makefile)


if __name__ == "__main__":
  unittest.main()
