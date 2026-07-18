"""Run working-tree recipe code as a Kubernetes Job."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

DEFAULT_CLIENT_IMAGE = ""
DEFAULT_MAX_SOURCE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_SERVER_SOURCE_BYTES = 50 * 1024 * 1024
EXCLUDED_SOURCE_NAMES = {
  ".git",
  ".gnupg",
  ".kube",
  ".mypy_cache",
  ".netrc",
  ".npmrc",
  ".pypirc",
  ".pytest_cache",
  ".ruff_cache",
  ".ssh",
  ".venv",
  ".aws",
  ".azure",
  ".docker",
  ".env",
  ".envrc",
  ".git-credentials",
  ".terraform",
  ".terraform.d",
  "__pycache__",
  "artifacts",
  "checkpoints",
  "models",
  "node_modules",
  "wandb",
  ".claude",
  ".codex",
  "experiments",
  "runs",
  "scratch",
}
EXCLUDED_SOURCE_SUFFIXES = {".bin", ".key", ".pem", ".pyc", ".pyo", ".pt", ".pth", ".safetensors"}
FAILED_START_REASONS = {
  "CreateContainerConfigError",
  "CreateContainerError",
  "ErrImageNeverPull",
  "ErrImagePull",
  "ImagePullBackOff",
  "InvalidImageName",
  "RunContainerError",
}
JOB_NAME_PATTERN = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")


class ClusterJobError(RuntimeError):
  """A user-actionable Kubernetes job launch failure."""


class ClusterJobUsageError(ClusterJobError):
  """Invalid local launch configuration."""


@dataclass(frozen=True)
class JobConfig:
  source: str
  include: tuple[str, ...] = ()
  entrypoint: str = ""
  args: str = ""
  image: str = DEFAULT_CLIENT_IMAGE
  image_pull_policy: Literal["Always", "IfNotPresent", "Never"] = "IfNotPresent"
  context: str = ""
  namespace: str = ""
  gateway_namespace: str = ""
  gateway_url: str = ""
  name: str = ""
  detach: bool = False
  timeout: float = 600.0
  max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES
  request_cpu: str = "250m"
  request_ephemeral_storage: str = "256Mi"
  request_memory: str = "512Mi"
  limit_ephemeral_storage: str = "2Gi"
  limit_memory: str = "2Gi"
  workspace_size: str = "1Gi"
  env_secret: str = ""
  active_deadline_seconds: int = 86400
  ttl_seconds: int = 3600


@dataclass(frozen=True)
class SourceDeployConfig:
  source: str = "src"
  context: str = ""
  namespace: str = ""
  deployment: str = "open-rl-gateway"
  container: str = "gateway"
  max_source_bytes: int = DEFAULT_MAX_SERVER_SOURCE_BYTES
  timeout: float = 300.0
  reset_workers: bool = False


def source_is_excluded(relative_path: Path) -> bool:
  names = {part.lower() for part in relative_path.parts}
  env_file = relative_path.name.lower().startswith(".env.")
  return bool(EXCLUDED_SOURCE_NAMES.intersection(names)) or env_file or relative_path.suffix.lower() in EXCLUDED_SOURCE_SUFFIXES


def walk_source_files(root: Path, source: Path) -> list[Path]:
  files: list[Path] = []
  for current, directories, filenames in os.walk(source):
    current_path = Path(current)
    directories[:] = sorted(
      name for name in directories if not (current_path / name).is_symlink() and not source_is_excluded((current_path / name).relative_to(root))
    )
    files.extend(
      path for name in sorted(filenames) if not (path := current_path / name).is_symlink() and not source_is_excluded(path.relative_to(root))
    )
  return files


def source_files(source: Path, include: tuple[str, ...] = ()) -> tuple[Path, list[Path]]:
  if source.is_file():
    if include:
      raise ClusterJobError("include can only be used when source is a directory")
    if source_is_excluded(Path(source.name)):
      raise ClusterJobError(f"Source file {source} is excluded from job uploads")
    return source.parent, [source]
  if not source.is_dir():
    raise ClusterJobError(f"Source path does not exist: {source}")

  files: list[Path] = []
  selected = include or (".",)
  for item in selected:
    relative = Path(item)
    if relative.is_absolute() or ".." in relative.parts:
      raise ClusterJobError("include paths must stay inside the source directory")
    path = source / relative
    if not path.exists() or path.is_symlink():
      raise ClusterJobError(f"Included source path does not exist or is a symlink: {item}")
    if source_is_excluded(relative):
      raise ClusterJobError(f"Included source path is excluded from job uploads: {item}")
    if path.is_file():
      files.append(path)
    elif path.is_dir():
      files.extend(walk_source_files(source, path))

  files = sorted(set(files))
  if not files:
    raise ClusterJobError(f"Source directory has no uploadable files: {source}")
  return source, files


def resolve_entrypoint(source: Path, root: Path, files: list[Path], entrypoint: str) -> str:
  if source.is_file():
    if entrypoint and PurePosixPath(entrypoint).name != source.name:
      raise ClusterJobError("entrypoint must name the selected source file when source is a file")
    return source.name

  if entrypoint:
    relative = Path(entrypoint)
    if relative.is_absolute() or ".." in relative.parts:
      raise ClusterJobError("entrypoint must stay inside the source directory")
    candidate = root / relative
    if not candidate.is_file() or candidate.is_symlink() or candidate not in files:
      raise ClusterJobError(f"Entrypoint was not found in uploaded source: {entrypoint}")
    return relative.as_posix()

  candidates = [root / "train.py", root / "main.py"]
  candidates.extend(path for path in files if path.parent == root and path.suffix == ".py")
  unique = []
  for candidate in candidates:
    if candidate in files and candidate not in unique:
      unique.append(candidate)
  if len(unique) == 1:
    return unique[0].relative_to(root).as_posix()
  if root / "train.py" in unique:
    return "train.py"
  if root / "main.py" in unique:
    return "main.py"
  raise ClusterJobError("Source directory needs entrypoint=<relative-python-file>")


def build_source_archive(
  source: str | Path,
  entrypoint: str = "",
  max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
  include: tuple[str, ...] = (),
) -> tuple[bytes, str, int]:
  source_path = Path(source).expanduser().resolve()
  root, files = source_files(source_path, include)
  resolved_entrypoint = resolve_entrypoint(source_path, root, files, entrypoint)
  source_bytes = sum(path.stat().st_size for path in files)
  if source_bytes > max_source_bytes:
    raise ClusterJobError(
      f"Refusing to upload {source_bytes} bytes of source (limit: {max_source_bytes}); exclude generated data or build it into the client image"
    )

  archive_buffer = io.BytesIO()
  with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
    for path in files:
      archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
  return archive_buffer.getvalue(), resolved_entrypoint, source_bytes


def build_source_directory_archive(source: str | Path, max_source_bytes: int = DEFAULT_MAX_SERVER_SOURCE_BYTES) -> tuple[bytes, str, int]:
  """Archive importable server source and return a stable content revision."""
  source_path = Path(source).expanduser().resolve()
  if not source_path.is_dir():
    raise ClusterJobError(f"Server source must be a directory: {source_path}")
  root, files = source_files(source_path)
  source_bytes = sum(path.stat().st_size for path in files)
  if source_bytes > max_source_bytes:
    raise ClusterJobError(f"Refusing to upload {source_bytes} bytes of server source (limit: {max_source_bytes})")

  digest = hashlib.sha256()
  archive_buffer = io.BytesIO()
  with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
    for path in files:
      relative = path.relative_to(root).as_posix()
      digest.update(relative.encode("utf-8") + b"\0")
      with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
          digest.update(chunk)
      digest.update(b"\0")
      archive.add(path, arcname=relative, recursive=False)
  return archive_buffer.getvalue(), digest.hexdigest()[:16], source_bytes


def safe_job_name(value: str) -> str:
  name = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
  return name[:40].rstrip("-") or "recipe"


def generated_job_name(config: JobConfig) -> str:
  if config.name:
    if len(config.name) > 63 or JOB_NAME_PATTERN.fullmatch(config.name) is None:
      raise ClusterJobError("name must already be a lowercase Kubernetes DNS label")
    return config.name
  stem = safe_job_name(Path(config.source).stem)
  suffix = secrets.token_hex(3)
  return f"open-rl-{stem[: 54 - len(suffix)]}-{suffix}"


def kubectl_command(config: JobConfig, namespace: str | None = None) -> list[str]:
  command = ["kubectl"]
  if config.context:
    command.extend(["--context", config.context])
  if namespace:
    command.extend(["--namespace", namespace])
  return command


def run_kubectl(
  command: list[str],
  *,
  input_data: str | bytes | None = None,
  capture_output: bool = True,
) -> subprocess.CompletedProcess:
  try:
    return subprocess.run(
      command,
      input=input_data,
      check=True,
      capture_output=capture_output,
      text=isinstance(input_data, str) or input_data is None,
    )
  except FileNotFoundError as exc:
    raise ClusterJobError("kubectl is required to launch an in-cluster job") from exc
  except subprocess.CalledProcessError as exc:
    stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
    detail = (stderr or str(exc)).strip()
    raise ClusterJobError(f"kubectl failed: {detail}") from exc


def resolve_namespace(config: JobConfig) -> str:
  if config.namespace:
    return config.namespace
  command = kubectl_command(config) + ["config", "view", "--minify", "-o", "jsonpath={..namespace}"]
  try:
    namespace = run_kubectl(command).stdout.strip()
  except ClusterJobError as exc:
    raise ClusterJobError("Could not determine a Kubernetes namespace; pass namespace=<name> and context=<name>") from exc
  return namespace or "default"


def internal_gateway_url(config: JobConfig, namespace: str) -> str:
  if config.gateway_url:
    return config.gateway_url.rstrip("/")
  gateway_namespace = config.gateway_namespace or namespace
  return f"http://open-rl-gateway-service.{gateway_namespace}.svc:8000"


def require_gateway_service(config: JobConfig, namespace: str) -> None:
  if config.gateway_url:
    return
  gateway_namespace = config.gateway_namespace or namespace
  command = kubectl_command(config, gateway_namespace) + ["get", "service/open-rl-gateway-service", "-o", "name"]
  try:
    run_kubectl(command)
  except ClusterJobError as exc:
    raise ClusterJobError(
      f"Gateway service open-rl-gateway-service was not found in namespace {gateway_namespace!r}; set namespace=<name> or gateway_namespace=<name>"
    ) from exc


def build_job_manifest(
  config: JobConfig,
  job_name: str,
  namespace: str,
  entrypoint: str,
  recipe_args: list[str],
) -> dict[str, Any]:
  gateway_url = internal_gateway_url(config, namespace)
  labels = {
    "app": "open-rl-client",
    "app.kubernetes.io/component": "recipe",
    "app.kubernetes.io/managed-by": "openrl",
  }
  wait_script = "\n".join(
    [
      "until [ -f /workspace/.openrl-ready ]; do sleep 0.2; done",
      "python=/app/examples/.venv/bin/python",
      'if [ ! -x "$python" ]; then python=/app/.venv/bin/python; fi',
      "if [ ! -x \"$python\" ]; then echo 'client image has no OpenRL Python environment' >&2; exit 127; fi",
      'exec "$python" "$@"',
    ]
  )
  container = {
    "name": "recipe",
    "image": config.image,
    "imagePullPolicy": config.image_pull_policy,
    "command": ["/bin/sh", "-ec"],
    "args": [wait_script, "openrl-job", f"/workspace/{entrypoint}", *recipe_args],
    "env": [
      {"name": "BASE_URL", "value": gateway_url},
      {"name": "HF_HOME", "value": "/tmp/huggingface"},
      {"name": "HOME", "value": "/tmp"},
      {"name": "OPEN_RL_EXAMPLES_UV_PROJECT_ENVIRONMENT", "value": "/app/examples/.venv"},
      {"name": "OPENRL_BASE_URL", "value": gateway_url},
      {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
      {"name": "PYTHONPATH", "value": "/workspace:/app/examples"},
      {"name": "PYTHONUNBUFFERED", "value": "1"},
      {"name": "TINKER_API_KEY", "value": "tml-dummy-key"},
      {"name": "TINKER_BASE_URL", "value": gateway_url},
    ],
    "resources": {
      "requests": {
        "cpu": config.request_cpu,
        "ephemeral-storage": config.request_ephemeral_storage,
        "memory": config.request_memory,
      },
      "limits": {
        "ephemeral-storage": config.limit_ephemeral_storage,
        "memory": config.limit_memory,
      },
    },
    "securityContext": {
      "allowPrivilegeEscalation": False,
      "capabilities": {"drop": ["ALL"]},
      "runAsGroup": 65532,
      "runAsNonRoot": True,
      "runAsUser": 65532,
    },
    "volumeMounts": [{"name": "workspace", "mountPath": "/workspace"}],
  }
  if config.env_secret:
    container["envFrom"] = [{"secretRef": {"name": config.env_secret}}]
  return {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {"name": job_name, "namespace": namespace, "labels": labels},
    "spec": {
      "activeDeadlineSeconds": config.active_deadline_seconds,
      "backoffLimit": 0,
      "ttlSecondsAfterFinished": config.ttl_seconds,
      "template": {
        "metadata": {"labels": labels},
        "spec": {
          "automountServiceAccountToken": False,
          "enableServiceLinks": False,
          "restartPolicy": "Never",
          "securityContext": {
            "fsGroup": 65532,
            "fsGroupChangePolicy": "OnRootMismatch",
            "seccompProfile": {"type": "RuntimeDefault"},
          },
          "containers": [container],
          "volumes": [{"name": "workspace", "emptyDir": {"sizeLimit": config.workspace_size}}],
        },
      },
    },
  }


def pod_start_failure(pod: dict[str, Any]) -> str | None:
  pod_status = pod.get("status", {})
  phase = pod_status.get("phase")
  if phase == "Failed":
    return f"{pod_status.get('reason') or 'Failed'}: {pod_status.get('message') or 'pod failed before the recipe started'}"
  for condition in pod_status.get("conditions") or []:
    if condition.get("type") == "PodScheduled" and condition.get("status") == "False":
      return f"{condition.get('reason') or 'Unschedulable'}: {condition.get('message') or 'pod could not be scheduled'}"
  statuses = [*(pod_status.get("initContainerStatuses") or []), *(pod_status.get("containerStatuses") or [])]
  for status in statuses:
    state = status.get("state", {})
    waiting = state.get("waiting") or {}
    reason = waiting.get("reason")
    if reason in FAILED_START_REASONS:
      return f"{reason}: {waiting.get('message') or 'container could not start'}"
    terminated = state.get("terminated") or {}
    if terminated:
      message = terminated.get("message") or f"container exited with code {terminated.get('exitCode')}"
      return f"{terminated.get('reason') or 'Terminated'}: {message}"
  return None


def wait_for_job_pod(config: JobConfig, namespace: str, job_name: str) -> tuple[str, dict[str, Any]]:
  command = kubectl_command(config, namespace)
  deadline = time.monotonic() + config.timeout
  last_phase = "not created"
  while time.monotonic() < deadline:
    response = run_kubectl(command + ["get", "pods", "-l", f"job-name={job_name}", "-o", "json"])
    items = json.loads(response.stdout).get("items") or []
    if items:
      pod = items[0]
      pod_name = str(pod["metadata"]["name"])
      last_phase = str(pod.get("status", {}).get("phase") or "Pending")
      if failure := pod_start_failure(pod):
        raise ClusterJobError(f"Job pod {pod_name} failed to start: {failure}")
      conditions = pod.get("status", {}).get("conditions") or []
      if any(condition.get("type") == "Ready" and condition.get("status") == "True" for condition in conditions):
        return pod_name, pod
    time.sleep(0.5)
  raise ClusterJobError(f"Timed out after {config.timeout:g}s waiting for job {job_name} (last phase: {last_phase})")


def upload_source(config: JobConfig, namespace: str, pod_name: str, archive: bytes) -> None:
  extractor = "import sys,tarfile; archive=tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz'); archive.extractall('/workspace',filter='data')"
  command = kubectl_command(config, namespace) + ["exec", "-i", pod_name, "--", "/usr/local/bin/python", "-c", extractor]
  run_kubectl(command, input_data=archive)
  run_kubectl(kubectl_command(config, namespace) + ["exec", pod_name, "--", "touch", "/workspace/.openrl-ready"])


def job_result(config: JobConfig, namespace: str, job_name: str, pod_name: str, status: str) -> dict[str, Any]:
  command = kubectl_command(config, namespace)
  return {
    "job": job_name,
    "namespace": namespace,
    "pod": pod_name,
    "gateway_url": internal_gateway_url(config, namespace),
    "status": status,
    "follow_command": command + ["logs", "-f", f"job/{job_name}"],
    "stop_command": command + ["delete", f"job/{job_name}"],
  }


def wait_for_job_result(config: JobConfig, namespace: str, job_name: str, pod_name: str) -> dict[str, Any]:
  command = kubectl_command(config, namespace)
  deadline = time.monotonic() + config.timeout
  try:
    subprocess.run(
      command + ["logs", "-f", f"job/{job_name}"],
      check=False,
      stdout=sys.stderr,
      stderr=sys.stderr,
      timeout=config.timeout,
    )
  except subprocess.TimeoutExpired as exc:
    raise ClusterJobError(
      f"Job {job_name} is still running after {config.timeout:g}s; follow it with {shlex.join(command + ['logs', '-f', f'job/{job_name}'])}"
    ) from exc
  while time.monotonic() < deadline:
    response = run_kubectl(command + ["get", f"job/{job_name}", "-o", "json"])
    status = json.loads(response.stdout).get("status") or {}
    if int(status.get("succeeded") or 0) > 0:
      return job_result(config, namespace, job_name, pod_name, "complete")
    if int(status.get("failed") or 0) > 0:
      result = job_result(config, namespace, job_name, pod_name, "failed")
      pod_response = run_kubectl(command + ["get", f"pod/{pod_name}", "-o", "json"])
      pod = json.loads(pod_response.stdout)
      statuses = pod.get("status", {}).get("containerStatuses") or []
      if statuses:
        result["exit_code"] = statuses[0].get("state", {}).get("terminated", {}).get("exitCode")
      return result
    time.sleep(0.5)
  raise ClusterJobError(f"Timed out after {config.timeout:g}s waiting for job {job_name} to finish")


def launch_job(config: JobConfig) -> dict[str, Any]:
  if not config.image:
    raise ClusterJobUsageError(
      "A compatible client image is required; pass image=<tag> or set OPENRL_CLIENT_IMAGE. Build it only when client dependencies change."
    )
  if config.timeout <= 0:
    raise ClusterJobUsageError("timeout must be positive")
  if config.active_deadline_seconds <= 0 or config.ttl_seconds < 0:
    raise ClusterJobUsageError("active_deadline_seconds must be positive and ttl_seconds cannot be negative")
  archive, entrypoint, source_bytes = build_source_archive(
    config.source,
    config.entrypoint,
    config.max_source_bytes,
    config.include,
  )
  try:
    recipe_args = shlex.split(config.args)
  except ValueError as exc:
    raise ClusterJobUsageError(f"Invalid recipe args: {exc}") from exc
  namespace = resolve_namespace(config)
  require_gateway_service(config, namespace)
  job_name = generated_job_name(config)
  manifest = build_job_manifest(config, job_name, namespace, entrypoint, recipe_args)
  command = kubectl_command(config, namespace)
  job_created = False
  recipe_started = False
  try:
    run_kubectl(command + ["create", "-f", "-"], input_data=json.dumps(manifest))
    job_created = True
    print(f"[openrl] created job/{job_name}; uploading {source_bytes} bytes of source", file=sys.stderr)
    pod_name, _ = wait_for_job_pod(config, namespace, job_name)
    upload_source(config, namespace, pod_name, archive)
    recipe_started = True
    result = job_result(config, namespace, job_name, pod_name, "running")
    print(f"[openrl] job/{job_name} is running against {result['gateway_url']}", file=sys.stderr)
    if config.detach:
      return result
    return wait_for_job_result(config, namespace, job_name, pod_name)
  except BaseException:
    if job_created and not recipe_started:
      subprocess.run(command + ["delete", f"job/{job_name}", "--wait=false"], check=False, capture_output=True, text=True)
    raise


def deploy_source(config: SourceDeployConfig) -> dict[str, Any]:
  """Install current Python source on the shared PVC and roll the gateway.

  Dependency and image changes use the explicit Docker + kubectl slow path.
  Normal edits under ``src`` avoid rebuilding the multi-gigabyte GPU image.
  """
  if config.timeout <= 0:
    raise ClusterJobUsageError("timeout must be positive")
  archive, revision, source_bytes = build_source_directory_archive(config.source, config.max_source_bytes)
  job_config = JobConfig(source=config.source, context=config.context, namespace=config.namespace)
  namespace = resolve_namespace(job_config)
  command = kubectl_command(job_config, namespace)
  pods_response = run_kubectl(command + ["get", "pods", "-l", f"app={config.deployment}", "-o", "json"])
  pods = json.loads(pods_response.stdout).get("items") or []
  ready_pods = [
    pod
    for pod in pods
    if pod.get("status", {}).get("phase") == "Running"
    and any(condition.get("type") == "Ready" and condition.get("status") == "True" for condition in pod.get("status", {}).get("conditions") or [])
  ]
  if not ready_pods:
    raise ClusterJobError(f"No ready {config.deployment} pod was found in namespace {namespace!r}; deploy the platform image first")
  pod_name = str(ready_pods[0]["metadata"]["name"])
  source_path = f"/mnt/shared/open-rl/source/{revision}"
  extractor = (
    "import os,shutil,sys,tarfile;"
    f"target={source_path!r};tmp=target+'.tmp';"
    "existing=os.path.isdir(target);"
    "sys.stdin.buffer.read() if existing else None;"
    "shutil.rmtree(tmp,ignore_errors=True) if not existing else None;"
    "os.makedirs(tmp,exist_ok=True) if not existing else None;"
    "tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz').extractall(tmp,filter='data') if not existing else None;"
    "os.rename(tmp,target) if not existing else None"
  )
  run_kubectl(
    command + ["exec", "-i", pod_name, "-c", config.container, "--", "/app/.venv/bin/python", "-c", extractor],
    input_data=archive,
  )
  run_kubectl(
    command
    + [
      "set",
      "env",
      f"deployment/{config.deployment}",
      f"PYTHONPATH={source_path}",
      f"OPEN_RL_SOURCE_REVISION={revision}",
    ]
  )
  run_kubectl(command + ["rollout", "status", f"deployment/{config.deployment}", f"--timeout={config.timeout:g}s"], capture_output=False)
  if config.reset_workers:
    run_kubectl(command + ["delete", "pods", "-l", "accel-timeslicer=true", "--ignore-not-found", "--wait=false"])
  return {
    "status": "deployed",
    "revision": revision,
    "source_bytes": source_bytes,
    "source_path": source_path,
    "namespace": namespace,
    "deployment": config.deployment,
    "workers_reset": config.reset_workers,
  }
