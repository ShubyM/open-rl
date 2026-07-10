#!/usr/bin/env python3
"""Launch an Open-RL E2E client job and stream its Kubernetes logs."""

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "k8s" / "eval" / "e2e-client-job.yaml"
FATAL_WAIT_REASONS = {
  "CreateContainerConfigError",
  "CreateContainerError",
  "CrashLoopBackOff",
  "ErrImagePull",
  "ImagePullBackOff",
  "InvalidImageName",
}


def make_job_name(scenario: str, suffix: str | None = None) -> str:
  scenario_name = re.sub(r"[^a-z0-9-]+", "-", scenario.lower()).strip("-") or "run"
  suffix = suffix or f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
  prefix = "open-rl-e2e-"
  available = 63 - len(prefix) - len(suffix) - 1
  return f"{prefix}{scenario_name[:available].rstrip('-')}-{suffix}"


def render_manifest(job_name: str, scenario: str, extra_args_str: str, image: str) -> str:
  manifest = MANIFEST.read_text(encoding="utf-8")
  replacements = {
    "E2E-JOB-NAME": job_name,
    "E2E-IMAGE": image,
    "E2E-SCENARIO": scenario,
  }
  for placeholder, value in replacements.items():
    if placeholder not in manifest:
      raise RuntimeError(f"{MANIFEST} no longer contains the {placeholder} placeholder")
    manifest = manifest.replace(placeholder, value)

  extra_args = shlex.split(extra_args_str) if extra_args_str else []
  args_yaml = "\n".join(f"        - {json.dumps(arg)}" for arg in extra_args)
  placeholder = '        - "E2E-EXTRA-ARGS"'
  if placeholder not in manifest:
    raise RuntimeError(f"{MANIFEST} no longer contains the E2E-EXTRA-ARGS placeholder")
  return manifest.replace(placeholder, args_yaml)


def pod_failure(pod: dict) -> str | None:
  status = pod.get("status", {})
  phase = status.get("phase", "Unknown")
  if phase == "Failed":
    return status.get("message") or status.get("reason") or "pod entered Failed"

  for container in status.get("containerStatuses", []):
    state = container.get("state") or {}
    waiting = state.get("waiting") or {}
    if waiting.get("reason") in FATAL_WAIT_REASONS:
      detail = f": {waiting['message']}" if waiting.get("message") else ""
      return f"container {container.get('name', 'unknown')} is waiting with {waiting['reason']}{detail}"
    terminated = state.get("terminated") or {}
    if terminated.get("exitCode") not in {None, 0}:
      return f"container {container.get('name', 'unknown')} exited with code {terminated['exitCode']}"
  return None


def wait_for_job_pod(kubectl: list[str], job_name: str, timeout: float) -> str:
  deadline = time.monotonic() + timeout
  last_status = "no pod created"
  while time.monotonic() < deadline:
    result = subprocess.run(
      kubectl + ["get", "pods", "-l", f"job-name={job_name}", "-o", "json"],
      capture_output=True,
      text=True,
    )
    if result.returncode == 0:
      pods = json.loads(result.stdout).get("items", [])
      if pods:
        pod = pods[0]
        pod_name = pod["metadata"]["name"]
        pod_status = pod.get("status") or {}
        phase = pod_status.get("phase", "Unknown")
        last_status = f"pod/{pod_name} phase={phase}"
        if failure := pod_failure(pod):
          raise RuntimeError(f"{last_status}: {failure}")
        if phase in {"Running", "Succeeded"}:
          return pod_name
    elif result.stderr:
      last_status = result.stderr.strip()
    time.sleep(2)
  raise TimeoutError(f"job/{job_name} did not start within {timeout:.0f}s ({last_status})")


def wait_for_job_completion(kubectl: list[str], job_name: str, timeout: float) -> None:
  deadline = time.monotonic() + timeout
  last_status = "status unavailable"
  while time.monotonic() < deadline:
    result = subprocess.run(kubectl + ["get", "job", job_name, "-o", "json"], capture_output=True, text=True)
    if result.returncode == 0:
      status = json.loads(result.stdout).get("status", {})
      if status.get("succeeded", 0) > 0:
        return
      if status.get("failed", 0) > 0:
        raise RuntimeError(f"job/{job_name} failed")
      last_status = f"active={status.get('active', 0)}"
    elif result.stderr:
      last_status = result.stderr.strip()
    time.sleep(2)
  raise TimeoutError(f"job/{job_name} did not complete within {timeout:.0f}s ({last_status})")


def print_diagnostics(kubectl: list[str], job_name: str) -> None:
  print(f"\n[cluster-e2e] diagnostics for job/{job_name}", file=sys.stderr)
  commands = [
    ["get", "pods", "-l", f"job-name={job_name}", "-o", "wide"],
    ["describe", "job", job_name],
    ["describe", "pods", "-l", f"job-name={job_name}"],
    ["get", "events", "--sort-by=.lastTimestamp"],
    ["logs", f"job/{job_name}", "--all-containers=true", "--tail=200"],
  ]
  for args in commands:
    print("\n$ " + " ".join(kubectl + args), file=sys.stderr)
    subprocess.run(kubectl + args, check=False)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scenario", required=True, help="E2E scenario, such as fft-gsm8k-rl-x2.")
  parser.add_argument("--args", default="", help="Arguments passed to run_training_e2e.py.")
  parser.add_argument("--image", required=True, help="Client container image to run.")
  parser.add_argument("--name", default="", help="Exact Kubernetes Job name. Defaults to a unique run name.")
  parser.add_argument("--namespace", default="", help="Kubernetes namespace. Defaults to the current context.")
  parser.add_argument("--timeout", type=float, default=900, help="Startup and completion timeout in seconds.")
  parser.add_argument("--replace", action="store_true", help="Delete an existing job with the selected name before launch.")
  parser.add_argument("--cleanup", action="store_true", help="Delete the completed job after a successful run.")
  parser.add_argument("--no-follow", action="store_true", help="Launch the job without following logs.")
  parser.add_argument("--print-only", action="store_true", help="Print commands and the manifest without running them.")
  args = parser.parse_args()

  job_name = args.name or make_job_name(args.scenario)
  kubectl = ["kubectl"] + (["--namespace", args.namespace] if args.namespace else [])
  manifest = render_manifest(job_name, args.scenario, args.args, args.image)

  if args.print_only:
    if args.replace:
      print("$ " + " ".join(kubectl + ["delete", "job", job_name, "--ignore-not-found"]))
    print("$ " + " ".join(kubectl + ["create", "-f", "<manifest>"]))
    print("$ " + " ".join(kubectl + ["logs", "-f", f"job/{job_name}"]))
    print("\n# manifest applied at <manifest>:\n")
    print(manifest)
    return

  manifest_path = None
  try:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as manifest_file:
      manifest_file.write(manifest)
      manifest_path = Path(manifest_file.name)

    if args.replace:
      subprocess.run(kubectl + ["delete", "job", job_name, "--ignore-not-found"], check=True)
    subprocess.run(kubectl + ["create", "-f", str(manifest_path)], check=True)
    print(f"[cluster-e2e] launched job/{job_name}")
    if args.no_follow:
      print(f"[cluster-e2e] follow with: {' '.join(kubectl)} logs -f job/{job_name}")
      return

    wait_for_job_pod(kubectl, job_name, args.timeout)
    subprocess.run(kubectl + ["logs", "-f", f"job/{job_name}"], check=False)
    wait_for_job_completion(kubectl, job_name, args.timeout)
    if args.cleanup:
      subprocess.run(kubectl + ["delete", "job", job_name, "--wait=false"], check=True)
  except Exception as exc:
    print(f"[cluster-e2e] {exc}", file=sys.stderr)
    print_diagnostics(kubectl, job_name)
    raise SystemExit(1) from exc
  finally:
    if manifest_path is not None:
      manifest_path.unlink(missing_ok=True)


if __name__ == "__main__":
  main()
