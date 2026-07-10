#!/usr/bin/env python3
"""Launch an Open-RL E2E client job and stream its logs."""

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "k8s" / "eval" / "e2e-client-job.yaml"


def render_manifest(scenario: str, extra_args: str, image: str) -> str:
  manifest = MANIFEST.read_text(encoding="utf-8")
  for placeholder, value in (("E2E-IMAGE", image), ("E2E-SCENARIO", scenario)):
    if placeholder not in manifest:
      raise RuntimeError(f"{MANIFEST} is missing {placeholder}")
    manifest = manifest.replace(placeholder, value)

  placeholder = '        - "E2E-EXTRA-ARGS"'
  rendered_args = "\n".join(f"        - {json.dumps(value)}" for value in shlex.split(extra_args))
  if placeholder not in manifest:
    raise RuntimeError(f"{MANIFEST} is missing E2E-EXTRA-ARGS")
  return manifest.replace(placeholder, rendered_args)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scenario", required=True)
  parser.add_argument("--args", default="", help="Arguments passed to run_training_e2e.py.")
  parser.add_argument("--image", required=True)
  parser.add_argument("--namespace", default="")
  parser.add_argument("--no-follow", action="store_true")
  parser.add_argument("--print-only", action="store_true")
  args = parser.parse_args()

  kubectl = ["kubectl"] + (["--namespace", args.namespace] if args.namespace else [])
  manifest = render_manifest(args.scenario, args.args, args.image)
  if args.print_only:
    print("$ " + " ".join(kubectl + ["create", "-f", "<manifest>"]))
    print("$ " + " ".join(kubectl + ["logs", "-f", "job/<generated-name>"]))
    print("\n# manifest\n")
    print(manifest)
    return

  manifest_path: Path | None = None
  try:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as manifest_file:
      manifest_file.write(manifest)
      manifest_path = Path(manifest_file.name)

    created = subprocess.run(
      kubectl + ["create", "-f", str(manifest_path), "-o", "name"],
      check=True,
      capture_output=True,
      text=True,
    ).stdout.strip()
    job_name = created.split("/", 1)[-1]
    print(f"[cluster-e2e] launched {created}")
    if args.no_follow:
      print(f"[cluster-e2e] follow with: {' '.join(kubectl)} logs -f job/{job_name}")
      return

    subprocess.run(
      kubectl + ["wait", "--for=condition=Ready", "pod", "-l", f"job-name={job_name}", "--timeout=600s"],
      check=True,
    )
    subprocess.run(kubectl + ["logs", "-f", f"job/{job_name}"], check=True)
    subprocess.run(kubectl + ["wait", "--for=condition=Complete", created, "--timeout=900s"], check=True)
  finally:
    if manifest_path is not None:
      manifest_path.unlink(missing_ok=True)


if __name__ == "__main__":
  main()
