#!/usr/bin/env python3
"""Run the one-off vLLM eval job (k8s/eval/vllm-eval-job.yaml) on the cluster.

Stamps the checkpoint path into the job manifest, applies it with kubectl, and
follows the logs until the accuracy line prints. Stdlib only - no uv needed:

  python3 scripts/run_cluster_eval.py --model-path /mnt/shared/open-rl/checkpoints/<model-id>/weights/final
  python3 scripts/run_cluster_eval.py --model-path Qwen/Qwen2.5-0.5B --examples 50
  python3 scripts/run_cluster_eval.py --model-path ... --data-path /mnt/shared/open-rl/evals/my-prompts.json

--print-only shows the kubectl commands and rendered manifest without running
anything (handy when kubectl points elsewhere or you want to run them by hand).
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "k8s" / "eval" / "vllm-eval-job.yaml"
JOB = "open-rl-vllm-eval"


def render_manifest(model_path: str, examples: int, data_path: str) -> str:
  manifest = MANIFEST.read_text(encoding="utf-8")
  for placeholder, value in [("EVAL-MODEL-PATH", model_path), ("EVAL-EXAMPLES", str(examples)), ("EVAL-DATA-PATH", data_path)]:
    if placeholder not in manifest:
      raise RuntimeError(f"{MANIFEST} no longer contains the {placeholder} placeholder")
    manifest = manifest.replace(placeholder, value)
  return manifest


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--model-path", required=True, help="Checkpoint directory on the shared PVC, or an HF model id.")
  parser.add_argument("--examples", type=int, default=100, help="GSM8K test examples to evaluate (ignored with --data-path).")
  parser.add_argument("--data-path", default="", help="Optional [{prompt, gold}] JSON on the PVC instead of GSM8K.")
  parser.add_argument("--namespace", default="", help="Kubernetes namespace (defaults to the kubectl context's).")
  parser.add_argument("--no-follow", action="store_true", help="Launch the job but do not follow its logs.")
  parser.add_argument("--print-only", action="store_true", help="Print the kubectl commands and manifest; run nothing.")
  args = parser.parse_args()

  kubectl = ["kubectl"] + (["-n", args.namespace] if args.namespace else [])
  manifest = render_manifest(args.model_path, args.examples, args.data_path)
  commands = [
    kubectl + ["delete", "job", JOB, "--ignore-not-found"],
    kubectl + ["apply", "-f", "<manifest>"],
    kubectl + ["wait", "--for=condition=Ready", "pod", "-l", f"job-name={JOB}", "--timeout=600s"],
    kubectl + ["logs", "-f", f"job/{JOB}"],
  ]

  if args.print_only:
    for command in commands:
      print("$ " + " ".join(command))
    print("\n# manifest applied at <manifest>:\n")
    print(manifest)
    return

  with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
    f.write(manifest)
    manifest_path = f.name

  subprocess.run(commands[0], check=True)
  subprocess.run(kubectl + ["apply", "-f", manifest_path], check=True)
  if args.no_follow:
    print(f"[cluster-eval] launched job/{JOB}; follow it with: {' '.join(kubectl)} logs -f job/{JOB}")
    return
  subprocess.run(commands[2], check=True)
  subprocess.run(commands[3], check=True)
  # logs -f ending does not imply success; report the job's actual verdict.
  done = subprocess.run(kubectl + ["wait", "--for=condition=Complete", f"job/{JOB}", "--timeout=30s"], capture_output=True, text=True)
  if done.returncode != 0:
    print(f"[cluster-eval] job did not complete cleanly; inspect with: {' '.join(kubectl)} describe job/{JOB}")
    sys.exit(1)


if __name__ == "__main__":
  main()
