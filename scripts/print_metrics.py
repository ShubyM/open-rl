#!/usr/bin/env python3
"""Script to extract and format live benchmark metrics from open-rl-gateway."""

import subprocess
import sys


def main() -> None:
  scenario = sys.argv[1] if len(sys.argv) > 1 else "fft_gsm8k_rl"
  metrics_path = f"/mnt/shared/open-rl/runs/fft-gsm8k-rl/open-rl-tmp/{scenario}/metrics.jsonl"
  py_script = (
    f"import json, os\n"
    f"path = '{metrics_path}'\n"
    f"if os.path.exists(path):\n"
    f"    rows = [json.loads(l) for l in open(path) if l.strip()]\n"
    f"    print('Step | Accuracy | Reward | Sampling | Train Step | Save Delta | Total Step Time')\n"
    f"    print('-' * 80)\n"
    f"    for r in rows:\n"
    f"        if 'env/all/correct' in r:\n"
    f"            step = r.get('progress/batch', '?')\n"
    f"            corr = r.get('env/all/correct', 0.0)\n"
    f"            rew = r.get('env/all/reward/total', 0.0)\n"
    f"            t_samp = r.get('time/sampling', 0.0)\n"
    f"            t_train = r.get('time/train_step', 0.0)\n"
    f"            t_save = r.get('time/save_checkpoint', 0.0)\n"
    f"            t_total = r.get('time/total', 0.0)\n"
    f"            s1 = f'{{str(step):>4}} | {{corr:>7.2%}}  | {{rew:>6.4f}} | {{t_samp:>7.1f}}s'\n"
    f"            s2 = f'{{t_train:>9.1f}}s | {{t_save:>9.1f}}s | {{t_total:>14.1f}}s'\n"
    f"            print(f'{{s1}} | {{s2}}')\n"
  )
  cmd = [
    "kubectl",
    "exec",
    "deployment/open-rl-gateway",
    "--",
    "python3",
    "-c",
    py_script,
  ]
  res = subprocess.run(cmd, capture_output=True, text=True)
  if res.stdout:
    print(res.stdout, end="")
  if res.stderr:
    print(res.stderr, file=sys.stderr, end="")


if __name__ == "__main__":
  main()
