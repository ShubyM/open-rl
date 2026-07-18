"""Small recipe contract against an OpenRL cluster that is already running."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.distributed

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tiny_rl_against_deployed_cluster(tmp_path: Path) -> None:
  base_url = os.getenv("OPENRL_BASE_URL")
  if not base_url:
    pytest.skip("Set OPENRL_BASE_URL to run the distributed contract")

  command = [
    "uv",
    "--project",
    "examples",
    "run",
    "--no-sync",
    "python",
    "examples/tiny/tiny_rl.py",
    f"base_url={base_url}",
    f"base_model={os.getenv('OPENRL_TEST_BASE_MODEL', 'Qwen/Qwen2.5-0.5B')}",
    f"log_dir={tmp_path}",
    "steps=1",
    "samples_per_prompt=2",
    "max_tokens=4",
    "save_final_state=false",
  ]
  subprocess.run(command, cwd=REPO_ROOT, check=True, timeout=900)
