"""End-to-end: pig-latin SFT against Qwen3-0.6B.

The example runs as a process in the examples project's own environment and
writes its metrics to a file this test asks for. Nothing from examples/ is
imported here, so the root suite needs no path tricks to find it.

Non-gated model kept small for CI speed. Skips the baseline eval and runs
5 steps, enough to see the loss drop hard and exact match clear zero.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests._server_fixture import REPO_ROOT, OpenRlServerCase

EXAMPLE = REPO_ROOT / "examples" / "sft" / "pig-latin" / "piglatin_sft.py"

# Slow and needs a model download, so it runs when asked, not on every `uv run pytest`.
run_e2e = unittest.skipUnless(os.getenv("OPEN_RL_RUN_E2E"), "set OPEN_RL_RUN_E2E=1 to run the pig-latin end-to-end test")


def run_piglatin(preset: str, **overrides: object) -> dict[str, float]:
  """Run the example in the examples project and return the metrics it wrote."""
  with tempfile.TemporaryDirectory() as tmp:
    metrics_path = Path(tmp) / "metrics.json"
    plot_path = Path(tmp) / "metrics.png"
    command = [
      "uv",
      "--project",
      "examples",
      "run",
      "--frozen",
      "python",
      str(EXAMPLE),
      preset,
      f"metrics_path={metrics_path}",
      f"plot_path={plot_path}",
      *(f"{key}={value}" for key, value in overrides.items()),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return json.loads(metrics_path.read_text())


@run_e2e
class TestPigLatinQwen(OpenRlServerCase):
  BASE_MODEL = "Qwen/Qwen3-0.6B"
  PORT = 9010

  def test_sft_improves(self) -> None:
    metrics = run_piglatin("qwen", base_url=self.BASE_URL, steps=5, assert_improvement=False, skip_before_eval=True)
    print(f"Training metrics: {metrics}")

    self.assertGreater(metrics["loss_drop"], 0.0, f"Loss did not drop: {metrics['loss_drop']:.3f}")
    self.assertGreater(metrics["after_exact"], 0.0, f"After training, exact match still 0: {metrics['after_exact']:.3f}")


if __name__ == "__main__":
  unittest.main()
