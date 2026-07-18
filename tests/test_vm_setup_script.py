import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "dev" / "infra" / "setup_vm.sh"


class VmSetupScriptTest(unittest.TestCase):
  def test_shell_is_valid(self) -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

  def test_dry_run_builds_the_tmux_stack_without_mutating_the_machine(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      bin_dir = Path(directory)
      for name in ("curl", "redis-server", "uv"):
        tool = bin_dir / name
        tool.write_text("#!/bin/sh\nexit 0\n")
        tool.chmod(0o755)

      nvidia_smi = bin_dir / "nvidia-smi"
      nvidia_smi.write_text("#!/bin/sh\nprintf '0, Test GPU, 81920 MiB\\n'\n")
      nvidia_smi.chmod(0o755)

      redis_cli = bin_dir / "redis-cli"
      redis_cli.write_text("#!/bin/sh\nexit 1\n")
      redis_cli.chmod(0o755)

      tmux = bin_dir / "tmux"
      tmux.write_text('#!/bin/sh\n[ "$1" = has-session ] && exit 1\nexit 0\n')
      tmux.chmod(0o755)

      environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "ATTACH": "0",
        "BASE_MODEL": "Qwen/Test-0.5B",
        "CUDA_VISIBLE_DEVICES": "2,3",
        "OPEN_RL_SETUP_DRY_RUN": "1",
        "OPEN_RL_TMUX_SESSION": "openrl-test-dry-run",
        "SAMPLER_CUDA_VISIBLE_DEVICES": "4",
      }
      result = subprocess.run(
        [str(SCRIPT)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
      )

    output = result.stdout + result.stderr
    self.assertIn("trainer GPUs: 2,3", output)
    self.assertIn("sampler GPUs: 4", output)
    self.assertIn("time-slicer:  noop", output)
    self.assertIn("tmux new-session", output)
    self.assertIn("tmux new-window", output)
    self.assertIn("OPEN_RL_ENABLE_FFT=true", output)
    self.assertIn("os.replace", output)
    self.assertIn("26214400", output)
    self.assertIn("using the no-op time-slicer", output)


if __name__ == "__main__":
  unittest.main()
