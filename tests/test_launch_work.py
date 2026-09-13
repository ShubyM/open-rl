"""Exercise launcher topology with fake host tools; no services are started."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class LaunchWorkTest(unittest.TestCase):
  def setUp(self) -> None:
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.root = Path(self.directory.name)
    self.repo = self.root / "repo"
    (self.repo / "scripts").mkdir(parents=True)
    (self.repo / "examples/harvey_labs/harvey-labs").mkdir(parents=True)
    self.script = self.repo / "scripts/launch_work.sh"
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/launch_work.sh", self.script)
    self.bin = self.root / "bin"
    self.bin.mkdir()
    self.log = self.root / "tmux.log"
    self.tool("git", 'if [[ "$*" == *rev-parse* ]]; then echo "$TEST_REPO"; fi')
    self.tool("tmux", '[ "$1" != has-session ] || exit 1\nprintf "%s\\n" "$*" >> "$TEST_TMUX_LOG"')
    self.tool("podman", '[ "$1" != info ] || echo "$TEST_REPO"')
    self.tool("df", 'echo "Avail"; echo "1000G"')
    self.tool("pgrep", "exit 0")
    self.tool("nvidia-smi", 'if [ "$1" = -L ]; then for i in {0..7}; do echo "GPU $i: H200"; done; else echo H200; fi')
    self.interpreter = self.tool("automodel-python", "exit 0")
    self.env = {
      "PATH": f"{self.bin}:{os.environ['PATH']}",
      "HOME": os.environ["HOME"],
      "CUDA_HOME": str(self.root),
      "TEST_REPO": str(self.repo),
      "TEST_TMUX_LOG": str(self.log),
      "OPEN_RL_SNAPSHOT_DIR": str(self.root / "snapshots"),
      "OPEN_RL_CHECKPOINT_DIR": str(self.root / "checkpoints"),
      "MODEL": "e4b",
      "TRAINER_BACKEND": "automodel",
      "AUTOMODEL_PYTHON": str(self.interpreter),
      "AFFINITY": "1",
    }

  def tool(self, name: str, body: str) -> Path:
    path = self.bin / name
    path.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path

  def launch(self, **settings: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(self.script)], env={**self.env, **settings}, capture_output=True, text=True, timeout=10)

  def test_one_gpu_automodel_gets_its_own_interpreter_and_redis_queue(self) -> None:
    result = self.launch(TRAIN_GPUS="1")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    commands = self.log.read_text().splitlines()
    trainer = next(line for line in commands if line.startswith("send-keys -t work:trainer "))
    gateway = next(line for line in commands if line.startswith("send-keys -t work:gateway "))
    self.assertIn(f"{self.interpreter} -m torch.distributed.run", trainer)
    self.assertIn("--nproc-per-node=1", trainer)
    self.assertIn("REDIS_URL=redis://127.0.0.1:6379", trainer)
    self.assertIn("OPEN_RL_EXTERNAL_TRAINER=1", gateway)
    self.assertIn("CUDA_VISIBLE_DEVICES= ", gateway)

  def test_single_gpu_fsdp_keeps_in_gateway_training(self) -> None:
    result = self.launch(TRAIN_GPUS="1", TRAINER_BACKEND="fsdp")
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    commands = self.log.read_text()
    self.assertNotIn("send-keys -t work:trainer ", commands)
    self.assertNotIn("OPEN_RL_EXTERNAL_TRAINER=1", commands)

  def test_invalid_parallelism_fails_before_creating_tmux_session(self) -> None:
    for settings in ({"AUTOMODEL_CP": "0"}, {"AUTOMODEL_TP": "-1"}, {"AUTOMODEL_CP": "3"}):
      with self.subTest(settings=settings):
        result = self.launch(TRAIN_GPUS="4", **settings)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR:", result.stderr)
        self.assertFalse(self.log.exists())

  def test_external_samplers_require_a_supported_adapter_rank(self) -> None:
    for rank in ("0", "-1", "128", "bad"):
      with self.subTest(rank=rank):
        result = self.launch(AUTOMODEL_LORA_RANK=rank)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AUTOMODEL_LORA_RANK must be between 1 and 64", result.stderr)
        self.assertFalse(self.log.exists())

  def test_sampler_must_have_at_least_one_gpu(self) -> None:
    result = self.launch(TRAIN_GPUS="8")
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("smaller than the available GPU count", result.stderr)
