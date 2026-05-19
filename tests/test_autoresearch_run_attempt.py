import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import tomli

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))
sys.modules.setdefault("tomllib", tomli)
sys.modules.setdefault(
  "chz",
  types.SimpleNamespace(chz=lambda cls: cls, entrypoint=lambda *_args, **_kwargs: None),
)

from rl.autoresearch import run_attempt  # noqa: E402


def alive(pid: int) -> bool:
  try:
    os.kill(pid, 0)
    return True
  except ProcessLookupError:
    return False


class TestRunAttempt(unittest.TestCase):
  def test_agent_log_seed_advances_between_attempts(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      source = root / "researcher" / "agent.log"
      work_dir = root / "researcher"
      source.parent.mkdir()
      source.write_text("setup\n", encoding="utf-8")
      args = types.SimpleNamespace(log_root=root / "runs")

      with patch.dict(os.environ, {"RESEARCHER_LOG_PATH": str(source), "WORK_DIR": str(work_dir)}):
        first = run_attempt.AttemptRun(args, types.SimpleNamespace(), "researcher-a", {}, 1, root / "runs" / "attempt-1")
        first.run_dir.mkdir(parents=True)
        self.assertTrue(first.seed_agent_log())
        self.assertEqual((first.run_dir / "agent.log").read_text(encoding="utf-8"), "setup\n")
        first.mark_agent_boundary()

        with source.open("a", encoding="utf-8") as f:
          f.write("agent writes attempt 2\n")
        second = run_attempt.AttemptRun(args, types.SimpleNamespace(), "researcher-a", {}, 2, root / "runs" / "attempt-2")
        second.run_dir.mkdir(parents=True)
        self.assertTrue(second.seed_agent_log())
        self.assertEqual((second.run_dir / "agent.log").read_text(encoding="utf-8"), "agent writes attempt 2\n")
        second.mark_agent_boundary()

        with source.open("a", encoding="utf-8") as f:
          f.write("agent writes attempt 3\n")
        third = run_attempt.AttemptRun(args, types.SimpleNamespace(), "researcher-a", {}, 3, root / "runs" / "attempt-3")
        third.run_dir.mkdir(parents=True)
        self.assertTrue(third.seed_agent_log())
        self.assertEqual((third.run_dir / "agent.log").read_text(encoding="utf-8"), "agent writes attempt 3\n")

  def test_run_logged_cleans_background_children(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      pid_path = root / "child.pid"
      code = run_attempt.run_logged(["bash", "-c", f"sleep 60 >/dev/null 2>&1 & echo $! > {pid_path}"], root / "run.log")

      self.assertEqual(code, 0)
      pid = int(pid_path.read_text(encoding="utf-8"))
      deadline = time.time() + 3
      while alive(pid) and time.time() < deadline:
        time.sleep(0.05)
      self.assertFalse(alive(pid))

  def test_run_logged_does_not_wait_for_background_stdout_holder(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      pid_path = root / "child.pid"
      start = time.time()
      code = run_attempt.run_logged(["bash", "-c", f"sleep 60 & echo $! > {pid_path}; echo done"], root / "run.log")

      self.assertEqual(code, 0)
      self.assertLess(time.time() - start, 3)
      pid = int(pid_path.read_text(encoding="utf-8"))
      deadline = time.time() + 3
      while alive(pid) and time.time() < deadline:
        time.sleep(0.05)
      self.assertFalse(alive(pid))


if __name__ == "__main__":
  unittest.main()
