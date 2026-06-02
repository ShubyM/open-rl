import os
import subprocess
import sys
from pathlib import Path


def launch_worker(run_id: str, scheduler_socket: str) -> subprocess.Popen:
  if not (os.getenv("REDIS_URL") or os.getenv("OPEN_RL_STORE_DIR")):
    raise RuntimeError("a shared store (REDIS_URL or OPEN_RL_STORE_DIR) is required when launching trainer worker subprocesses")

  env = os.environ.copy()
  env["OPEN_RL_SCHEDULER_SOCKET"] = scheduler_socket
  env["OPEN_RL_WORKER_RUN_ID"] = run_id
  env["OPEN_RL_DISABLE_WORKER_HEALTHZ"] = "1"
  env.setdefault("PYTHONUNBUFFERED", "1")

  server_dir = Path(__file__).resolve().parent
  process = subprocess.Popen(
    [sys.executable, str(server_dir / "clock_cycle.py")],
    cwd=server_dir,
    env=env,
    start_new_session=True,
  )
  return process
