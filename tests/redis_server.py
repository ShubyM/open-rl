"""A throwaway redis-server for tests that need real Redis semantics (streams,
scripts, expiry). Uses OPEN_RL_TEST_REDIS_URL when set."""

import os
import shutil
import socket
import subprocess
import time
import unittest

TEST_REDIS_URL = os.getenv("OPEN_RL_TEST_REDIS_URL")
needs_redis = unittest.skipUnless(TEST_REDIS_URL or shutil.which("redis-server"), "needs OPEN_RL_TEST_REDIS_URL or redis-server on PATH")


class RedisServer:
  def __init__(self) -> None:
    self.process: subprocess.Popen | None = None
    if TEST_REDIS_URL:
      self.url = TEST_REDIS_URL
      return
    with socket.socket() as sock:
      sock.bind(("127.0.0.1", 0))
      port = sock.getsockname()[1]
    self.url = f"redis://127.0.0.1:{port}"
    self.process = subprocess.Popen(["redis-server", "--port", str(port), "--save", ""], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 10
    while True:
      try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
          return
      except OSError:
        if time.monotonic() > deadline:
          raise RuntimeError("redis-server did not come up") from None
        time.sleep(0.05)

  def stop(self) -> None:
    if self.process is not None:
      self.process.terminate()
      self.process.wait(timeout=10)
