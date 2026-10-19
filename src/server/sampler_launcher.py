"""Runs a pool sampler's vLLM for a replica that shares its GPU with its trainer.

  python -m server.sampler_launcher vllm serve <model> --port <port> ... --enable-sleep-mode

vLLM takes its GPU memory as it starts, so the launcher registers the replica's workload with
the node's time slicer and starts vLLM inside one turn, waits until it is healthy, puts it to
sleep, releases the turn and writes the awake marker the startup probe looks for. From then on
the pool's dispatcher leases the replica for each turn over its own connection. The launcher
keeps the registration, which lasts as long as its connection, and supervises vLLM. It forwards
SIGTERM and SIGINT and exits with vLLM's status. vLLM ignores a SIGTERM while its engine is
starting, so one it has not acted on within a few seconds is followed by SIGKILL, and the turn
goes with the launcher's connection. When the slicer reports on that connection that it could
not park vLLM at the end of a dispatcher's turn, the launcher stops vLLM and exits 1, so the
GPU frees, the grant moves on and the container restarts.
"""

import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path

import httpx

from accel_timeslicer.time_slicer import time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import SAMPLER_CLAIM

AWAKE_FILE = os.getenv("OPEN_RL_SAMPLER_AWAKE_FILE", "/tmp/open-rl-sampler-awake")
# A vLLM that is up exits well under a second after a SIGTERM; one still starting ignores it.
TERM_GRACE_SECONDS = 5.0


def log(message: str) -> None:
  print(f"[sampler_launcher] {message}", file=sys.stderr, flush=True)


def server_url(command: list[str]) -> str:
  port = command[command.index("--port") + 1] if "--port" in command else "8000"
  return f"http://127.0.0.1:{port}"


def exit_status(returncode: int) -> int:
  return returncode if returncode >= 0 else 128 - returncode


class Launcher:
  def __init__(self, command: list[str]) -> None:
    self.command, self.url = command, server_url(command)
    self.boot: asyncio.Task | None = None
    self.child: asyncio.subprocess.Process | None = None

  def forward(self, sig: signal.Signals) -> None:
    """Hands a signal to vLLM. One that comes before vLLM starts, perhaps while the acquire waits behind
    the trainer, ends the boot instead, and the turn goes with the connection when this process exits."""
    if self.child is None:
      self.boot.cancel()
    elif self.child.returncode is None:
      with contextlib.suppress(ProcessLookupError):
        self.child.send_signal(sig)
      asyncio.get_running_loop().call_later(TERM_GRACE_SECONDS, self.kill)

  def kill(self) -> None:
    if self.child.returncode is None:
      log(f"vLLM has not exited {TERM_GRACE_SECONDS:g}s after the signal; killing it")
      with contextlib.suppress(ProcessLookupError):
        os.killpg(self.child.pid, signal.SIGKILL)

  async def stop(self, reason: str) -> int:
    log(f"{reason}; stopping vLLM")
    self.forward(signal.SIGTERM)
    await self.child.wait()
    return 1

  async def run(self) -> int:
    awake = Path(AWAKE_FILE)
    awake.unlink(missing_ok=True)
    self.boot = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
      loop.add_signal_handler(sig, self.forward, sig)
    slicer, workload = time_slicer_client_from_env(), workload_from_env(claim=SAMPLER_CLAIM)
    payload = workload.as_payload()
    try:
      await slicer.register(workload)
      await slicer.request({"command": "ACQUIRE", **payload})
      # vLLM gets a process group of its own, so the kill takes its engine processes with it.
      self.child = await asyncio.create_subprocess_exec(*self.command, start_new_session=True)
    except asyncio.CancelledError:
      log("stopped before vLLM started")
      return 1
    async with httpx.AsyncClient(base_url=self.url) as http:
      while not await self.healthy(http):
        if self.child.returncode is not None:
          log(f"vLLM exited with {self.child.returncode} before it was healthy")
          return exit_status(self.child.returncode)
        await asyncio.sleep(0.2)
      try:
        (await http.post("/sleep", params={"level": 1}, timeout=120)).raise_for_status()
      except httpx.HTTPError as exc:
        return await self.stop(f"vLLM did not sleep ({exc!r})")
    try:
      await slicer.request({"command": "RELEASE", **payload})
    except Exception as exc:
      return await self.stop(f"the time slicer could not park vLLM ({exc!r})")
    awake.touch()
    log("vLLM asleep and released; the pool's dispatcher takes its turns from here")
    # A dispatcher turn the slicer could not park leaves vLLM on the GPU, and the grant moves only once this process is
    # gone. The slicer says so on this connection, and then vLLM and the launcher stop.
    fault, exited = asyncio.create_task(slicer.fault()), asyncio.create_task(self.child.wait())
    await asyncio.wait({fault, exited}, return_when=asyncio.FIRST_COMPLETED)
    if fault.done() and fault.result():
      return await self.stop(f"the time slicer could not park vLLM after a turn ({fault.result()})")
    fault.cancel()
    return exit_status(await exited)

  async def healthy(self, http: httpx.AsyncClient) -> bool:
    try:
      return (await http.get("/health", timeout=2)).status_code == 200
    except httpx.HTTPError:
      return False


def main() -> None:
  sys.exit(asyncio.run(Launcher(sys.argv[1:]).run()))


if __name__ == "__main__":
  main()
