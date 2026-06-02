import argparse
import asyncio
import json
import logging
import os
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from checkpoint import CheckpointRestorer, CudaCheckpointRestorer
from utils import timed

logger = logging.getLogger(__name__)


@dataclass
class Worker:
  pid: int
  connection_id: int | None
  checkpointed: bool = False
  failed: bool = False


class TrainerScheduler:
  def __init__(self, restorer: CheckpointRestorer):
    self.restorer = restorer
    self.workers: dict[str, Worker] = {}
    self.waiting_run_ids: deque[str] = deque()
    self.active_run_id: str | None = None
    self.condition = asyncio.Condition()

  def clear_run(self, run_id: str) -> None:
    if run_id in self.waiting_run_ids:
      self.waiting_run_ids.remove(run_id)
    if self.active_run_id == run_id:
      self.active_run_id = None

  async def register(self, run_id: str, pid: int, connection_id: int | None = None) -> dict[str, Any]:
    async with self.condition:
      worker = self.workers.get(run_id)

      if worker is not None:
        return {"ok": False, "error": f"run '{run_id}' is already registered"}

      self.workers[run_id] = Worker(pid=pid, connection_id=connection_id)
      self.condition.notify_all()
      return {"ok": True}

  async def acquire(self, run_id: str) -> dict[str, Any]:
    async with self.condition:
      worker = self.workers.get(run_id)
      if worker is None:
        return {"ok": False, "error": f"run '{run_id}' is not registered"}
      if worker.failed:
        return {"ok": False, "error": f"run '{run_id}' is failed"}
      if run_id in self.waiting_run_ids or self.active_run_id == run_id:
        return {"ok": False, "error": f"run '{run_id}' already has a pending or active acquire"}

      self.waiting_run_ids.append(run_id)
      try:
        while self.active_run_id is not None or (run_id in self.waiting_run_ids and self.waiting_run_ids[0] != run_id):
          await self.condition.wait()
      except BaseException:
        if run_id in self.waiting_run_ids:
          self.waiting_run_ids.remove(run_id)
        self.condition.notify_all()
        raise

      worker = self.workers.get(run_id)
      if worker is None or worker.failed or run_id not in self.waiting_run_ids:
        self.clear_run(run_id)
        self.condition.notify_all()
        return {"ok": False, "error": f"run '{run_id}' is not available"}

      self.waiting_run_ids.popleft()
      self.active_run_id = run_id
      if worker.checkpointed:
        await self.run_restore(worker.pid)
        worker.checkpointed = False

      self.condition.notify_all()
      return {"ok": True}

  async def release(self, run_id: str) -> dict[str, Any]:
    async with self.condition:
      worker = self.workers.get(run_id)
      if worker is None:
        return {"ok": False, "error": f"run '{run_id}' is not registered"}
      if self.active_run_id != run_id:
        return {"ok": False, "error": f"run '{run_id}' does not hold an active acquire"}

      await self.run_checkpoint(worker.pid)
      worker.checkpointed = True
      self.clear_run(run_id)
      self.condition.notify_all()
      return {"ok": True}

  async def unregister(self, run_id: str) -> dict[str, Any]:
    async with self.condition:
      if run_id not in self.workers:
        return {"ok": False, "error": f"run '{run_id}' is not registered"}

      self.clear_run(run_id)
      del self.workers[run_id]
      self.condition.notify_all()
      return {"ok": True}

  async def connection_closed(self, connection_id: int) -> None:
    async with self.condition:
      for run_id, worker in self.workers.items():
        if worker.connection_id != connection_id:
          continue
        self.clear_run(run_id)
        worker.failed = True
        worker.checkpointed = False
        worker.connection_id = None
      self.condition.notify_all()

  @timed
  async def run_checkpoint(self, pid: int) -> None:
    logger.info("checkpoint pid=%s", pid)
    try:
      await asyncio.to_thread(self.restorer.checkpoint, pid)
    except Exception:
      logger.critical("checkpoint failed for pid %s; GPU state is unknown, killing scheduler", pid, exc_info=True)
      os._exit(1)

  @timed
  async def run_restore(self, pid: int) -> None:
    logger.info("restore pid=%s", pid)
    try:
      await asyncio.to_thread(self.restorer.restore, pid)
    except Exception:
      logger.critical("restore failed for pid %s; GPU state is unknown, killing scheduler", pid, exc_info=True)
      os._exit(1)


async def serve_scheduler(scheduler: TrainerScheduler, socket_path: str) -> asyncio.Server:
  socket = Path(socket_path)
  socket.parent.mkdir(parents=True, exist_ok=True)
  socket.unlink(missing_ok=True)

  async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    connection_id = id(writer)
    try:
      while line := await reader.readline():
        response = await dispatch(scheduler, line, connection_id)
        writer.write(json.dumps(response).encode("utf-8") + b"\n")
        await writer.drain()
    finally:
      await scheduler.connection_closed(connection_id)
      writer.close()
      await writer.wait_closed()

  return await asyncio.start_unix_server(handle_connection, path=socket_path)


async def dispatch(scheduler: TrainerScheduler, line: bytes, connection_id: int) -> dict[str, Any]:
  payload = json.loads(line.decode("utf-8"))

  command = payload.get("command", "").upper()
  run_id = payload.get("run_id")

  assert run_id is not None, "run_id is required"

  match command:
    case "REGISTER":
      return await scheduler.register(run_id, payload["pid"], connection_id=connection_id)
    case "ACQUIRE":
      return await scheduler.acquire(run_id)
    case "RELEASE":
      return await scheduler.release(run_id)
    case "UNREGISTER":
      return await scheduler.unregister(run_id)
    case _:
      return {"ok": False, "error": f"unknown command '{command}'"}


class TrainerSchedulerClient:
  def __init__(self, socket_path: str):
    self.socket_path = socket_path
    self.reader: asyncio.StreamReader | None = None
    self.writer: asyncio.StreamWriter | None = None

  async def connect(self) -> None:
    if self.writer is not None and not self.writer.is_closing():
      return
    self.reader, self.writer = await asyncio.open_unix_connection(self.socket_path)

  async def close(self) -> None:
    if self.writer is None:
      return
    self.writer.close()
    await self.writer.wait_closed()
    self.reader = None
    self.writer = None

  async def register(self, run_id: str, pid: int) -> dict[str, Any]:
    return await self.request({"command": "REGISTER", "run_id": run_id, "pid": pid})

  async def unregister(self, run_id: str) -> dict[str, Any]:
    return await self.request({"command": "UNREGISTER", "run_id": run_id})

  @asynccontextmanager
  async def acquire(self, run_id: str) -> AsyncIterator[None]:
    await self.request({"command": "ACQUIRE", "run_id": run_id})
    try:
      yield
    finally:
      await self.request({"command": "RELEASE", "run_id": run_id})

  async def request(self, payload: dict[str, Any]) -> dict[str, Any]:
    await self.connect()
    assert self.reader is not None
    assert self.writer is not None

    self.writer.write(json.dumps(payload).encode("utf-8") + b"\n")
    await self.writer.drain()
    line = await self.reader.readline()
    if not line:
      raise RuntimeError("trainer scheduler connection closed")

    response = json.loads(line.decode("utf-8"))
    if not response.get("ok"):
      raise RuntimeError(response.get("error", "trainer scheduler command failed"))
    return response


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run the OpenRL trainer scheduler.")
  parser.add_argument("--socket", default=os.getenv("OPEN_RL_SCHEDULER_SOCKET", "/tmp/open-rl/trainer-scheduler.sock"))
  parser.add_argument("--cuda-checkpoint-bin", default=os.getenv("CUDA_CHECKPOINT_BIN", "cuda-checkpoint"))
  parser.add_argument("--cuda-checkpoint-timeout-ms", type=int, default=None)
  return parser.parse_args()


async def main_async() -> None:
  args = parse_args()
  restorer = CudaCheckpointRestorer(args.cuda_checkpoint_bin, args.cuda_checkpoint_timeout_ms)
  scheduler = TrainerScheduler(restorer)
  server = await serve_scheduler(scheduler, args.socket)
  logger.info("listening on %s", args.socket)
  async with server:
    await server.serve_forever()


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="[SCHEDULER] %(levelname)s %(message)s")
  asyncio.run(main_async())


if __name__ == "__main__":
  main()
