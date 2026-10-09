"""Tool calls must not starve the event loop's default thread pool.

Run from the repo root: uv --project examples run python -m unittest harvey_labs.test_agent_sandbox
(with examples/ on PYTHONPATH).
"""

import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor

from harvey_labs.agent_sandbox import AgentLabSandbox, RemoteFiles


class BlockingExecutor:
  """Like LAB's ToolExecutor: a synchronous call that waits on a sandbox request
  run on the event loop, which itself moves bytes with asyncio.to_thread, as the
  Agent Sandbox client's file transfers do."""

  def __init__(self, files: RemoteFiles, release: asyncio.Event) -> None:
    self.files = files
    self.release = release

  def execute(self, name: str, arguments: dict) -> str:
    async def transfer() -> str:
      # The tool's thread stays blocked until every call is in flight; then the
      # transfer needs a default-pool thread.
      await self.release.wait()
      return await asyncio.to_thread(lambda: f"{name} done")

    return self.files.call(transfer())


def sandbox(loop: asyncio.AbstractEventLoop, release: asyncio.Event) -> AgentLabSandbox:
  box = AgentLabSandbox.__new__(AgentLabSandbox)
  box.loop = loop
  box.executor = BlockingExecutor(RemoteFiles.__new__(RemoteFiles), release)
  box.executor.files.sandbox = box
  return box


class ToolThreadsTest(unittest.IsolatedAsyncioTestCase):
  async def test_more_tool_calls_than_default_threads_still_finish(self) -> None:
    loop = asyncio.get_running_loop()
    # Fewer default threads than concurrent tool calls, like 48 episodes on a 32-thread pool.
    loop.set_default_executor(ThreadPoolExecutor(max_workers=4))
    release = asyncio.Event()
    calls = [asyncio.create_task(sandbox(loop, release).execute_tool("bash", {})) for _ in range(16)]
    await asyncio.sleep(0.2)
    loop.call_soon(release.set)
    self.assertEqual(await asyncio.wait_for(asyncio.gather(*calls), timeout=10), ["bash done"] * 16)

  async def test_the_default_pool_stays_free_while_tools_block(self) -> None:
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=4))
    release = asyncio.Event()
    calls = [asyncio.create_task(sandbox(loop, release).execute_tool("bash", {})) for _ in range(16)]
    await asyncio.sleep(0.2)
    # A claim renewal's thread work gets a thread at once, not after the tools.
    self.assertEqual(await asyncio.wait_for(asyncio.to_thread(lambda: "renewed"), timeout=2), "renewed")
    release.set()
    await asyncio.wait_for(asyncio.gather(*calls), timeout=10)


if __name__ == "__main__":
  unittest.main()
