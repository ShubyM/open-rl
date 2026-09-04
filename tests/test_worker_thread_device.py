"""Every executor thread must hold this rank's CUDA device.

Torch's current device is thread-local and every worker call is handed off with
asyncio.to_thread, so the device has to be established per thread. It was not:
set_device ran once, inside create_model, which left exactly the one pool thread
that served that request pointing at the right GPU. Threads the pool spawned
later still pointed at cuda:0.

That is invisible on rank 0, where cuda:0 is the correct answer, and wrong on
every other rank -- so it survives any test run on a single GPU, and it survived
four steps of run33 before a burst grew the pool and rank 2 hit
`cuda:0 and cuda:2` mid-forward. These cases need no GPU: they patch set_device
and assert on which threads it reached.
"""

import asyncio
import importlib
import os
import sys
import threading
import types
import unittest
import unittest.mock
from unittest.mock import patch

# How many tasks are forced to run at once. A threading.Barrier makes the pool
# genuinely spawn this many threads rather than reusing one warm thread, which
# is the condition run33 needed and never met until step 4.
CONCURRENCY = 4
BARRIER_TIMEOUT = 30.0
RANK = 2


def load_processor_module():
  stubs = {
    "peft": types.SimpleNamespace(
      LoraConfig=object,
      PeftModelForCausalLM=object,
      get_peft_model=lambda *_args, **_kwargs: None,
    ),
    "transformers": types.SimpleNamespace(
      AutoConfig=object,
      AutoModelForCausalLM=object,
      AutoTokenizer=object,
      PreTrainedModel=object,
      PreTrainedTokenizerBase=object,
    ),
  }
  env = {"OPEN_RL_ENABLE_FFT": "true", "REDIS_URL": "redis://localhost:6379"}
  with patch.dict(sys.modules, stubs), patch.dict(os.environ, env):
    sys.modules.pop("server.training_requests_processor", None)
    return importlib.import_module("server.training_requests_processor")


processor = load_processor_module()


class WorkerThreadDeviceTest(unittest.TestCase):
  def setUp(self) -> None:
    # thread name -> device it was told to use. Recording the thread, not just
    # the argument, is the whole point: the old code called set_device with the
    # right device on the wrong thread.
    self.pinned: dict[str, int] = {}
    self.lock = threading.Lock()

    def record_set_device(device) -> None:
      index = device.index if hasattr(device, "index") else int(device)
      with self.lock:
        self.pinned[threading.current_thread().name] = index

    self.enterContext(unittest.mock.patch.object(processor.torch.cuda, "is_available", return_value=True))
    self.enterContext(unittest.mock.patch.object(processor.torch.cuda, "set_device", record_set_device))
    self.enterContext(unittest.mock.patch.object(processor, "local_rank", return_value=RANK))

  def burst(self) -> set[str]:
    """Run CONCURRENCY tasks that must overlap, and return the threads used."""
    barrier = threading.Barrier(CONCURRENCY)
    used: set[str] = set()

    def note_thread() -> None:
      with self.lock:
        used.add(threading.current_thread().name)

    def task() -> None:
      # Nobody leaves until all CONCURRENCY tasks have arrived, so the pool
      # cannot satisfy this with a single reused thread.
      barrier.wait(timeout=BARRIER_TIMEOUT)
      note_thread()

    async def main() -> None:
      processor.pin_worker_threads_to_this_rank()
      # A sequential call first: this is the steady state that looked healthy
      # for four steps, and it must keep working.
      await asyncio.to_thread(note_thread)
      await asyncio.gather(*(asyncio.to_thread(task) for _ in range(CONCURRENCY)))

    asyncio.run(main())
    return used

  def test_every_thread_that_runs_work_holds_this_ranks_device(self) -> None:
    used = self.burst()

    self.assertGreaterEqual(len(used), CONCURRENCY, "the burst did not actually spread across threads")
    unpinned = sorted(name for name in used if name not in self.pinned)
    self.assertEqual(unpinned, [], f"threads ran worker code without a device: {unpinned}")
    wrong = sorted((name, self.pinned[name]) for name in used if self.pinned[name] != RANK)
    self.assertEqual(wrong, [], f"threads pinned to the wrong device: {wrong}")

  def test_the_calling_thread_is_pinned_too(self) -> None:
    # Not everything goes through to_thread; the loop thread itself touches
    # torch, so it cannot be left on whatever device it started with.
    async def main() -> None:
      processor.pin_worker_threads_to_this_rank()

    asyncio.run(main())
    self.assertEqual(self.pinned.get("MainThread"), RANK)

  def test_the_unpinned_default_executor_is_what_broke_run33(self) -> None:
    # The control. Same burst without the fix: the pool's threads never hear
    # about the device, so anything allocated on them lands on cuda:0.
    barrier = threading.Barrier(CONCURRENCY)
    used: set[str] = set()

    def task() -> None:
      barrier.wait(timeout=BARRIER_TIMEOUT)
      with self.lock:
        used.add(threading.current_thread().name)

    async def main() -> None:
      await asyncio.gather(*(asyncio.to_thread(task) for _ in range(CONCURRENCY)))

    asyncio.run(main())

    self.assertGreaterEqual(len(used), CONCURRENCY)
    self.assertEqual(sorted(name for name in used if name in self.pinned), [])

  def test_a_cpu_only_process_is_left_alone(self) -> None:
    # No CUDA means no device to pin and no reason to replace the executor;
    # the CPU test suite itself runs in this state.
    with unittest.mock.patch.object(processor.torch.cuda, "is_available", return_value=False):

      async def main() -> None:
        processor.pin_worker_threads_to_this_rank()
        return await asyncio.to_thread(lambda: threading.current_thread().name)

      name = asyncio.run(main())

    self.assertEqual(self.pinned, {})
    self.assertNotIn(name, self.pinned)


if __name__ == "__main__":
  unittest.main()
