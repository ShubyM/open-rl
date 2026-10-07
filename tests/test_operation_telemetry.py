import asyncio
import os
import unittest
from unittest.mock import patch

from server.store import InMemoryStore
from server.telemetry import ops


class OperationTelemetryTest(unittest.TestCase):
  def test_sampler_ops_belong_to_the_run_and_land_in_its_turn(self) -> None:
    store = InMemoryStore()

    async def generate() -> dict:
      return {"sequences": []}

    async def run() -> None:
      with patch.dict(os.environ, {"OPEN_RL_TIME_SLICE_JOB_ID": "lora-m-0-sampler"}):
        async with ops.gpu_turn(None, None, store, "sampler", "m"):
          request = {"request_id": "r1", "lora_id": "tinker://run-a/sampler_weights/000003"}
          await ops.observe_operation(store, request, "sampler", None, generate)

    asyncio.run(run())
    samples = asyncio.run(store.read_samples(ops.key("run-a")))
    self.assertEqual([(s["operation"], s["run_id"]) for s in samples], [("sample", "run-a")])
    turns = asyncio.run(store.read_samples(ops.turn_key("lora-m-0-sampler")))
    self.assertEqual([op["name"] for op in turns[0]["ops"]], ["sample"])

  def test_no_turn_is_recorded_without_a_workload_name(self) -> None:
    store = InMemoryStore()

    async def run() -> None:
      with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPEN_RL_TIME_SLICE_JOB_ID", None)
        async with ops.gpu_turn(None, None, store, "trainer", "m"):
          pass

    asyncio.run(run())
    self.assertEqual(store.sample_store, {})


if __name__ == "__main__":
  unittest.main()
