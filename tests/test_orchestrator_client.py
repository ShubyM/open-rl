import os
import unittest
from unittest.mock import patch

from server import training_requests_processor
from snapshot_agent.client import SnapshotAgentClient
from snapshot_agent.orchestrator import accelerator_orchestrator_pb2 as pb2
from snapshot_agent.orchestrator_client import OrchestratorSnapshotClient


class _FakeStub:
  def __init__(self, acquire_response: pb2.AcquireResponse | None = None, yield_error: Exception | None = None):
    self.acquire_requests: list[pb2.AcquireRequest] = []
    self.yield_requests: list[pb2.YieldRequest] = []
    self.acquire_response = acquire_response or pb2.AcquireResponse(success=True, waited_ms=1500, context_restored=True)
    self.yield_error = yield_error

  async def Acquire(self, request: pb2.AcquireRequest) -> pb2.AcquireResponse:
    self.acquire_requests.append(request)
    return self.acquire_response

  async def Yield(self, request: pb2.YieldRequest) -> pb2.YieldResponse:
    self.yield_requests.append(request)
    if self.yield_error is not None:
      raise self.yield_error
    return pb2.YieldResponse(success=True)


def _client_with_stub(stub: _FakeStub) -> OrchestratorSnapshotClient:
  client = OrchestratorSnapshotClient("orchestrator:50051", job_id="job-a", group_id="trainers")
  client._stub = stub
  return client


class OrchestratorSnapshotClientTest(unittest.IsolatedAsyncioTestCase):
  async def test_acquire_sends_acquire_then_yield_with_job_and_group(self) -> None:
    stub = _FakeStub()
    client = _client_with_stub(stub)

    async with client.acquire(pid=1234):
      self.assertEqual(len(stub.acquire_requests), 1)
      self.assertEqual(stub.acquire_requests[0].job_id, "job-a")
      self.assertEqual(stub.acquire_requests[0].group_id, "trainers")
      self.assertEqual(stub.yield_requests, [])

    self.assertEqual(len(stub.yield_requests), 1)
    self.assertEqual(stub.yield_requests[0].job_id, "job-a")
    self.assertEqual(stub.yield_requests[0].group_id, "trainers")

  async def test_denied_acquire_raises_and_does_not_yield(self) -> None:
    stub = _FakeStub(acquire_response=pb2.AcquireResponse(success=False))
    client = _client_with_stub(stub)

    with self.assertRaisesRegex(RuntimeError, "denied Acquire"):
      async with client.acquire(pid=1234):
        self.fail("body must not run when Acquire is denied")

    self.assertEqual(stub.yield_requests, [])

  async def test_yield_failure_is_swallowed(self) -> None:
    stub = _FakeStub(yield_error=RuntimeError("orchestrator went away"))
    client = _client_with_stub(stub)

    with patch("snapshot_agent.orchestrator_client.traceback.print_exc"):
      async with client.acquire(pid=1234):
        pass

    self.assertEqual(len(stub.yield_requests), 1)

  async def test_body_exception_still_yields_and_propagates(self) -> None:
    stub = _FakeStub()
    client = _client_with_stub(stub)

    with self.assertRaisesRegex(ValueError, "boom"):
      async with client.acquire(pid=1234):
        raise ValueError("boom")

    self.assertEqual(len(stub.yield_requests), 1)

  async def test_register_and_unregister_are_noops(self) -> None:
    client = OrchestratorSnapshotClient("orchestrator:50051", job_id="job-a", group_id="trainers")

    self.assertEqual(await client.register(1234), {"ok": True})
    self.assertEqual(await client.unregister(1234), {"ok": True})
    await client.close()


class CreateSnapshotClientTest(unittest.TestCase):
  def test_orchestrator_addr_selects_orchestrator_client(self) -> None:
    env = {"OPEN_RL_TIME_SLICE_ORCHESTRATOR_ADDR": "orchestrator:50051"}
    with patch.dict(os.environ, env, clear=True):
      client = training_requests_processor.create_snapshot_client("model-a")

    self.assertIsInstance(client, OrchestratorSnapshotClient)
    self.assertEqual(client.job_id, "model-a")
    self.assertEqual(client.group_id, "trainers")

  def test_job_id_env_overrides_model_id(self) -> None:
    env = {
      "OPEN_RL_TIME_SLICE_ORCHESTRATOR_ADDR": "orchestrator:50051",
      "OPEN_RL_TIME_SLICE_JOB_ID": "sanitized-job",
      "OPEN_RL_TIME_SLICE_GROUP": "group-b",
    }
    with patch.dict(os.environ, env, clear=True):
      client = training_requests_processor.create_snapshot_client("Model_A")

    self.assertIsInstance(client, OrchestratorSnapshotClient)
    self.assertEqual(client.job_id, "sanitized-job")
    self.assertEqual(client.group_id, "group-b")

  def test_orchestrator_addr_without_job_id_raises(self) -> None:
    env = {"OPEN_RL_TIME_SLICE_ORCHESTRATOR_ADDR": "orchestrator:50051"}
    with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(RuntimeError, "job id"):
      training_requests_processor.create_snapshot_client(None)

  def test_no_orchestrator_addr_selects_unix_socket_client(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      client = training_requests_processor.create_snapshot_client("model-a")

    self.assertIsInstance(client, SnapshotAgentClient)
    self.assertEqual(client.socket_path, "/tmp/open-rl/snapshot-agent.sock")


if __name__ == "__main__":
  unittest.main()
