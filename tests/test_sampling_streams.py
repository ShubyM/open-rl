"""The stream rules from the design, checked against a real Redis: acceptance
limits, one lease holder, guarded completion, claim renewal and recovery."""

import asyncio
import unittest

import redis.asyncio as redis

from server import sampling_streams as streams
from server.sampling_streams import SamplingStreams, SetClosed, SetFull
from tests.redis_server import RedisServer, needs_redis

SET = "set-test"
REQUEST = {"prompt_token_ids": [1, 2, 3], "max_tokens": 4, "num_samples": 1, "lora_id": None, "model_id": "m"}


@needs_redis
class SamplingStreamsTest(unittest.IsolatedAsyncioTestCase):
  server: RedisServer

  @classmethod
  def setUpClass(cls) -> None:
    cls.server = RedisServer()

  @classmethod
  def tearDownClass(cls) -> None:
    cls.server.stop()

  async def asyncSetUp(self) -> None:
    self.client = redis.from_url(self.server.url)
    await self.client.flushdb()
    self.streams = SamplingStreams(self.client, result_ttl=60, guard_ttl=120)
    await self.streams.open(SET)

  async def asyncTearDown(self) -> None:
    await self.client.aclose()

  async def accept(self, limit: int = 10) -> str:
    return await self.streams.accept(SET, REQUEST, limit=limit, deadline_seconds=60)

  async def claim_one(self, owner: str = "d1") -> streams.Claim:
    await self.streams.ensure_group(SET)
    self.assertTrue(await self.streams.lease(SET, owner, 30))
    [claim] = await self.streams.read_new(SET, owner, 1, 100)
    return claim

  async def test_accepted_requests_are_pending_until_finalized(self) -> None:
    request_id = await self.accept()
    self.assertEqual(streams.set_of(request_id), SET)
    self.assertEqual(await self.streams.result(request_id), {"type": "try_again"})
    self.assertEqual(await self.streams.unfinished(SET), 1)

  async def test_a_closed_or_full_set_refuses_before_returning_an_id(self) -> None:
    await self.accept(limit=1)
    with self.assertRaises(SetFull):
      await self.accept(limit=1)
    await self.streams.close(SET)
    with self.assertRaises(SetClosed):
      await self.accept()
    self.assertEqual(await self.streams.unfinished(SET), 1)

  async def test_requests_accepted_before_the_dispatcher_starts_are_read(self) -> None:
    first, second = await self.accept(), await self.accept()
    await self.streams.ensure_group(SET)
    read = await self.streams.read_new(SET, "d1", 10, 100)
    self.assertEqual([c.request_id for c in read], [first, second])
    self.assertEqual(read[0].input["prompt_token_ids"], [1, 2, 3])

  async def test_one_lease_holder_and_no_renewal_after_takeover(self) -> None:
    self.assertTrue(await self.streams.lease(SET, "d1", 0.2))
    self.assertFalse(await self.streams.lease(SET, "d2", 0.2))
    await asyncio.sleep(0.3)
    self.assertTrue(await self.streams.lease(SET, "d2", 30))
    self.assertFalse(await self.streams.lease(SET, "d1", 30))

  async def test_only_the_current_attempt_saves_the_one_final_result(self) -> None:
    request_id = await self.accept()
    await self.claim_one()
    self.assertEqual(await self.streams.begin(SET, request_id, "d1", "a1", 3), ("ok", 1))
    self.assertEqual(await self.streams.begin(SET, request_id, "d1", "a2", 3), ("ok", 2))
    late = await self.streams.finalize(SET, request_id, owner="d1", attempt_id="a1", outcome="success", result={"type": "sample", "n": 1})
    self.assertEqual(late, "stale")
    done = await self.streams.finalize(SET, request_id, owner="d1", attempt_id="a2", outcome="success", result={"type": "sample", "n": 2})
    self.assertEqual(done, "ok")
    again = await self.streams.finalize(SET, request_id, owner="d1", attempt_id="a2", outcome="failure", result=streams.failed("no"))
    self.assertEqual(again, "final")
    self.assertEqual(await self.streams.result(request_id), {"type": "sample", "n": 2})
    self.assertEqual(await self.streams.unfinished(SET), 0)
    self.assertEqual(await self.client.xlen(streams.keys(SET)["stream"]), 0)
    self.assertEqual((await self.client.xpending(streams.keys(SET)["stream"], streams.GROUP))["pending"], 0)

  async def test_an_old_lease_holder_cannot_start_or_publish(self) -> None:
    request_id = await self.accept()
    await self.claim_one()
    self.assertEqual((await self.streams.begin(SET, request_id, "d1", "a1", 3))[0], "ok")
    await self.client.delete(streams.keys(SET)["lease"])
    self.assertTrue(await self.streams.lease(SET, "d2", 30))
    self.assertEqual((await self.streams.begin(SET, request_id, "d1", "a2", 3))[0], "lease")
    published = await self.streams.finalize(SET, request_id, owner="d1", attempt_id="a1", outcome="success", result={"type": "sample"})
    self.assertEqual(published, "lease")
    self.assertEqual(await self.streams.result(request_id), {"type": "try_again"})

  async def test_attempts_are_bounded(self) -> None:
    request_id = await self.accept()
    await self.claim_one()
    for n in (1, 2):
      self.assertEqual(await self.streams.begin(SET, request_id, "d1", f"a{n}", 2), ("ok", n))
    self.assertEqual((await self.streams.begin(SET, request_id, "d1", "a3", 2))[0], "exhausted")

  async def test_cancellation_wins_over_a_later_success(self) -> None:
    request_id = await self.accept()
    await self.claim_one()
    await self.streams.begin(SET, request_id, "d1", "a1", 3)
    await self.streams.close(SET)
    self.assertEqual(await self.streams.cancel_unfinished(SET, "set removed"), 1)
    late = await self.streams.finalize(SET, request_id, owner="d1", attempt_id="a1", outcome="success", result={"type": "sample"})
    self.assertEqual(late, "final")
    result = await self.streams.result(request_id)
    self.assertEqual((result["type"], result["error_message"]), ("RequestFailedResponse", "set removed"))
    self.assertEqual(await self.streams.unfinished(SET), 0)

  async def test_unrenewed_claims_are_recovered_and_renewed_ones_are_not(self) -> None:
    kept, dropped = await self.accept(), await self.accept()
    await self.streams.ensure_group(SET)
    self.assertTrue(await self.streams.lease(SET, "d1", 30))
    read = {c.request_id: c for c in await self.streams.read_new(SET, "d1", 2, 100)}
    await asyncio.sleep(0.3)
    self.assertEqual(await self.streams.renew_claims(SET, "d1", [read[kept].stream_id]), 1)
    recovered = await self.streams.reclaim(SET, "d2", 10, min_idle=0.2)
    self.assertEqual([c.request_id for c in recovered], [dropped])
    # d2 now owns the dropped entry, so d1 cannot renew it.
    self.assertEqual(await self.streams.renew_claims(SET, "d1", [read[dropped].stream_id]), 0)

  async def test_claims_are_not_renewed_without_the_lease(self) -> None:
    await self.accept()
    claim = await self.claim_one()
    await self.client.delete(streams.keys(SET)["lease"])
    self.assertTrue(await self.streams.lease(SET, "d2", 30))
    self.assertEqual(await self.streams.renew_claims(SET, "d1", [claim.stream_id]), -1)

  async def test_expired_results_never_read_as_pending(self) -> None:
    request_id = await self.accept()
    await self.claim_one()
    await self.streams.begin(SET, request_id, "d1", "a1", 3)
    await self.streams.finalize(SET, request_id, owner="d1", attempt_id="a1", outcome="success", result={"type": "sample"})
    await self.client.delete(streams.result_key(SET, request_id))
    self.assertIn("expired", (await self.streams.result(request_id))["error_message"])
    unknown = await self.streams.result(streams.new_request_id(SET))
    self.assertIn("not found", unknown["error_message"])


if __name__ == "__main__":
  unittest.main()
