"""The stream rules from the design, against a real Redis: acceptance, one saved
result per request, recovery of entries no consumer is working on, and set removal."""

import asyncio
import unittest

import redis as sync_redis
import redis.asyncio as redis

from server import sampling_streams as streams
from server.sampling_streams import SamplingStreams, SetClosed, SetFull
from tests.redis_server import RedisServer, needs_redis

SET = "set-test"
REQUEST = {"prompt_token_ids": [1, 2, 3], "max_tokens": 4, "num_samples": 1, "lora_id": None, "model_id": "m"}
SAMPLE = {"type": "sample", "sequences": []}


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
    self.streams = SamplingStreams(self.client, result_ttl=60)
    await self.streams.open(SET)
    await self.streams.ensure_group(SET)

  async def asyncTearDown(self) -> None:
    await self.client.aclose()

  async def accept(self, limit: int = 10) -> str:
    return await self.streams.accept(SET, REQUEST, limit=limit, deadline_seconds=60)

  async def test_an_accepted_request_is_pending_until_its_result_is_saved(self) -> None:
    request_id = await self.accept()
    set_id, entry_id = streams.parse(request_id)
    self.assertEqual(set_id, SET)
    self.assertEqual(await self.streams.result(request_id), {"type": "try_again"})
    self.assertTrue(await self.streams.finish(SET, entry_id, SAMPLE))
    self.assertEqual(await self.streams.result(request_id), SAMPLE)
    self.assertEqual(await self.streams.unfinished(SET), 0)

  async def test_a_closed_or_full_set_refuses_before_returning_an_id(self) -> None:
    await self.accept(limit=1)
    with self.assertRaises(SetFull):
      await self.accept(limit=1)
    await self.streams.close(SET)
    with self.assertRaises(SetClosed):
      await self.accept()
    self.assertEqual(await self.streams.unfinished(SET), 1)

  async def test_requests_accepted_before_the_dispatcher_starts_are_read(self) -> None:
    await self.client.delete(streams.stream_key(SET))
    first, second = await self.accept(), await self.accept()
    await self.streams.ensure_group(SET)
    read = await self.streams.read_new(SET, "d1", 10, 100)
    self.assertEqual([c.request_id for c in read], [first, second])
    self.assertEqual(read[0].input["prompt_token_ids"], [1, 2, 3])

  async def test_only_the_first_result_is_saved(self) -> None:
    await self.accept()
    [claim] = await self.streams.read_new(SET, "d1", 1, 100)
    self.assertTrue(await self.streams.finish(SET, claim.entry_id, SAMPLE))
    self.assertFalse(await self.streams.finish(SET, claim.entry_id, streams.failed("late attempt")))
    self.assertEqual(await self.streams.result(claim.request_id), SAMPLE)
    self.assertEqual((await self.client.xpending(streams.stream_key(SET), streams.GROUP))["pending"], 0)

  async def test_untouched_entries_are_recovered_and_touched_ones_are_not(self) -> None:
    await self.accept()
    await self.accept()
    kept, dropped = await self.streams.read_new(SET, "d1", 2, 100)
    await asyncio.sleep(0.3)
    await self.streams.touch(SET, "d1", [kept.entry_id])
    recovered = await self.streams.reclaim(SET, "d2", 10, min_idle=0.2)
    self.assertEqual([c.entry_id for c in recovered], [dropped.entry_id])

  async def test_removing_a_set_closes_it_and_cancels_its_unfinished_requests(self) -> None:
    request_id = await self.accept()
    sync = sync_redis.Redis.from_url(self.server.url)
    try:
      self.assertEqual(streams.close_and_cancel(sync, SET, "set removed"), 1)
    finally:
      sync.close()
    self.assertEqual((await self.streams.result(request_id))["error_message"], "set removed")
    self.assertEqual(await self.streams.unfinished(SET), 0)
    with self.assertRaises(SetClosed):
      await self.accept()
    # A late attempt cannot replace the cancellation.
    self.assertFalse(await self.streams.finish(SET, streams.parse(request_id)[1], SAMPLE))

  async def test_unknown_and_expired_ids_never_read_as_pending(self) -> None:
    request_id = await self.accept()
    await self.streams.finish(SET, streams.parse(request_id)[1], SAMPLE)
    await self.client.delete(streams.result_key(SET, streams.parse(request_id)[1]))
    self.assertIn("not found", (await self.streams.result(request_id))["error_message"])
    self.assertEqual((await self.streams.result("not-a-stream-id"))["type"], "RequestFailedResponse")


if __name__ == "__main__":
  unittest.main()
