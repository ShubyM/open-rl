"""The process's Kubernetes watches: restarted once they end, and quiet about a kind the cluster does not serve."""

import asyncio
import unittest
from unittest.mock import patch

from lightkube import ApiError
from lightkube.models.meta_v1 import Status

from server import kube


class WatchTest(unittest.IsolatedAsyncioTestCase):
  async def test_a_watch_that_ended_is_started_again(self) -> None:
    started = []

    async def follow(api, kind, resource, namespace):
      started.append(kind)
      await asyncio.Event().wait()

    with patch.object(kube, "follow", follow), patch.object(kube, "client", object()), patch.dict(kube.watchers, clear=True):
      kube.watch(["pods"])
      await asyncio.sleep(0)
      kube.watchers["pods"].cancel()  # a lifespan that stopped, or a test's loop that closed
      await asyncio.sleep(0)
      kube.watch(["pods"])
      await asyncio.sleep(0)
      kube.watch(["pods"])  # a live watch is left alone
      await asyncio.sleep(0)
      kube.watchers["pods"].cancel()
    self.assertEqual(started, ["pods", "pods"])

  async def test_a_kind_the_cluster_does_not_serve_is_listed_again_only_at_the_resync(self) -> None:
    class NoCrd:
      def list(self, resource, namespace):
        raise ApiError(status=Status(code=404))

    waits = []

    async def sleep(seconds):
      waits.append(seconds)
      raise asyncio.CancelledError

    with patch.object(kube.asyncio, "sleep", sleep), patch.dict(kube.watched), patch.dict(kube.watch_errors):
      with self.assertRaises(asyncio.CancelledError):
        await kube.follow(NoCrd(), "workloads", None, "ns")
      self.assertEqual((kube.watched["workloads"], kube.watch_errors["workloads"]), (None, None))
    self.assertEqual(waits, [kube.RESYNC_SECONDS])


if __name__ == "__main__":
  unittest.main()
