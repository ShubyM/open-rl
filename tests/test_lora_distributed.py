"""Data-parallel LoRA collectives: gradient sync math and rank gating.

The gradient contract under test: BaseTrainerWorker.forward_backward scales
each rank's loss by the shard count, so all_reduce_gradients (sum / world)
must reproduce the exact single-process gradient over the full datum list.
"""

import multiprocessing as mp
import os
import unittest
from unittest.mock import patch

import torch


def _run_rank(rank: int, world_size: int, port: int, results) -> None:
  os.environ.update(
    {
      "RANK": str(rank),
      "LOCAL_RANK": str(rank),
      "WORLD_SIZE": str(world_size),
      "MASTER_ADDR": "127.0.0.1",
      "MASTER_PORT": str(port),
      "OPEN_RL_CONTROL_BACKEND": "gloo",
    }
  )
  from training import distributed

  distributed.initialize()
  try:
    # --- broadcast_parameters: every rank ends with rank 0's values ---
    param = torch.nn.Parameter(torch.full((4,), float(rank)))
    distributed.broadcast_parameters([param])
    broadcast_ok = torch.equal(param.data, torch.zeros(4))

    # --- all_reduce_gradients reproduces the single-process gradient ---
    # Single process over datums [1, 2, 3, 4] with unit-weight identity loss:
    # total gradient = 1+2+3+4 = 10 per element.
    # Data-parallel: rank 0 shards [1, 3], rank 1 shards [2, 4]; each rank's
    # loss is scaled by shard_count=2 (mirroring forward_backward), so local
    # grads are 2*(1+3)=8 and 2*(2+4)=12. sum/world = (8+12)/2 = 10.
    shard = [1.0, 3.0] if rank == 0 else [2.0, 4.0]
    weight = torch.nn.Parameter(torch.zeros(3))
    for value in shard:
      loss = (weight * value).sum() * float(world_size)
      loss.backward()
    distributed.all_reduce_gradients([weight])
    grad_ok = torch.allclose(weight.grad, torch.full((3,), 10.0))

    # --- a rank missing a grad contributes zeros, not a deadlock ---
    holey = torch.nn.Parameter(torch.zeros(2))
    if rank == 0:
      (holey * 6.0).sum().backward()  # only rank 0 touched it: local grad 6
    distributed.all_reduce_gradients([holey])
    holey_ok = torch.allclose(holey.grad, torch.full((2,), 3.0))  # 6/world

    results[rank] = {"broadcast": broadcast_ok, "grad": grad_ok, "holey": holey_ok}
  finally:
    distributed.close()


class LoraDataParallelCollectivesTest(unittest.TestCase):
  def test_two_rank_gradient_sync_matches_single_process(self):
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
      results = manager.dict()
      port = 29431
      procs = [ctx.Process(target=_run_rank, args=(r, 2, port, results)) for r in range(2)]
      for p in procs:
        p.start()
      for p in procs:
        p.join(timeout=120)
        self.assertEqual(p.exitcode, 0)
      for rank in range(2):
        self.assertTrue(results[rank]["broadcast"], f"rank {rank}: broadcast_parameters")
        self.assertTrue(results[rank]["grad"], f"rank {rank}: gradient sync mismatch")
        self.assertTrue(results[rank]["holey"], f"rank {rank}: missing-grad rank")


class RankGatingTest(unittest.TestCase):
  def test_single_process_helpers_are_no_ops(self):
    from training import distributed

    param = torch.nn.Parameter(torch.ones(2))
    (param * 5.0).sum().backward()
    distributed.all_reduce_gradients([param])
    self.assertTrue(torch.allclose(param.grad, torch.full((2,), 5.0)))
    distributed.broadcast_parameters([param])
    self.assertTrue(torch.equal(param.data, torch.ones(2)))

  def test_non_primary_rank_does_not_publish_futures(self):
    import asyncio

    from server import training_requests_processor as trp

    class Store:
      def __init__(self):
        self.published = []

      async def set_future(self, request_id, result):
        self.published.append(request_id)

    class Proc(trp.LoraTrainingRequestsProcessor):
      def __init__(self, store):
        self.store = store

      async def handle_request(self, raw_request, model_id=None):
        return raw_request["request_id"], {"type": "ok"}

    store = Store()
    proc = Proc(store)
    with patch.object(trp, "is_primary", return_value=False):
      asyncio.run(proc.process_request({"request_id": "r1"}, "m"))
    self.assertEqual(store.published, [])
    with patch.object(trp, "is_primary", return_value=True):
      asyncio.run(proc.process_request({"request_id": "r2"}, "m"))
    self.assertEqual(store.published, ["r2"])


if __name__ == "__main__":
  unittest.main()
