"""Data-parallel sharding in BaseTrainerWorker against real FSDP2 on a CPU gloo mesh."""

import os
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from server.api import wire
from training.trainer_worker import BaseTrainerWorker

VOCAB = 12


class TableModel(torch.nn.Module):
  """logprob of a token is -table[token]; FSDP2 shards the table over the mesh."""

  def __init__(self):
    super().__init__()
    self.table = torch.nn.Parameter(torch.linspace(0.1, 1.2, VOCAB).unsqueeze(-1))

  def forward(self, targets: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.embedding(targets, self.table).squeeze(-1)


class TableWorker(BaseTrainerWorker):
  """Shards over the group it is given, the way a backend with a DP subgroup does."""

  backward_runs_collectives = True

  def __init__(self, group):
    super().__init__()
    self.device = torch.device("cpu")
    self.group = group

  def shard_rank(self) -> int:
    return dist.get_rank(self.group) if self.group is not None else 0

  def shard_count(self) -> int:
    return dist.get_world_size(self.group) if self.group is not None else 1

  def shard_all_reduce_max(self, value: int) -> int:
    tensor = torch.tensor([value])
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.group)
    return int(tensor.item())

  def shard_all_reduce_sum(self, value: float) -> float:
    tensor = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
    return float(tensor.item())

  def shard_all_gather_object(self, value):
    parts: list = [None] * dist.get_world_size(self.group)
    dist.all_gather_object(parts, value, group=self.group)
    return parts

  def compute_target_logprobs(self, model, input_ids, attention_mask, target_token_ids):
    return model(target_token_ids)


def table_data() -> list[wire.Datum]:
  # Three datums over two ranks: rank 1 runs one filler pass.
  return [
    wire.Datum(
      np.asarray([1, 2, 3], dtype=np.int32),
      {"target_tokens": np.asarray([2, 3, 4], dtype=np.int64), "weights": np.asarray([1.0, 0.5, 0.25], dtype=np.float32)},
    ),
    wire.Datum(
      np.asarray([5, 6], dtype=np.int32),
      {"target_tokens": np.asarray([6, 7], dtype=np.int64), "weights": np.asarray([2.0, 0.75], dtype=np.float32)},
    ),
    wire.Datum(
      np.asarray([8], dtype=np.int32),
      {"target_tokens": np.asarray([9], dtype=np.int64), "weights": np.asarray([1.5], dtype=np.float32)},
    ),
  ]


def run_fsdp_data_parallel(rank: int, world_size: int, port: int) -> None:
  os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
  dist.init_process_group("gloo", rank=rank, world_size=world_size)
  try:
    reference_model = TableModel()
    expected = wire.fb_output_to_json(TableWorker(None).forward_backward(reference_model, table_data(), "cross_entropy")["proto_bytes"])

    mesh = init_device_mesh("cpu", (world_size,))
    model = fully_shard(TableModel(), mesh=mesh)
    actual = wire.fb_output_to_json(TableWorker(mesh.get_group()).forward_backward(model, table_data(), "cross_entropy")["proto_bytes"])

    # FSDP2 averaged the gradient over the mesh; the loss scaling must turn
    # that average back into the full-batch sum.
    torch.testing.assert_close(model.table.grad.full_tensor(), reference_model.table.grad)
    assert actual["metrics"] == expected["metrics"], (actual["metrics"], expected["metrics"])
    assert actual["loss_fn_outputs"] == expected["loss_fn_outputs"]
  finally:
    dist.destroy_process_group()


class DataParallelTest(unittest.TestCase):
  def test_sharded_gradient_and_outputs_match_one_process(self) -> None:
    mp.spawn(run_fsdp_data_parallel, args=(2, 29541), nprocs=2, join=True)


if __name__ == "__main__":
  unittest.main()
