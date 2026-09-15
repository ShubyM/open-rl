"""Small torch.distributed boundary for trainer workers launched by torchrun."""

import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


def world_size() -> int:
  return int(os.getenv("WORLD_SIZE", "1"))


def rank() -> int:
  return int(os.getenv("RANK", "0"))


def local_rank() -> int:
  return int(os.getenv("LOCAL_RANK", "0"))


def is_distributed() -> bool:
  return world_size() > 1


def is_primary() -> bool:
  return rank() == 0


def initialize() -> None:
  """Join the process group torchrun created and select this rank's GPU."""
  if not is_distributed() or dist.is_initialized():
    return
  if torch.cuda.is_available():
    torch.cuda.set_device(local_rank())
  default_backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
  dist.init_process_group(
    backend=os.getenv("OPEN_RL_CONTROL_BACKEND", default_backend),
    timeout=timedelta(seconds=int(os.getenv("OPEN_RL_DISTRIBUTED_TIMEOUT", "1800"))),
  )


def close() -> None:
  if dist.is_initialized():
    dist.destroy_process_group()


def barrier() -> None:
  if is_distributed():
    dist.barrier()


def broadcast_object(value: Any = None) -> Any:
  """Rank 0's value on every rank."""
  if not is_distributed():
    return value
  values = [value if is_primary() else None]
  dist.broadcast_object_list(values, src=0)
  return values[0]


# Reductions over one process group. None stands for a process training alone.


def group_rank(group: dist.ProcessGroup | None) -> int:
  return 0 if group is None else dist.get_rank(group)


def group_size(group: dist.ProcessGroup | None) -> int:
  return 1 if group is None else dist.get_world_size(group)


def reduce_device() -> torch.device:
  return torch.device("cuda", local_rank()) if torch.cuda.is_available() else torch.device("cpu")


def all_reduce_sum(value: float, group: dist.ProcessGroup | None) -> float:
  if group is None:
    return value
  tensor = torch.tensor([value], dtype=torch.float64, device=reduce_device())
  dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
  return float(tensor.item())


def all_reduce_max(value: int, group: dist.ProcessGroup | None) -> int:
  if group is None:
    return value
  tensor = torch.tensor([value], dtype=torch.int64, device=reduce_device())
  dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
  return int(tensor.item())


def all_gather_object(value: Any, group: dist.ProcessGroup | None) -> list[Any]:
  if group is None:
    return [value]
  values: list[Any] = [None] * dist.get_world_size(group)
  dist.all_gather_object(values, value, group=group)
  return values
