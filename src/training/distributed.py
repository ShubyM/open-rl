"""Small torch.distributed boundary shared by the FFT worker and its queue loop."""

import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

_fsdp_group: dist.ProcessGroup | None = None


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
  """Initialize the process group created by torchrun and select this rank's GPU."""
  if not is_distributed() or dist.is_initialized():
    return
  if torch.cuda.is_available():
    torch.cuda.set_device(local_rank())
  dist.init_process_group(
    backend=os.getenv("OPEN_RL_CONTROL_BACKEND", "gloo"),
    timeout=timedelta(seconds=int(os.getenv("OPEN_RL_DISTRIBUTED_TIMEOUT", "1800"))),
  )


def fsdp_group() -> dist.ProcessGroup:
  """Create the CUDA group lazily, after rank 0 owns the trainer GPU lease."""
  global _fsdp_group
  if not is_distributed() or not dist.is_initialized():
    raise RuntimeError("FSDP process group requested before distributed initialization")
  if not torch.cuda.is_available():
    raise RuntimeError("OPEN_RL FSDP requires CUDA and one torchrun process per GPU")
  if _fsdp_group is None:
    _fsdp_group = dist.new_group(backend="nccl")
  return _fsdp_group


def barrier() -> None:
  if is_distributed():
    dist.barrier()


def broadcast_object(value: Any = None) -> Any:
  if not is_distributed():
    return value
  values = [value if is_primary() else None]
  dist.broadcast_object_list(values, src=0)
  return values[0]


def all_reduce_sum(value: float) -> float:
  if not is_distributed():
    return value
  tensor = torch.tensor([value], dtype=torch.float64)
  dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
  return tensor.item()


def all_reduce_max(value: int) -> int:
  if not is_distributed():
    return value
  tensor = torch.tensor([value], dtype=torch.int64)
  dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
  return int(tensor.item())


def all_gather_object(value: Any) -> list[Any]:
  if not is_distributed():
    return [value]
  values: list[Any] = [None] * world_size()
  dist.all_gather_object(values, value)
  return values


def close() -> None:
  global _fsdp_group
  if dist.is_initialized():
    dist.destroy_process_group()
  _fsdp_group = None
