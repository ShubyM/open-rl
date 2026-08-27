"""The torch.distributed boundary, kept to what the trainer actually calls.

Everything here is a no-op when WORLD_SIZE is unset, which is every path that
exists today: a worker launched without torchrun behaves exactly as it did
before this module existed. Nothing below is imported for its own sake -- each
function has a call site in trainer_worker.py, the request loop, or a
multi-rank backend.
"""

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
  """Join the process group torchrun set up and claim this rank's GPU."""
  if not is_distributed() or dist.is_initialized():
    return
  if torch.cuda.is_available():
    torch.cuda.set_device(local_rank())
  # gloo by default because the request loop broadcasts pickled request batches,
  # which is CPU work. A backend needing CUDA collectives on the default group
  # (Megatron's tensor parallelism does) sets "cpu:gloo,cuda:nccl" so both kinds
  # of traffic come off one group.
  dist.init_process_group(
    backend=os.getenv("OPEN_RL_CONTROL_BACKEND", "gloo"),
    timeout=timedelta(seconds=int(os.getenv("OPEN_RL_DISTRIBUTED_TIMEOUT", "1800"))),
  )


def shutdown() -> None:
  if dist.is_initialized():
    dist.destroy_process_group()


def barrier() -> None:
  if is_distributed():
    dist.barrier()


def broadcast_object(value: Any = None) -> Any:
  """Hand rank 0's value to every rank. Non-primary ranks pass None."""
  if not is_distributed():
    return value
  values = [value if is_primary() else None]
  dist.broadcast_object_list(values, src=0)
  return values[0]


def _collective_device(group: Any) -> torch.device:
  """NCCL only moves CUDA tensors; gloo is happiest on CPU."""
  if torch.cuda.is_available() and "nccl" in str(dist.get_backend(group)):
    return torch.device("cuda", local_rank())
  return torch.device("cpu")


def all_reduce_max(value: int, group: Any = None) -> int:
  if not is_distributed():
    return value
  tensor = torch.tensor([value], dtype=torch.int64, device=_collective_device(group))
  dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
  return int(tensor.item())


def all_reduce_sum(value: float, group: Any = None) -> float:
  if not is_distributed():
    return value
  tensor = torch.tensor([value], dtype=torch.float64, device=_collective_device(group))
  dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
  return float(tensor.item())


def all_gather_object(value: Any, group: Any = None) -> list[Any]:
  if not is_distributed():
    return [value]
  gathered: list[Any] = [None] * dist.get_world_size(group)
  dist.all_gather_object(gathered, value, group=group)
  return gathered
