"""Pinned host copies of one model's training state.

A dedicated trainer runs under the time slicer's GPU lease and must vacate the
device between leases. The mirror is where the state goes: pinned host buffers
for parameters and buffers, pending gradients, and optimizer state, allocated
once and reused across leases. sleep() moves the state there and wake_up()
brings it back.

In delta mode the parameter copies double as the baseline a sparse delta is
taken against. sync() and diff() keep the invariant that a trainable
parameter's host copy equals what the sampler last received, so delta mode
needs no host memory beyond what offload already uses.
"""

import gc
import itertools
import time

import torch

# (the device a tensor lives on, its pinned host copy)
Entry = tuple[torch.device, torch.Tensor]


def per_param_state(optimizer: torch.optim.Optimizer | None) -> list[tuple[torch.Tensor, dict]]:
  if optimizer is None:
    return []
  return [(param, state) for param, state in optimizer.state.items() if isinstance(state, dict)]


class HostMirror:
  def __init__(self) -> None:
    self.offloaded = False
    # Parameters and buffers are keyed by the tensor itself; gradients by their
    # parameter, since backward makes a new grad tensor; optimizer state by
    # (parameter, key), since the optimizer replaces its state tensors.
    self.weights: dict[torch.Tensor, Entry] = {}
    self.grads: dict[torch.Tensor, Entry] = {}
    self.optimizer_state: dict[tuple[torch.Tensor, str], Entry] = {}

  @staticmethod
  def host_copy(table: dict, key: object, like: torch.Tensor) -> torch.Tensor:
    """The pinned host copy under key, allocated to match `like` on first use or when its shape changes."""
    entry = table.get(key)
    if entry is None or entry[1].shape != like.shape or entry[1].dtype != like.dtype:
      entry = (like.device, torch.empty(like.shape, dtype=like.dtype, device="cpu", pin_memory=torch.cuda.is_available()))
      table[key] = entry
    return entry[1]

  # -- delta baseline ---------------------------------------------------------

  def sync(self, model: torch.nn.Module) -> None:
    """Make every trainable parameter's host copy equal its device value."""
    for param in model.parameters():
      if param.requires_grad:
        self.host_copy(self.weights, param, param.data).copy_(param.data, non_blocking=True)

  def diff(self, param: torch.nn.Parameter) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Flat indices and values, on the host, of the elements of param that changed
    since its host copy was last synced; None if nothing changed. Advances the copy.
    A parameter seen for the first time is synced and reports no change."""
    entry = self.weights.get(param)
    if entry is None:
      self.host_copy(self.weights, param, param.data).copy_(param.data, non_blocking=True)
      return None
    baseline = entry[1].view(-1)
    current = param.data.view(-1)
    changed = current.ne(baseline.to(param.device, non_blocking=True))
    indices = changed.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
      return None
    # int32 indices halve the file when they fit; the sampler accepts either.
    index_dtype = torch.int32 if param.numel() <= 2**31 else torch.int64
    indices = indices.to(index_dtype).cpu()
    values = current[changed].cpu()
    baseline[indices.to(torch.int64)] = values
    return indices, values

  # -- GPU lease ----------------------------------------------------------------

  def sleep(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> None:
    """Copy the device-resident training state to the host and free it on the device.

    Parameters are left pointing at their host copies, so shapes stay right and
    save_pretrained reads them directly while the worker is off the GPU.
    """
    if self.offloaded or not torch.cuda.is_available():
      return
    start = time.perf_counter()

    # Queue every device-to-host copy, then wait once, then free: the copies
    # overlap and nothing is released before it has landed.
    for tensor in itertools.chain(model.parameters(), model.buffers()):
      if tensor.device.type == "cuda":
        self.host_copy(self.weights, tensor, tensor.data).copy_(tensor.data, non_blocking=True)
      grad = tensor.grad if isinstance(tensor, torch.nn.Parameter) else None
      if grad is not None and grad.device.type == "cuda":
        self.host_copy(self.grads, tensor, grad).copy_(grad, non_blocking=True)
    for param, state in per_param_state(optimizer):
      for key, value in list(state.items()):
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
          self.host_copy(self.optimizer_state, (param, key), value).copy_(value, non_blocking=True)

    torch.cuda.synchronize()

    for tensor in itertools.chain(model.parameters(), model.buffers()):
      entry = self.weights.get(tensor)
      if entry is not None:
        tensor.data = entry[1]
      grad = tensor.grad if isinstance(tensor, torch.nn.Parameter) else None
      if grad is not None and tensor in self.grads:
        grad.data = self.grads[tensor][1]
    for param, state in per_param_state(optimizer):
      for key in list(state):
        entry = self.optimizer_state.get((param, key))
        if entry is not None:
          state[key] = entry[1]

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    self.offloaded = True
    print(f"[HostMirror] Offloaded weights & states to pinned host memory in {(time.perf_counter() - start) * 1000:.1f} ms.")

  def wake_up(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> None:
    """Bring the training state back to the device. The host copies stay allocated for the next sleep."""
    if not self.offloaded:
      return
    start = time.perf_counter()

    for tensor in itertools.chain(model.parameters(), model.buffers()):
      entry = self.weights.get(tensor)
      if entry is not None:
        tensor.data = entry[1].to(entry[0], non_blocking=True)
      grad = tensor.grad if isinstance(tensor, torch.nn.Parameter) else None
      if grad is not None and tensor in self.grads:
        device, host = self.grads[tensor]
        grad.data = host.to(device, non_blocking=True)
    for param, state in per_param_state(optimizer):
      for key in list(state):
        entry = self.optimizer_state.get((param, key))
        if entry is not None:
          state[key] = entry[1].to(entry[0], non_blocking=True)

    torch.cuda.synchronize()
    self.offloaded = False
    print(f"[HostMirror] Reloaded weights & states to the device in {(time.perf_counter() - start) * 1000:.1f} ms.")
