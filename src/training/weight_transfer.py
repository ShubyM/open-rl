"""Push trained weights into a running vLLM engine over NCCL.

UNREFERENCED, and kept for one run as the fallback for the path that replaced
it. This existed because the Megatron worker was believed unable to use the
filesystem route every other backend takes -- the trainer writes a LoRA adapter,
the sampler loads it with /v1/load_lora_adapter -- on the grounds that its
export merges the adapter into the base weights and produces a whole HF
checkpoint that stock vLLM cannot hot-reload. That is true of
export_hf_weights and false of megatron-bridge, which also has save_hf_adapter.
megatron_worker.write_adapter uses it, and this module's whole reason to exist
went with it. Delete once the adapter path has a run behind it.

vLLM 0.25.1 ships the third option. `vllm serve --weight-transfer-config
'{"backend":"nccl"}'` with VLLM_SERVER_DEV_MODE=1 exposes an RLHF router whose
workers join a NCCL group with the trainer and receive tensors directly into
the live engine's memory, no file anywhere. That is what SkyRL, OpenRLHF and
slime all converged on.

Two things about the protocol are easy to get wrong, and both are why this is
a module rather than three lines at the call site:

  * /init_weight_transfer_engine and /update_weights BLOCK on the server while
    it joins the group and receives. The trainer has to be broadcasting at the
    same moment, so each is posted from a thread while this rank does its half.
    Post them sequentially and both sides deadlock waiting for the other.
  * update_info carries names, dtypes and shapes up front, but the weights
    arrive from a one-shot generator. Rather than trust that a separately
    derived name list matches the generator's order -- a mismatch would load
    tensors under the wrong names, which no error would catch -- this takes a
    factory and walks it twice: once for metadata, once to send. The generator
    is deterministic, so the orders agree by construction.

Names need no translation. The bridge yields checkpoint-format names
(model.language_model.*), which is exactly what vLLM's load_weights already
consumes -- it read the same names out of model.safetensors at startup. This
is the one place the naming is free; the PEFT path needs the remap in
lora_trainer_worker.py precisely because adapter files do not use them.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

import torch


def _request(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
  data = None if payload is None else json.dumps(payload).encode()
  headers = {"Content-Type": "application/json"} if data else {}
  request = urllib.request.Request(url, data=data, headers=headers)
  with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
    body = response.read()
  return json.loads(body) if body else {}


class _Call(threading.Thread):
  """A POST that has to overlap with this rank's half of the handshake.

  A bare Thread swallows the exception and joins cleanly, which would turn a
  rejected request into a hang on the NCCL side. Keep it and re-raise.
  """

  def __init__(self, url: str, payload: dict[str, Any] | None, timeout: float) -> None:
    super().__init__(daemon=True)
    self._url, self._payload, self._timeout = url, payload, timeout
    self.error: BaseException | None = None

  def run(self) -> None:
    try:
      _request(self._url, self._payload, self._timeout)
    except BaseException as exc:  # noqa: BLE001
      self.error = exc

  def finish(self) -> None:
    self.join()
    if self.error is not None:
      raise RuntimeError(f"vLLM rejected {self._url}: {self.error}") from self.error


def drain(weights: Callable[[], Iterator[tuple[str, torch.Tensor]]]) -> None:
  """What the ranks that are not sending have to do while rank 0 syncs.

  export_hf_weights gathers, so producing each tensor is a collective across
  the whole trainer group. A rank that skips the export while rank 0 walks it
  hangs the step. Walk it twice, discarding: once against rank 0's metadata
  pass, once against its send.
  """
  for _ in range(2):
    for _name, _tensor in weights():
      pass


class VllmWeightTransfer:
  """A NCCL group between this trainer rank and one vLLM server's workers.

  Only one trainer rank should own an instance. export_hf_weights gathers, so
  every rank holds the full tensors and one sender is enough; having several
  would put multiple senders in the same group at rank 0. The rest call
  drain().
  """

  def __init__(
    self,
    base_url: str,
    master_address: str,
    master_port: int,
    device: torch.device | str,
    timeout: float = 1800.0,
  ) -> None:
    self.base_url = base_url.rstrip("/")
    self.master_address = master_address
    self.master_port = master_port
    self.device = device
    self.timeout = timeout
    self._group: Any = None

  def _ensure_group(self) -> None:
    """Join the trainer to the engine's workers, once per process."""
    if self._group is not None:
      return
    from vllm.distributed.weight_transfer.nccl_common import stateless_init_process_group

    # get_world_size counts the engine's workers; this rank is the extra one at
    # rank 0, so the engine's ranks start at 1.
    workers = int(_request(f"{self.base_url}/get_world_size", None, 60.0)["world_size"])
    init = _Call(
      f"{self.base_url}/init_weight_transfer_engine",
      {
        "init_info": {
          "master_address": self.master_address,
          "master_port": self.master_port,
          "rank_offset": 1,
          "world_size": workers + 1,
        }
      },
      self.timeout,
    )
    init.start()
    self._group = stateless_init_process_group(
      self.master_address, self.master_port, 0, workers + 1, self.device
    )
    init.finish()

  def sync(self, weights: Callable[[], Iterator[tuple[str, torch.Tensor]]]) -> int:
    """Broadcast one full set of weights. Returns how many tensors were sent.

    `weights` is called twice and must yield the same sequence both times --
    see the module docstring. Every trainer rank must reach drain() or this,
    the same number of times, or the export's gathers hang.
    """
    from vllm.distributed.weight_transfer.nccl_engine import (
      NCCLTrainerSendWeightsArgs,
      NCCLWeightTransferEngine,
    )

    self._ensure_group()
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    for name, tensor in weights():
      names.append(name)
      # getattr(torch, dtype_name) on the far side, so "bfloat16" not "torch.bfloat16".
      dtype_names.append(str(tensor.dtype).rsplit(".", 1)[-1])
      shapes.append(list(tensor.shape))

    _request(f"{self.base_url}/start_weight_update", {}, self.timeout)
    update = _Call(
      f"{self.base_url}/update_weights",
      {"update_info": {"names": names, "dtype_names": dtype_names, "shapes": shapes, "packed": True}},
      self.timeout,
    )
    update.start()
    try:
      NCCLWeightTransferEngine.trainer_send_weights(
        weights(),
        NCCLTrainerSendWeightsArgs(group=self._group, src=0, packed=True),
      )
    finally:
      update.finish()
    _request(f"{self.base_url}/finish_weight_update", {}, self.timeout)
    return len(names)
