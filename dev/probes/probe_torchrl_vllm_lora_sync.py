"""Probe whether TorchRL/vLLM NCCL weight sync accepts LoRA-only tensors.

Run metadata-only on any machine:

  uv run --directory src/server --extra cpu --with torchrl \
    python ../../dev/probes/probe_torchrl_vllm_lora_sync.py --metadata-only

Run the live vLLM/NCCL check on a GPU machine with vLLM available:

  uv run --directory src/server --extra vllm --extra gpu --with torchrl \
    python ../../dev/probes/probe_torchrl_vllm_lora_sync.py --live-vllm

If that transient env resolves incompatible torch/torchvision CUDA wheels,
install TorchRL/Ray into the same vLLM image instead of using `uv --with`.
On WSL with a single visible GPU, NCCL group initialization can fail before this
probe reaches the LoRA-name application check; use a two-GPU setup for the real
broadcast test.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import pickle
import socket
import sys
import tempfile
from pathlib import Path

import pybase64 as base64

# Keep Transformers from importing optional vision modules. This probe only
# needs text models, and mixed vLLM/TorchRL scratch envs can have mismatched
# torchvision wheels.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "src" / "server"
sys.path.insert(0, str(SERVER_DIR))

from state_delta import normalize_lora_tensor_name  # noqa: E402
from weight_sync import LoraTensorSelector, TorchRLVLLMTransferEngine  # noqa: E402


def unused_tcp_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return int(sock.getsockname()[1])


def write_tiny_llama_model(path: Path) -> None:
  vocab = {str(token): token for token in range(64)}
  vocab.update({"[UNK]": 64, "[PAD]": 65, "[BOS]": 66, "[EOS]": 67})
  tokenizer_model = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
  tokenizer_model.pre_tokenizer = Whitespace()
  tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_model,
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[BOS]",
    eos_token="[EOS]",
  )
  tokenizer.save_pretrained(path)

  config = LlamaConfig(
    vocab_size=68,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=64,
    pad_token_id=65,
    bos_token_id=66,
    eos_token_id=67,
  )
  LlamaForCausalLM(config).save_pretrained(path)


def make_lora_model(model_path: Path):
  base = LlamaForCausalLM.from_pretrained(model_path)
  config = LoraConfig(
    task_type="CAUSAL_LM",
    r=2,
    lora_alpha=4,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
  )
  return get_peft_model(base, config, adapter_name="probe")


def print_probe_metadata(model_path: Path) -> list:
  model = make_lora_model(model_path)
  tensors = LoraTensorSelector().select(model, "probe")
  metadata = TorchRLVLLMTransferEngine.model_metadata_for(tensors)

  print(f"tiny_model={model_path}")
  print(f"selected_lora_tensors={len(tensors)}")
  for item in tensors[:8]:
    print(f"  {item.name} dtype={item.dtype} shape={item.shape}")
  if len(tensors) > 8:
    print(f"  ... {len(tensors) - 8} more")
  print(f"metadata_entries={len(metadata)}")

  with contextlib.suppress(Exception):
    from torchrl.weight_update.llm import get_model_metadata

    default_metadata = get_model_metadata(make_lora_model(model_path))
    lora_keys = [key for key in default_metadata if "lora_" in key]
    print(f"torchrl_default_metadata_entries={len(default_metadata)}")
    print(f"torchrl_default_lora_keys={len(lora_keys)}")
    print("torchrl_default_merges_lora=", len(lora_keys) == 0)

  return tensors


def run_live_vllm_probe(model_path: Path, tensors: list, device: int) -> None:
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for live vLLM/NCCL probe")

  import threading

  from vllm import LLM
  from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
  )
  from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
  )

  torch.cuda.set_device(device)
  llm = LLM(
    model=str(model_path),
    enforce_eager=True,
    max_model_len=64,
    gpu_memory_utilization=0.20,
    dtype="float32",
    enable_lora=True,
    max_lora_rank=8,
    weight_transfer_config={"backend": "nccl"},
  )

  master_port = unused_tcp_port()
  trainer_group: dict[str, object] = {}
  errors: dict[str, Exception] = {}

  def init_trainer_group() -> None:
    try:
      trainer_group["group"] = NCCLWeightTransferEngine.trainer_init(
        NCCLWeightTransferInitInfo(
          master_address="127.0.0.1",
          master_port=master_port,
          rank_offset=0,
          world_size=2,
        )
      )
    except Exception as exc:
      errors["trainer_init"] = exc

  trainer_thread = threading.Thread(target=init_trainer_group)
  trainer_thread.start()
  llm.init_weight_transfer_engine(
    WeightTransferInitRequest(
      init_info={
        "master_address": "127.0.0.1",
        "master_port": master_port,
        "rank_offset": 1,
        "world_size": 2,
      }
    )
  )
  trainer_thread.join(timeout=30)
  if trainer_thread.is_alive():
    raise RuntimeError("trainer NCCL init did not finish")
  if errors:
    raise next(iter(errors.values()))

  update_info = NCCLWeightTransferUpdateInfo(
    names=[item.name for item in tensors],
    dtype_names=[str(item.tensor.dtype).removeprefix("torch.") for item in tensors],
    shapes=[list(item.shape) for item in tensors],
    packed=False,
    is_checkpoint_format=False,
  )

  def update_worker() -> None:
    try:
      llm.update_weights(WeightTransferUpdateRequest(update_info=update_info.__dict__))
    except Exception as exc:
      errors["worker_update"] = exc

  update_thread = threading.Thread(target=update_worker)
  update_thread.start()
  NCCLWeightTransferEngine.trainer_send_weights(
    ((item.name, item.tensor.to(f"cuda:{device}")) for item in tensors),
    NCCLTrainerSendWeightsArgs(group=trainer_group["group"], packed=False),
  )
  update_thread.join(timeout=30)
  if update_thread.is_alive():
    raise RuntimeError("vLLM weight update did not finish")
  if errors:
    raise next(iter(errors.values()))

  print("LIVE_VLLM_LORA_ONLY_SYNC=success")


def run_live_vllm_ipc_probe(model_path: Path, tensors: list, device: int) -> None:
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for live vLLM IPC probe")

  from torch.multiprocessing.reductions import reduce_tensor
  from vllm import LLM
  from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
  )
  from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferUpdateInfo

  torch.cuda.set_device(device)
  llm = LLM(
    model=str(model_path),
    enforce_eager=True,
    max_model_len=64,
    gpu_memory_utilization=0.20,
    dtype="float32",
    enable_lora=True,
    max_lora_rank=8,
    weight_transfer_config={"backend": "ipc"},
  )
  llm.init_weight_transfer_engine(WeightTransferInitRequest(init_info={}))

  props = torch.cuda.get_device_properties(device)
  gpu_uuid = str(props.uuid)
  cuda_tensors = [item.tensor.to(f"cuda:{device}").detach().contiguous() for item in tensors]
  update_info = IPCWeightTransferUpdateInfo(
    names=[item.name for item in tensors],
    dtype_names=[str(item.tensor.dtype).removeprefix("torch.") for item in tensors],
    shapes=[list(item.shape) for item in tensors],
    ipc_handles_pickled=base64.b64encode(pickle.dumps([{gpu_uuid: reduce_tensor(tensor)} for tensor in cuda_tensors])).decode("utf-8"),
    is_checkpoint_format=False,
  )

  llm.update_weights(WeightTransferUpdateRequest(update_info=update_info.__dict__))
  torch.cuda.synchronize()
  print("LIVE_VLLM_IPC_LORA_ONLY_SYNC=success")


def run_live_custom_lora_probe(model_path: Path, tensors: list, device: int) -> None:
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for live custom LoRA probe")

  from safetensors.torch import save_file
  from vllm import LLM, SamplingParams
  from vllm.lora.request import LoRARequest

  torch.cuda.set_device(device)
  adapter_dir = model_path.parent / "custom-lora-adapter"
  adapter_dir.mkdir(parents=True, exist_ok=True)
  payload = {
    normalize_lora_tensor_name(item.name, "probe"): item.tensor.to(torch.float16)
    for item in tensors
    if normalize_lora_tensor_name(item.name, "probe") is not None
  }
  save_file(
    payload,
    adapter_dir / "adapter_model.safetensors",
  )
  config = {
    "base_model_name_or_path": str(model_path),
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "lora_alpha": 4,
    "lora_dropout": 0.0,
    "peft_type": "LORA",
    "r": 2,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "task_type": "CAUSAL_LM",
  }
  with (adapter_dir / "adapter_config.json").open("w") as f:
    import json

    json.dump(config, f)

  llm = LLM(
    model=str(model_path),
    enforce_eager=True,
    max_model_len=64,
    gpu_memory_utilization=0.20,
    dtype="float16",
    enable_lora=True,
    max_lora_rank=8,
  )
  request = LoRARequest("custom-probe@1", 1, str(adapter_dir), load_inplace=True)
  outputs = llm.generate(
    prompts=["1 2"],
    sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
    lora_request=request,
  )
  print(f"LIVE_CUSTOM_LORA_APPLY=success outputs={len(outputs)}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--metadata-only", action="store_true", help="Only inspect LoRA tensor metadata.")
  parser.add_argument("--live-vllm", action="store_true", help="Run live TorchRL/vLLM NCCL update.")
  parser.add_argument("--live-vllm-ipc", action="store_true", help="Run live vLLM CUDA IPC update.")
  parser.add_argument("--live-custom-lora", action="store_true", help="Run live custom vLLM LoRA adapter apply.")
  parser.add_argument("--device", type=int, default=0)
  args = parser.parse_args()

  if not args.metadata_only and not args.live_vllm and not args.live_vllm_ipc and not args.live_custom_lora:
    args.metadata_only = True

  with tempfile.TemporaryDirectory(prefix="openrl-vllm-lora-probe-") as tmp_dir:
    model_path = Path(tmp_dir) / "tiny-llama"
    write_tiny_llama_model(model_path)
    tensors = print_probe_metadata(model_path)
    if args.live_vllm:
      run_live_vllm_probe(model_path, tensors, args.device)
    if args.live_vllm_ipc:
      run_live_vllm_ipc_probe(model_path, tensors, args.device)
    if args.live_custom_lora:
      run_live_custom_lora_probe(model_path, tensors, args.device)


if __name__ == "__main__":
  main()
