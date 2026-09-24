"""One trainer worker for LoRA and full fine-tuning.

The request processor calls one method per Tinker training command. The
worker holds the base HF module and one TrainedModel per model_id. A LoRA
model_id is a PeftModel of its own over the base's tensors, so tenants never
switch adapters; a full fine-tuning model_id is the base module itself. The
optimizer state is plain named tensors next to the module, so it sleeps,
saves and compares the way the weights do.
"""

import itertools
import json
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from training import losses
from training.lora import lora_model, own_tree
from training.types import Datum, FFTConfig, LoraConfig

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"


def default_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def load_module(source: str, device: torch.device) -> tuple[torch.nn.Module, PreTrainedTokenizerBase | None]:
  """An HF causal LM from the hub or a checkpoint directory, spread over every visible GPU when there are several."""
  num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
  device_map = "auto" if num_gpus > 1 else str(device)
  print(f"Loading {source} (device map: {device_map}, visible GPUs: {num_gpus})...")
  dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
  module = AutoModelForCausalLM.from_pretrained(source, dtype=dtype, device_map=device_map)
  if ENABLE_GRADIENT_CHECKPOINTING:
    try:
      module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
      module.enable_input_require_grads()
    except Exception as exc:
      print(f"Failed to enable gradient checkpointing: {exc}")
  try:
    tokenizer = AutoTokenizer.from_pretrained(source)
  except Exception as exc:
    # The tokenizer only supplies the pad id here, and padding sits under the attention mask anyway.
    print(f"No tokenizer at {source} ({exc}); padding with token id 0.")
    tokenizer = None
  print(f"Loaded {source}.")
  return module, tokenizer


def token_logprobs(module: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
  """Selected target logprobs with shape [batch, max_target_len]."""
  outputs = module(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
  logits = outputs.logits[:, : target_token_ids.shape[1], :]
  return torch.nn.functional.log_softmax(logits, dim=-1).gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)


def sanitize_float(val: float) -> float:
  if math.isinf(val):
    return -9999.0 if val < 0 else 9999.0
  if math.isnan(val):
    return 0.0
  return val


def clip_grad_norm(grads: list[torch.Tensor], max_norm: float) -> float:
  """Scale grads in place so their global norm is at most max_norm and return the norm they had. Matches torch.nn.utils.clip_grad_norm_."""
  if not grads:
    return 0.0
  with torch.no_grad():
    total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(grad) for grad in grads]))
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for grad in grads:
      grad.mul_(clip_coef.to(grad.device))
  return float(total_norm)


@dataclass
class TrainedModel:
  """What one model_id owns: its module and its AdamW state."""

  module: torch.nn.Module
  adam_m: dict[str, torch.Tensor] = field(default_factory=dict)
  adam_v: dict[str, torch.Tensor] = field(default_factory=dict)
  adam_step: int = 0

  @property
  def is_lora(self) -> bool:
    return isinstance(self.module, PeftModel)

  @property
  def params(self) -> dict[str, torch.nn.Parameter]:
    """The parameters training changes. Their gradients are their .grad."""
    return {name: param for name, param in self.module.named_parameters() if param.requires_grad}

  def tensors(self) -> Iterator[torch.Tensor]:
    yield from self.module.parameters()
    yield from self.module.buffers()
    for param in self.module.parameters():
      if param.grad is not None:
        yield param.grad
    yield from self.adam_m.values()
    yield from self.adam_v.values()


class TrainerWorker:
  def __init__(self):
    self.base: torch.nn.Module | None = None
    self.base_name: str | None = None
    self.tokenizer: PreTrainedTokenizerBase | None = None
    self.device = default_device()
    self.models: dict[str, TrainedModel] = {}
    # Whether sleep() offloads to the host between time-slicer turns. FFTConfig can turn it off.
    self.cpu_offload = True
    # Pinned host buffers by tensor id, allocated on the first sleep and reused after.
    self.host: dict[int, tuple[torch.device, torch.Tensor]] = {}

  @property
  def pad_token_id(self) -> int:
    if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
      return self.tokenizer.pad_token_id
    return 0

  def load_base_model(self, base_model_name: str) -> None:
    if self.base is not None and self.base_name == base_model_name:
      return
    self.base, self.tokenizer = load_module(base_model_name, self.device)
    self.base_name = base_model_name

  def create_model(self, base_model_name: str, model_id: str, config: LoraConfig | FFTConfig | None = None) -> None:
    """A fresh model_id on base_model_name. With a LoraConfig it is an adapter; otherwise it is the base module itself,
    which is why a full fine-tuning process serves one model_id."""
    self.load_base_model(base_model_name)
    if config is not None and config.seed is not None:
      torch.manual_seed(config.seed)
    lora_config = config if isinstance(config, LoraConfig) else None
    if lora_config is not None:
      module = lora_model(self.base, model_id, lora_config)
    else:
      self.base.requires_grad_(True)
      module = self.base
    module.train()
    self.models[model_id] = TrainedModel(module)
    print(f"Created '{model_id}' on {base_model_name} ({f'LoRA rank {lora_config.rank}' if lora_config else 'full fine-tuning'}).")

  # -- checkpoints -----------------------------------------------------------------

  def save_state(self, model_id: str, path: str, include_optimizer: bool = False, kind: str = "state") -> None:
    """The module in its own library's layout (PEFT's under <path>/<model_id>/, HF's at <path>), the AdamW state beside it when asked."""
    model = self.models[model_id]
    os.makedirs(path, exist_ok=True)
    model.module.save_pretrained(path)
    if not model.is_lora and self.tokenizer is not None:
      self.tokenizer.save_pretrained(path)

    has_adam = include_optimizer and bool(model.adam_m)
    if has_adam:
      adam = {f"m.{name}": tensor.detach().contiguous().cpu() for name, tensor in model.adam_m.items()}
      adam.update({f"v.{name}": tensor.detach().contiguous().cpu() for name, tensor in model.adam_v.items()})
      save_file(adam, os.path.join(path, "adam.safetensors"), metadata={"step": str(model.adam_step)})

    metadata = {
      "base_model": self.base_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": has_adam,
      "model_id": model_id,
      "timestamp": time.time(),
    }
    with open(os.path.join(path, "metadata.json"), "w") as f:
      json.dump(metadata, f)
    print(f"Saved '{model_id}' to {path}")

  def load_state(self, model_id: str, path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    """model_id from a directory save_state wrote, or one the PEFT workers before it wrote. Its metadata names the base model."""
    metadata = read_metadata(path)
    adapter_dir = lora_adapter_dir(path, metadata.get("model_id"))
    if adapter_dir is not None:
      self.load_base_model(metadata["base_model"])
      module = PeftModel.from_pretrained(own_tree(self.base), adapter_dir, adapter_name=model_id, is_trainable=True)
    else:
      self.base, self.tokenizer = load_module(path, self.device)
      self.base_name = metadata["base_model"]
      self.base.requires_grad_(True)
      module = self.base
    module.train()
    model = TrainedModel(module)
    if restore_optimizer:
      read_adam(model, path)
    self.models[model_id] = model
    print(f"Loaded '{model_id}' from {path}" + (" with its optimizer" if model.adam_m else ""))
    return {"model_id": model_id, "base_model": metadata["base_model"]}

  # -- the step --------------------------------------------------------------------

  def forward_backward(
    self, model_id: str, data: list[Datum], loss_fn: str, loss_config: dict | None = None, forward_only: bool = False
  ) -> dict[str, Any]:
    """Run the loss over data for model_id, accumulating gradients on its params, and return Tinker-shaped outputs.

    With ``forward_only`` (TrainingClient.forward) the pass runs without
    autograd and in eval mode, and no gradient is accumulated: the cookbook's
    NLL evaluator calls it on held-out data right before a training step, so
    a backward here would fold the test set into the next optim_step.
    """
    module = self.models[model_id].module
    loss_fn_outputs: list[dict[str, Any] | None] = [None] * len(data)

    if forward_only:
      module.eval()
    else:
      module.train()

    with torch.set_grad_enabled(not forward_only):
      total_loss = self.run_batches(module, data, loss_fn, loss_config, forward_only, loss_fn_outputs)
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    return self.finish(data, loss_fn_outputs, total_loss)

  def run_batches(
    self,
    module: torch.nn.Module,
    data: list[Datum],
    loss_fn: str,
    loss_config: dict | None,
    forward_only: bool,
    loss_fn_outputs: list[dict[str, Any] | None],
  ) -> float:
    """Run every batch, fill ``loss_fn_outputs`` in place, and return the summed loss."""
    total_loss = 0.0
    for batch in self.make_training_batches(data):
      batch_indices = [idx for idx, _ in batch]
      batch_data = [datum for _, datum in batch]

      input_ids, attention_mask, input_lengths = self.pad_model_inputs(batch_data)
      target_token_ids, weights, lengths = self.pad_targets_and_weights(batch_data, input_lengths)
      target_logprobs = token_logprobs(module, input_ids, attention_mask, target_token_ids)

      match loss_fn:
        case "cross_entropy":
          elementwise_loss = losses.cross_entropy_loss(target_logprobs, weights)
        case "importance_sampling":
          old_logprobs = self.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32)
          advantages = self.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32)
          elementwise_loss = losses.importance_sampling_loss(target_logprobs, weights, old_logprobs, advantages)
        case "ppo":
          old_logprobs = self.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32)
          advantages = self.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32)
          elementwise_loss = losses.ppo_loss(target_logprobs, weights, old_logprobs, advantages, loss_config)
        case _:
          raise NotImplementedError(f"Loss {loss_fn} not supported")

      loss = elementwise_loss.sum(dim=1).sum()
      if not forward_only:
        loss.backward()
      total_loss += loss.item()

      detached_logprobs = target_logprobs.detach().cpu()
      for row, original_idx in enumerate(batch_indices):
        logprobs_list = detached_logprobs[row, : lengths[row]].tolist()
        logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
        loss_fn_outputs[original_idx] = {"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}}
    return total_loss

  def finish(self, data: list[Datum], loss_fn_outputs: list[dict[str, Any] | None], total_loss: float) -> dict[str, Any]:
    mean_loss = total_loss / max(1, len(data))
    completed_loss_fn_outputs = []
    for output in loss_fn_outputs:
      if output is None:
        raise RuntimeError("forward_backward did not produce one loss_fn_output per input datum")
      completed_loss_fn_outputs.append(output)

    return {
      "metrics": {"loss:mean": sanitize_float(mean_loss), "loss:sum": sanitize_float(total_loss)},
      "loss_fn_outputs": completed_loss_fn_outputs,
      "loss_fn_output_type": "ArrayRecord",
    }

  def optim_step(self, model_id: str, adam_params: dict[str, Any]) -> dict[str, Any]:
    """One AdamW step on model_id's params from their accumulated grads, which it then clears.

    This is torch.optim.AdamW's single-tensor update written over the params,
    so the optimizer state is plain named tensors that sleep, save and
    compare like the weights do.
    """
    model = self.models[model_id]
    lr = adam_params.get("learning_rate", 1e-4)
    beta1 = adam_params.get("beta1", 0.9)
    beta2 = adam_params.get("beta2", 0.95)
    eps = adam_params.get("eps", 1e-12)
    weight_decay = adam_params.get("weight_decay", 0.0)
    max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
    if max_grad_norm <= 0.0:
      max_grad_norm = math.inf

    params = model.params
    total_norm = clip_grad_norm([param.grad for param in params.values() if param.grad is not None], max_grad_norm)

    with torch.no_grad():
      model.adam_step += 1
      step = model.adam_step
      bias_correction1 = 1 - beta1**step
      bias_correction2_sqrt = math.sqrt(1 - beta2**step)
      step_size = lr / bias_correction1
      for name, param in params.items():
        grad = param.grad
        if grad is None:
          continue
        m = model.adam_m.setdefault(name, torch.zeros_like(param))
        v = model.adam_v.setdefault(name, torch.zeros_like(param))
        param.mul_(1 - lr * weight_decay)
        m.lerp_(grad, 1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        param.addcdiv_(m, (v.sqrt() / bias_correction2_sqrt).add_(eps), value=-step_size)
        param.grad = None

    return {"metrics": {"grad_norm:mean": sanitize_float(total_norm)}}

  # -- the time-slicer turn ------------------------------------------------------------

  def tensors(self) -> Iterator[torch.Tensor]:
    """Every tensor this worker keeps on the accelerator, once each. LoRA tenants share the base's."""
    seen: set[int] = set()
    base = itertools.chain(self.base.parameters(), self.base.buffers()) if self.base is not None else ()
    for tensor in itertools.chain(base, *(model.tensors() for model in self.models.values())):
      if id(tensor) not in seen:
        seen.add(id(tensor))
        yield tensor

  def sleep(self) -> None:
    """Copy every device tensor to pinned host memory and give the device memory back.

    The copies are issued together and synchronized once, and only then is the
    device storage dropped, so a failed copy never loses a tensor.
    """
    if not self.cpu_offload or not torch.cuda.is_available():
      return
    started = time.perf_counter()
    moved = []
    for tensor in self.tensors():
      if tensor.device.type != "cuda":
        continue
      entry = self.host.get(id(tensor))
      if entry is None or entry[1].shape != tensor.shape or entry[1].dtype != tensor.dtype:
        entry = self.host[id(tensor)] = (tensor.device, torch.empty_like(tensor, device="cpu", pin_memory=True))
      entry[1].copy_(tensor.data, non_blocking=True)
      moved.append((tensor, entry[1]))
    torch.cuda.synchronize()
    for tensor, host in moved:
      tensor.data = host
    torch.cuda.empty_cache()
    print(f"[Trainer] Offloaded {len(moved)} tensors to pinned host memory in {(time.perf_counter() - started) * 1000:.1f} ms.")

  def wake_up(self) -> None:
    """Bring slept tensors back to the device each one came from."""
    if not self.cpu_offload or not torch.cuda.is_available():
      return
    started = time.perf_counter()
    for tensor in self.tensors():
      entry = self.host.get(id(tensor))
      if entry is not None and tensor.device.type == "cpu":
        tensor.data = tensor.data.to(entry[0], non_blocking=True)
    torch.cuda.synchronize()
    print(f"[Trainer] Reloaded tensors to the device in {(time.perf_counter() - started) * 1000:.1f} ms.")

  # -- batching --------------------------------------------------------------------

  def make_training_batches(self, data: list[Datum]) -> list[list[tuple[int, Datum]]]:
    """Group examples for the single padded forward/backward path."""
    if len(data) <= 1:
      return [[(idx, datum)] for idx, datum in enumerate(data)]

    token_budget = int(os.getenv("OPEN_RL_TRAIN_TOKEN_BUDGET", "0"))

    if token_budget <= 0:
      return [[(idx, datum)] for idx, datum in enumerate(data)]

    ordered_data = sorted(enumerate(data), key=lambda item: len(item[1].model_input))
    batches: list[list[tuple[int, Datum]]] = []
    batch: list[tuple[int, Datum]] = []
    batch_max_len = 0

    for item in ordered_data:
      length = len(item[1].model_input)
      next_max_len = max(batch_max_len, length)
      next_size = len(batch) + 1
      over_token_budget = next_max_len * next_size > token_budget

      if batch and over_token_budget:
        batches.append(batch)
        batch = []
        batch_max_len = 0

      batch.append(item)
      batch_max_len = max(batch_max_len, length)

    if batch:
      batches.append(batch)

    return batches

  def pad_sequences(
    self, sequences: list[list[int] | list[float]], lengths: list[int], dtype: torch.dtype, pad_value: int | float = 0
  ) -> torch.Tensor:
    """Return padded values with shape [batch, max(lengths)]."""
    padded = torch.full((len(sequences), max(lengths)), pad_value, dtype=dtype, device=self.device)
    for row, sequence in enumerate(sequences):
      length = lengths[row]
      padded[row, :length] = padded.new_tensor(sequence[:length])
    return padded

  def pad_model_inputs(self, data: list[Datum]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return input_ids and attention_mask with shape [batch, max_input_len]."""
    batch_size = len(data)
    input_lengths = [len(datum.model_input) for datum in data]
    max_input_len = max(input_lengths)

    input_ids = self.pad_sequences([datum.model_input for datum in data], input_lengths, torch.long, self.pad_token_id)
    attention_mask = input_ids.new_zeros((batch_size, max_input_len))
    for row, input_len in enumerate(input_lengths):
      attention_mask[row, :input_len] = 1

    return input_ids, attention_mask, input_lengths

  def pad_targets_and_weights(self, data: list[Datum], input_lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return target_token_ids and weights with shape [batch, max_target_len]."""
    batch_size = len(data)
    target_lengths = [len(datum.loss_fn_inputs["target_tokens"].data) for datum in data]
    lengths = [min(input_lengths[row], target_lengths[row]) for row in range(batch_size)]
    target_token_ids = self.pad_sequences([datum.loss_fn_inputs["target_tokens"].data for datum in data], lengths, torch.long)
    weight_sequences = [
      datum.loss_fn_inputs["weights"].data if "weights" in datum.loss_fn_inputs else [1.0] * target_lengths[row] for row, datum in enumerate(data)
    ]
    weights = self.pad_sequences(weight_sequences, lengths, torch.float32)

    return target_token_ids, weights, lengths


# -- checkpoint files ------------------------------------------------------------------


def read_metadata(path: str) -> dict[str, Any]:
  metadata_path = os.path.join(path, "metadata.json")
  if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"No metadata.json found at {path}")
  with open(metadata_path) as f:
    metadata = json.load(f)
  if not metadata.get("base_model"):
    raise ValueError(f"metadata.json at {path} missing base_model")
  return metadata


def lora_adapter_dir(path: str, saved_model_id: str | None) -> str | None:
  """Where PEFT put the adapter, under the saved adapter's name or at the root, or None for a full checkpoint."""
  candidates = [os.path.join(path, saved_model_id)] if saved_model_id else []
  candidates.append(path)
  return next((d for d in candidates if os.path.exists(os.path.join(d, "adapter_config.json"))), None)


def read_adam(model: TrainedModel, path: str) -> None:
  """AdamW state from adam.safetensors, or from the torch optimizer.pt the PEFT-based workers wrote."""
  params = model.params
  adam_path = os.path.join(path, "adam.safetensors")
  torch_path = os.path.join(path, "optimizer.pt")
  if os.path.exists(adam_path):
    with safe_open(adam_path, "pt") as f:
      model.adam_step = int(f.metadata()["step"])
    for key, tensor in load_file(adam_path).items():
      kind, name = key.split(".", 1)
      (model.adam_m if kind == "m" else model.adam_v)[name] = tensor.to(params[name].device)
  elif os.path.exists(torch_path):
    # torch.optim.AdamW state is indexed by parameter position; PEFT's parameters()
    # run in module order, the same order params does.
    state = torch.load(torch_path, map_location="cpu", weights_only=False)["state"]
    names = list(params)
    if len(state) != len(names):
      raise ValueError(f"{torch_path} holds {len(state)} parameter states for {len(names)} trainable parameters")
    for index, name in enumerate(names):
      model.adam_m[name] = state[index]["exp_avg"].to(params[name].device)
      model.adam_v[name] = state[index]["exp_avg_sq"].to(params[name].device)
      model.adam_step = int(state[index]["step"])
