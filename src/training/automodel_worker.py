"""A trainer worker built on NVIDIA NeMo Automodel, run under torchrun.

Automodel builds a (dp, tp) FSDP2 mesh from the torchrun world; DP is what is
left after TP. Only the DP axis shards datums, since TP ranks cooperate on one
sequence and must see identical data. The shard_* hooks scope to that axis.
"""

import json
import math
import os
import shutil
import time
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer

from training.distributed import barrier, is_primary
from training.trainer_worker import BaseTrainerWorker, Datum

TMP_DIR = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
AUTOMODEL_TP = int(os.getenv("OPEN_RL_AUTOMODEL_TP", "1"))
AUTOMODEL_SEED = int(os.getenv("OPEN_RL_AUTOMODEL_SEED", "1234"))

# Rank 0 trains full parameters. Alpha 16 and the kaiming A init are PEFT's
# defaults, so an lr means the same thing here as on the FSDP worker.
AUTOMODEL_LORA_RANK = int(os.getenv("OPEN_RL_AUTOMODEL_LORA_RANK", "16"))
AUTOMODEL_LORA_ALPHA = int(os.getenv("OPEN_RL_AUTOMODEL_LORA_ALPHA", "16"))
# Matched as model.*.layers.*.<name>: the decoder projections, not a vision
# tower or an MTP head the samplers cannot place.
AUTOMODEL_LORA_TARGETS = os.getenv(
  "OPEN_RL_AUTOMODEL_LORA_TARGETS",
  "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,gate_proj,up_proj,down_proj",
)


def require_automodel():
  try:
    from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
  except ImportError as exc:
    raise RuntimeError(
      "OPEN_RL_TRAINER_BACKEND=automodel needs nemo-automodel, which is not a project dependency. "
      f"Run the trainer from the automodel image or scripts/setup_automodel_env.sh. Import failed with: {exc}"
    ) from exc
  return NeMoAutoModelForCausalLM


def is_dtensor(tensor: Any) -> bool:
  from torch.distributed.tensor import DTensor

  return isinstance(tensor, DTensor)


class AutomodelConfig(BaseModel):
  seed: int | None = None


class AutomodelTrainingWorker(BaseTrainerWorker):
  config_class = AutomodelConfig

  # FSDP2 reduces gradients inside backward, so every DP rank needs the same
  # number of passes.
  backward_runs_collectives = True

  def __init__(self):
    super().__init__()
    self.model: torch.nn.Module | None = None
    self.distributed_setup: Any = None
    self.device_mesh: Any = None
    self.base_model_name: str | None = None
    self.trainable_params: list[torch.nn.Parameter] = []
    self.optimizer: torch.optim.Optimizer | None = None
    self.checkpointer: Any = None
    self.is_lora = AUTOMODEL_LORA_RANK > 0

  def build_distributed_setup(self, base_model_name: str) -> Any:
    if not dist.is_initialized():
      raise RuntimeError("The Automodel backend runs under torchrun; torch.distributed is not initialized.")
    from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
    from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes

    world = dist.get_world_size()
    if world % AUTOMODEL_TP:
      raise RuntimeError(f"WORLD_SIZE={world} is not divisible by OPEN_RL_AUTOMODEL_TP={AUTOMODEL_TP}")
    strategy = FSDP2Config(activation_checkpointing=False, tp_plan=self.tensor_parallel_plan(base_model_name) if AUTOMODEL_TP > 1 else None)
    mesh_context = MeshContext.build(strategy, ParallelismSizes(tp_size=AUTOMODEL_TP), world_size=world)
    print(f"Automodel device mesh: DP={world // AUTOMODEL_TP} TP={AUTOMODEL_TP}")
    return DistributedSetup(mesh_context=mesh_context, strategy_config=strategy, activation_checkpointing=False)

  def tensor_parallel_plan(self, base_model_name: str) -> dict[str, Any]:
    """The HF text-model TP plan, prefixed the way the module tree is laid out.

    Automodel's automatic plan reads _tp_plan off the language model, which its
    native Qwen3.5 backbone does not carry. Layers missing from the plan (GDN)
    stay replicated.
    """
    from nemo_automodel.components.distributed.parallelizer import translate_to_torch_parallel_style
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    config = AutoConfig.from_pretrained(base_model_name)
    architectures = config.architectures or []
    prefix = "model.language_model" if architectures and architectures[0].endswith("ForConditionalGeneration") else "model"
    plan: dict[str, Any] = {f"{prefix}.embed_tokens": RowwiseParallel(input_layouts=Replicate())}
    for name, style_name in (getattr(config.get_text_config(), "base_model_tp_plan", None) or {}).items():
      try:
        style = translate_to_torch_parallel_style(style_name)
      except ValueError:
        # MoE-only entries match nothing on a dense checkpoint; skipping one
        # leaves that module replicated.
        continue
      if style is not None:
        plan[f"{prefix}.{name}"] = style
    # Replicated logits, so the base class computes logprobs from plain tensors.
    plan["lm_head"] = ColwiseParallel(output_layouts=Replicate())
    return plan

  def dp_group(self):
    return self.device_mesh["dp_shard"].get_group()

  def shard_rank(self) -> int:
    return 0 if self.device_mesh is None else self.device_mesh["dp_shard"].get_local_rank()

  def shard_count(self) -> int:
    return 1 if self.device_mesh is None else self.device_mesh["dp_shard"].size()

  def shard_all_reduce_max(self, value: int) -> int:
    tensor = torch.tensor([value], dtype=torch.long, device=self.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.dp_group())
    return int(tensor.item())

  def shard_all_reduce_sum(self, value: float) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=self.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.dp_group())
    return float(tensor.item())

  def shard_all_gather_object(self, value: Any) -> list[Any]:
    gathered: list[Any] = [None] * self.shard_count()
    dist.all_gather_object(gathered, value, group=self.dp_group())
    return gathered

  def build_peft_config(self) -> Any:
    from nemo_automodel.components._peft.lora import PeftConfig

    names = [name.strip() for name in AUTOMODEL_LORA_TARGETS.split(",") if name.strip()]
    return PeftConfig(
      target_modules=[f"model.*.layers.*.{name}" for name in names],
      dim=AUTOMODEL_LORA_RANK,
      alpha=AUTOMODEL_LORA_ALPHA,
      lora_A_init="kaiming",
      use_triton=False,
    )

  def load_base_model(self, base_model_name: str) -> None:
    if self.model is not None and self.base_model_name == base_model_name:
      return
    NeMoAutoModelForCausalLM = require_automodel()
    torch.cuda.set_device(self.device)
    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if self.distributed_setup is None:
      self.distributed_setup = self.build_distributed_setup(base_model_name)
      self.device_mesh = self.distributed_setup.mesh_context.device_mesh
    # from_pretrained applies LoRA before FSDP2 shards anything, loads the base
    # weights and freezes all but the adapters.
    self.model = NeMoAutoModelForCausalLM.from_pretrained(
      base_model_name,
      torch_dtype=torch.bfloat16,
      use_liger_kernel=False,
      distributed_setup=self.distributed_setup,
      peft_config=self.build_peft_config() if self.is_lora else None,
    )
    print(f"Loaded Automodel {base_model_name} (LoRA rank {AUTOMODEL_LORA_RANK}).")

  def create_model(self, base_model_name: str, model_id: str | None = None, config: AutomodelConfig | None = None) -> None:
    self.load_base_model(base_model_name)
    torch.manual_seed(config.seed if config is not None and config.seed is not None else AUTOMODEL_SEED)
    if not self.is_lora:
      for param in self.model.parameters():
        param.requires_grad_(True)
    self.trainable_params = [param for param in self.model.parameters() if param.requires_grad]
    if not self.trainable_params:
      raise ValueError("No trainable parameters found in the Automodel model")
    self.optimizer = None

  def forward_backward(
    self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None, forward_only: bool = False
  ) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    return super().forward_backward(self.model, data, loss_fn, loss_config, forward_only=forward_only)

  def optim_step(self, adam_params: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if self.optimizer is None:
      # foreach=False: the fused path cannot mix DTensors from different meshes.
      self.optimizer = torch.optim.AdamW(
        self.trainable_params,
        lr=adam_params.get("learning_rate", 1e-4),
        betas=(adam_params.get("beta1", 0.9), adam_params.get("beta2", 0.95)),
        eps=adam_params.get("eps", 1e-12),
        weight_decay=adam_params.get("weight_decay", 0.0),
        foreach=False,
      )
    if adam_params.get("learning_rate") is not None:
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = adam_params["learning_rate"]

    max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
    total_norm = self.clip_gradients(max_grad_norm if max_grad_norm > 0 else math.inf)
    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
    return {"metrics": {"grad_norm:mean": self.sanitize_float(total_norm)}}

  def clip_gradients(self, max_grad_norm: float) -> float:
    """Global grad norm over DTensors that may live on different meshes.

    clip_grad_norm_ stacks per-tensor norms, which fails when TP-sharded and
    replicated parameters sit side by side. A DTensor's vector_norm already
    reduces over its own placements, so only the stack of scalars is local.
    """
    norms = []
    for param in self.trainable_params:
      if param.grad is None:
        continue
      norm = torch.linalg.vector_norm(param.grad.detach().float())
      norms.append(norm.full_tensor() if is_dtensor(norm) else norm)
    if not norms:
      return 0.0
    total_norm = float(torch.linalg.vector_norm(torch.stack(norms)))
    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
      for param in self.trainable_params:
        if param.grad is not None:
          param.grad.mul_(clip_coef)
    return total_norm

  def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("Sampling from the Automodel trainer is unsupported; use the vLLM sampler.")

  def get_checkpointer(self) -> Any:
    """Automodel's Checkpointer gathers DTensor shards, maps native keys back to
    the hub layout, and writes a PEFT adapter that vLLM loads."""
    if self.checkpointer is None:
      from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

      config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=os.path.join(TMP_DIR, "automodel"),
        model_save_format="safetensors",
        save_consolidated=not self.is_lora,
        is_peft=self.is_lora,
        model_repo_id=self.base_model_name,
      )
      self.checkpointer = Checkpointer(config, dp_rank=self.shard_rank(), tp_rank=self.device_mesh["tp"].get_local_rank(), pp_rank=0)
    return self.checkpointer

  def write_weights(self, path: str) -> None:
    """Write the adapter (or the consolidated model) as plain files in path.

    Every rank calls this because the gathers are collective. The checkpointer
    nests its output under model/, which rank 0 lifts into path.
    """
    self.get_checkpointer().save_model(
      self.model, weights_path=path, peft_config=self.build_peft_config() if self.is_lora else None, tokenizer=self.tokenizer
    )
    if is_primary():
      model_dir = os.path.join(path, "model")
      source = model_dir if self.is_lora else os.path.join(model_dir, "consolidated")
      for entry in os.listdir(source):
        os.replace(os.path.join(source, entry), os.path.join(path, entry))
      shutil.rmtree(model_dir, ignore_errors=True)
    barrier()

  def write_staged(self, path: str, metadata: dict[str, Any] | None = None) -> None:
    """Write into a staging dir and rename it over path, so a reader never
    sees a half-written directory."""
    staging, previous = f"{path}.staging-{os.getpid()}", f"{path}.previous-{os.getpid()}"
    if is_primary():
      shutil.rmtree(staging, ignore_errors=True)
      os.makedirs(staging)
    barrier()
    self.write_weights(staging)
    if is_primary():
      if metadata is not None:
        with open(os.path.join(staging, "metadata.json"), "w") as f:
          json.dump(metadata, f)
      if os.path.exists(path):
        os.rename(path, previous)
      os.rename(staging, path)
      shutil.rmtree(previous, ignore_errors=True)
    barrier()

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    # Weights only for now; the optimizer state is not saved.
    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": False,
      "model_id": model_id,
      "timestamp": time.time(),
    }
    self.write_staged(state_path, metadata)
    return {"path": state_path}

  def load_from_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError("The Automodel trainer does not load checkpoints yet.")

  def save_for_sampler(self, model_id: str, alias: str | None, ref: str | None) -> str | None:
    """Write the adapter where the LoRA sampler hot-loads it, peft/<id>/<id>."""
    if not self.is_lora:
      raise NotImplementedError("The Automodel trainer publishes sampler weights for LoRA only.")
    self.write_staged(os.path.join(TMP_DIR, "peft", model_id, model_id))
    return None
