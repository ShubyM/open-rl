"""Check real FFT delta/full checkpoints against vLLM weights and sampling."""

import asyncio
import gc
import os
import tempfile
from pathlib import Path

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["ENABLE_GRADIENT_CHECKPOINTING"] = "0"

import torch
import vllm
from safetensors.torch import load_file, save_file
from vllm import SamplingParams
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

import server.delta_weight_transfer_engine  # noqa: F401
from training.fft_trainer_worker import FFTConfig, FFTTrainingWorker


def apply_delta(worker, delta_dir: str):
  """Run PR #154's private worker/model-runner path as the control."""
  transfer = worker.weight_transfer_engine
  assert type(transfer).__name__ == "DeltaSnapshotWeightTransferEngine"
  update = transfer.parse_update_info({"target_weights_path": delta_dir})
  return transfer.receive_weights(update, worker.model_runner.model.load_weights)


async def update_weights(engine, weights_dir: str, sync_strategy: str) -> None:
  if sync_strategy == "full":
    await engine.collective_rpc("reload_weights", kwargs={"weights_path": weights_dir})
    return

  update_path = os.getenv("WEIGHT_UPDATE_PATH", "pr-private")
  if update_path == "pr-private":
    await engine.collective_rpc(apply_delta, args=(weights_dir,))
    return
  if update_path == "vllm-public":
    update = WeightTransferUpdateRequest(
      update_info={
        "target_weights_path": weights_dir,
        "base_model_path": os.environ["BASE_MODEL"],
      }
    )
    await engine.start_weight_update()
    await engine.update_weights(update)
    await engine.finish_weight_update()
    return
  raise ValueError(f"Unknown WEIGHT_UPDATE_PATH: {update_path}")


def compare_fused_qkv(worker, expected_path: str) -> dict[str, object]:
  """Compare the live fused vLLM QKV parameter with the trainer tensors."""
  model = worker.model_runner.model
  parameters = dict(model.named_parameters())
  fused_name = next(name for name in parameters if name.endswith("model.layers.0.self_attn.qkv_proj.weight"))
  fused = parameters[fused_name].detach()
  expected = load_file(expected_path, device="cpu")
  packed = torch.cat([expected[name] for name in ("q", "k", "v")], dim=0).to(device=fused.device, dtype=fused.dtype)
  difference = (fused - packed).abs().float()
  return {
    "equal": torch.equal(fused, packed),
    "fused_shape": list(fused.shape),
    "max_abs_difference": float(difference.max()),
    "unequal_elements": int(torch.count_nonzero(difference)),
  }


def save_expected_qkv(parameters: dict[str, torch.nn.Parameter], names: list[str], path: Path) -> None:
  save_file(
    {projection: parameters[name].detach().cpu().contiguous() for projection, name in zip(("q", "k", "v"), names, strict=True)},
    path,
  )


def trainer_score(model, prompt: list[int]) -> tuple[int, torch.Tensor]:
  was_training = model.training
  model.eval()
  with torch.inference_mode():
    scores = model(torch.tensor([prompt], device=model.device)).logits[0, -1].float().log_softmax(-1).cpu()
  model.train(was_training)
  return int(scores.argmax()), scores


def train_and_save(
  trainer: FFTTrainingWorker,
  gradients: dict[str, torch.Tensor],
  sign: float,
  weights_dir: Path,
  sync_strategy: str,
) -> None:
  assert trainer.model is not None
  parameters = dict(trainer.model.named_parameters())
  for name, gradient in gradients.items():
    parameters[name].grad = gradient.mul(sign)
  trainer.optim_step(
    {
      "learning_rate": 0.05,
      "beta1": 0.0,
      "beta2": 0.0,
      "eps": 1e-8,
      "weight_decay": 0.0,
    },
    model_id="weight-representation-test",
  )
  if sync_strategy == "delta":
    trainer.save_state_delta("weight-representation-test", str(weights_dir))
  else:
    trainer.save_state("weight-representation-test", str(weights_dir), kind="sampler")


async def sampler_score(engine, prompt: list[int], revision: str) -> tuple[int, float]:
  final = None
  async for output in engine.generate(
    {"prompt_token_ids": prompt},
    SamplingParams(temperature=0, max_tokens=1, logprobs=1),
    request_id=f"weight-representation-{revision}",
  ):
    final = output
  assert final is not None
  token = final.outputs[0].token_ids[0]
  return token, final.outputs[0].logprobs[0][token].logprob


def score_residual(actual: tuple[int, float], expected: tuple[int, torch.Tensor]) -> float:
  actual_token, actual_logprob = actual
  expected_token, expected_scores = expected
  assert actual_token == expected_token, (actual_token, expected_token)
  expected_logprob = float(expected_scores[actual_token])
  return actual_logprob - expected_logprob


async def main() -> None:
  assert vllm.__version__.startswith("0.25.1"), vllm.__version__
  model_name = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
  sync_strategy = os.getenv("WEIGHT_SYNC_STRATEGY", "delta").lower()
  if sync_strategy not in ("delta", "full"):
    raise ValueError(f"Unknown WEIGHT_SYNC_STRATEGY: {sync_strategy}")
  os.environ["BASE_MODEL"] = model_name
  trainer = FFTTrainingWorker()
  trainer.create_model(
    model_name,
    model_id="weight-representation-test",
    config=FFTConfig(cpu_offload=False, weight_sync_strategy=sync_strategy, seed=0),
  )
  assert trainer.model is not None
  assert trainer.tokenizer is not None

  # HF stores Q/K/V separately; vLLM stores them in a fused QKV parameter.
  names = [f"model.layers.0.self_attn.{projection}_proj.weight" for projection in ("q", "k", "v")]
  parameters = dict(trainer.model.named_parameters())
  torch.manual_seed(0)
  gradients = {name: torch.randn_like(parameters[name]) for name in names}

  prompt = trainer.tokenizer.encode("The answer is", add_special_tokens=False)
  expected_base = trainer_score(trainer.model, prompt)

  with tempfile.TemporaryDirectory() as temp_dir:
    first_weights = Path(temp_dir) / "step-1"
    second_weights = Path(temp_dir) / "step-2"
    base_qkv = Path(temp_dir) / "base-qkv.safetensors"
    first_qkv = Path(temp_dir) / "step-1-qkv.safetensors"
    second_qkv = Path(temp_dir) / "step-2-qkv.safetensors"

    save_expected_qkv(parameters, names, base_qkv)
    train_and_save(trainer, gradients, 1.0, first_weights, sync_strategy)
    expected_first = trainer_score(trainer.model, prompt)
    save_expected_qkv(parameters, names, first_qkv)
    train_and_save(trainer, gradients, -1.0, second_weights, sync_strategy)
    expected_second = trainer_score(trainer.model, prompt)
    save_expected_qkv(parameters, names, second_qkv)

    # Keep the real trainer-produced deltas, but release its GPU model before
    # constructing vLLM so this test also fits on smaller development GPUs.
    trainer.model.to("cpu")
    trainer.optimizer = None
    gradients.clear()
    gc.collect()
    torch.cuda.empty_cache()

    engine_kwargs = {
      "model": model_name,
      "dtype": "bfloat16",
      "enable_sleep_mode": True,
      "enforce_eager": True,
      "max_model_len": 64,
      "gpu_memory_utilization": 0.35,
    }
    if sync_strategy == "delta":
      engine_kwargs["weight_transfer_config"] = WeightTransferConfig(backend="delta_snapshot")
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))
    base_weights = (await engine.collective_rpc(compare_fused_qkv, args=(str(base_qkv),)))[0]
    baseline = await sampler_score(engine, prompt, "base")

    await engine.sleep(level=1)
    await engine.wake_up(tags=["weights"])
    await update_weights(engine, str(first_weights), sync_strategy)
    first_weight_comparison = (await engine.collective_rpc(compare_fused_qkv, args=(str(first_qkv),)))[0]
    await engine.wake_up(tags=["kv_cache"])
    first = await sampler_score(engine, prompt, "step-1")

    await engine.sleep(level=1)
    await engine.wake_up(tags=["weights"])
    await update_weights(engine, str(second_weights), sync_strategy)
    second_weight_comparison = (await engine.collective_rpc(compare_fused_qkv, args=(str(second_qkv),)))[0]
    await engine.wake_up(tags=["kv_cache"])
    second = await sampler_score(engine, prompt, "step-2")

  base_residual = score_residual(baseline, expected_base)
  first_residual = score_residual(first, expected_first)
  second_residual = score_residual(second, expected_second)
  print(
    "comparison:",
    {
      "base": (baseline, expected_base[0], base_residual),
      "step-1": (first, expected_first[0], first_residual),
      "step-2": (second, expected_second[0], second_residual),
      "weights": {"base": base_weights, "step-1": first_weight_comparison, "step-2": second_weight_comparison},
    },
  )
  # Transformers and vLLM need not have identical absolute logits. Verify the
  # representation conversion directly against the live fused vLLM parameter.
  assert base_weights["equal"], base_weights
  assert first_weight_comparison["equal"], first_weight_comparison
  assert second_weight_comparison["equal"], second_weight_comparison
  assert first != baseline, "first FFT step did not change the sampled score"
  assert second != first, "second FFT step did not change the sampled score"
  update_path = os.getenv("WEIGHT_UPDATE_PATH", "pr-private") if sync_strategy == "delta" else "reload_weights"
  print(f"PASS ({sync_strategy}/{update_path}): FFT trainer -> fused vLLM QKV -> second FFT switch matches")


if __name__ == "__main__":
  asyncio.run(main())
