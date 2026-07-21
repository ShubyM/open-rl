"""Exercise sampler rollouts driving live FFT updates and weight resync."""

import asyncio
import os
import tempfile
from pathlib import Path

import vllm
from test_vllm_weight_representation import (
  compare_fused_qkv,
  sampler_score,
  save_expected_qkv,
  score_residual,
  trainer_score,
  update_weights,
)
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from training.fft_trainer_worker import FFTConfig, FFTTrainingWorker
from training.trainer_worker import Datum, TensorData


def rollout_datum(prompt: list[int], rollout: tuple[int, float]) -> Datum:
  """Build a one-token importance-sampling datum from a sampler rollout."""
  token, logprob = rollout
  tokens = prompt + [token]
  prompt_padding = [0.0] * (len(prompt) - 1)
  return Datum(
    model_input=tokens[:-1],
    loss_fn_inputs={
      "target_tokens": TensorData(data=tokens[1:]),
      "weights": TensorData(data=prompt_padding + [1.0]),
      "logprobs": TensorData(data=prompt_padding + [logprob]),
      # Penalize the sampled token so each feedback step has a visible effect.
      "advantages": TensorData(data=prompt_padding + [-1.0]),
    },
  )


def train_from_rollout(trainer: FFTTrainingWorker, rollout: tuple[int, float], prompt: list[int], qkv_names: set[str]) -> float:
  """Run real importance-sampling backward/Adam while limiting the test patch to QKV."""
  assert trainer.model is not None
  result = trainer.forward_backward([rollout_datum(prompt, rollout)], "importance_sampling", model_id="weight-feedback-test")
  for name, parameter in trainer.model.named_parameters():
    if name not in qkv_names:
      parameter.grad = None
  trainer.optim_step(
    {
      "learning_rate": 0.05,
      "beta1": 0.0,
      "beta2": 0.0,
      "eps": 1e-8,
      "weight_decay": 0.0,
    },
    model_id="weight-feedback-test",
  )
  return float(result["metrics"]["loss:mean"])


def save_weights(trainer: FFTTrainingWorker, path: Path, sync_strategy: str) -> None:
  if sync_strategy == "delta":
    trainer.save_state_delta("weight-feedback-test", str(path))
  else:
    trainer.save_state("weight-feedback-test", str(path), kind="sampler")


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
    model_id="weight-feedback-test",
    config=FFTConfig(cpu_offload=False, weight_sync_strategy=sync_strategy, seed=0),
  )
  assert trainer.model is not None
  assert trainer.tokenizer is not None

  qkv_names = [f"model.layers.0.self_attn.{projection}_proj.weight" for projection in ("q", "k", "v")]
  qkv_name_set = set(qkv_names)
  parameters = dict(trainer.model.named_parameters())
  prompt = trainer.tokenizer.encode("The answer is", add_special_tokens=False)

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

  history: list[dict[str, object]] = []
  with tempfile.TemporaryDirectory() as temp_dir:
    base_qkv = Path(temp_dir) / "base-qkv.safetensors"
    save_expected_qkv(parameters, qkv_names, base_qkv)
    base_weights = (await engine.collective_rpc(compare_fused_qkv, args=(str(base_qkv),)))[0]
    assert base_weights["equal"], base_weights

    rollout = await sampler_score(engine, prompt, "feedback-base")
    score_residual(rollout, trainer_score(trainer.model, prompt))

    for step in (1, 2):
      loss = train_from_rollout(trainer, rollout, prompt, qkv_name_set)
      checkpoint = Path(temp_dir) / f"step-{step}"
      expected_qkv = Path(temp_dir) / f"step-{step}-qkv.safetensors"
      save_weights(trainer, checkpoint, sync_strategy)
      save_expected_qkv(parameters, qkv_names, expected_qkv)
      expected_score = trainer_score(trainer.model, prompt)

      await engine.sleep(level=1)
      await engine.wake_up(tags=["weights"])
      await update_weights(engine, str(checkpoint), sync_strategy)
      weight_comparison = (await engine.collective_rpc(compare_fused_qkv, args=(str(expected_qkv),)))[0]
      await engine.wake_up(tags=["kv_cache"])
      updated_sample = await sampler_score(engine, prompt, f"feedback-step-{step}")

      assert weight_comparison["equal"], weight_comparison
      residual = score_residual(updated_sample, expected_score)
      assert updated_sample != rollout, f"Feedback step {step} did not change sampler output: {rollout}"
      history.append(
        {
          "step": step,
          "rollout": rollout,
          "loss": loss,
          "updated_sample": updated_sample,
          "trainer_token": expected_score[0],
          "logprob_residual": residual,
          "weights": weight_comparison,
        }
      )
      rollout = updated_sample

  update_path = os.getenv("WEIGHT_UPDATE_PATH", "vllm-public") if sync_strategy == "delta" else "reload_weights"
  print("feedback comparison:", history)
  print(f"PASS ({sync_strategy}/{update_path}): sampler rollout -> FFT backward/Adam -> sampler resync, twice")


if __name__ == "__main__":
  asyncio.run(main())
