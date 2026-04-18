"""Text-to-SQL SFT + RL recipe.

See `docs/guides/text-to-sql.md` for direct `uv run` commands.

Common phase modes:
- `phase=full`: run SFT, then RL
- `phase=sft_only`: stop after SFT
- `phase=rl_only`: skip SFT and run RL, optionally from `resume_state_path`
"""

from __future__ import annotations

import asyncio
import logging
import os
import statistics
import sys
from pathlib import Path
from typing import Any, cast

import chz
import tinker
from texttosql_grpo_utils import load_example_splits, shuffled_batches
from texttosql_rewards import compute_sql_reward
from texttosql_sft import (
  BASE_URL,
  DATASET,
  clean_sql_for_execution,
  evaluate,
  normalize_sql,
  require_server,
)
from tinker import types
from tinker_cookbook.utils import ml_log
from transformers import AutoTokenizer

LOG_DIR = Path(__file__).resolve().parent / "artifacts" / "texttosql_sft_grpo_{preset}"

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


# ---------------------------------------------------------------------------
# Config + presets
# ---------------------------------------------------------------------------
@chz.chz
class Config:
  # Model / runtime
  base_model: str
  tokenizer_name: str
  rank: int = 32
  phase: str = "full"  # "full" | "sft_only" | "rl_only"
  prompt_format: str = "plain_sql_completion"
  base_url: str = os.getenv("TINKER_BASE_URL") or os.getenv("OPEN_RL_BASE_URL") or BASE_URL
  seed: int = 30
  grad_clip_norm: float = 0.3
  log_dir: str = str(LOG_DIR)

  # Dataset / evaluation
  dataset_name: str = DATASET
  dataset_limit: int = 12_500
  train_limit: int = 100
  rl_train_limit: int = 64
  eval_limit: int = 25
  eval_max_tokens: int = 64

  # SFT phase
  sft_steps: int = 100
  sft_batch_size: int = 1
  sft_learning_rate: float = 5e-5
  sft_eval_every: int = 100

  # RL phase
  rl_steps: int = 40
  rl_prompts_per_step: int = 4
  rl_samples_per_prompt: int = 4
  rl_learning_rate: float = 1e-5
  rl_temperature: float = 0.8
  rl_max_tokens: int = 64
  rl_eval_every: int = 10
  rl_loss_fn: str = "grpo"
  rl_clip_range: float = 0.2
  rl_kl_coeff: float = 0.1

  # Reward shaping
  compile_reward: float = 0.25
  match_reward: float = 1.0
  compile_error_penalty: float = -0.5
  similarity_reward: float = 0.0

  # Checkpointing
  resume_state_path: str = ""
  resume_with_optimizer: bool = False
  save_sft_state_name: str = ""
  save_final_state_name: str = ""


GEMMA4_E2B = {"base_model": "google/gemma-4-e2b", "tokenizer_name": "google/gemma-4-e2b"}

PRESETS = {
  name: chz.Blueprint(Config).apply(overrides, layer_name=name)
  for name, overrides in {
    "gemma4_e2b_rl_recipe": {
      **GEMMA4_E2B,
      "rl_eval_every": 4,
      "rl_loss_fn": "importance_sampling",
      "rl_samples_per_prompt": 6,
      # Keep a small compile signal so RL does not collapse into pure string mimicry.
      "match_reward": 2.0,
      "compile_error_penalty": -0.25,
    },
    "gemma4_e2b_smoke": {
      **GEMMA4_E2B,
      "rank": 16,
      "train_limit": 16,
      "rl_train_limit": 8,
      "eval_limit": 4,
      "sft_steps": 4,
      "sft_eval_every": 2,
      "rl_steps": 2,
      "rl_prompts_per_step": 2,
      "rl_samples_per_prompt": 3,
      "rl_temperature": 0.7,
      "rl_eval_every": 1,
    },
  }.items()
}


# ---------------------------------------------------------------------------
# Training phases (SFT + GRPO-style RL)
# ---------------------------------------------------------------------------
async def run_sft_phase(
  train_examples: list[dict[str, Any]],
  eval_examples: list[dict[str, Any]],
  step_offset: int = 0,
) -> tuple[float, float]:
  if config.sft_steps <= 0:
    return await snapshot_eval("texttosql_sft_skip", eval_examples)

  batches = shuffled_batches(train_examples, config.sft_batch_size, config.seed)
  losses: list[float] = []
  exec_match, similarity = 0.0, 0.0
  logging.info(
    "Starting SFT: steps=%s batch=%s lr=%g train_examples=%s",
    config.sft_steps,
    min(config.sft_batch_size, len(train_examples)),
    config.sft_learning_rate,
    len(train_examples),
  )

  for local_step in range(1, config.sft_steps + 1):
    batch = next(batches)
    datums = [ex["datum"] for ex in batch]
    active_tokens = sum(ex["active_tokens"] for ex in batch)

    fwdbwd_future = await trainer.forward_backward_async(datums, "cross_entropy")
    optim_future = await trainer.optim_step_async(types.AdamParams(learning_rate=config.sft_learning_rate, grad_clip_norm=config.grad_clip_norm))
    fwdbwd = await fwdbwd_future
    await optim_future

    loss = float(fwdbwd.metrics.get("loss:sum", 0.0)) / max(1, active_tokens)
    losses.append(loss)
    global_step = step_offset + local_step
    ml_logger.log_metrics({"phase": "sft_train", "loss": loss}, step=global_step)

    if local_step % config.sft_eval_every == 0 or local_step == config.sft_steps:
      exec_match, similarity = await snapshot_eval(f"texttosql_sft_s{local_step}", eval_examples)
      ml_logger.log_metrics(
        {"phase": "sft_eval", "execution_match": exec_match, "similarity": similarity},
        step=global_step,
      )
      logging.info(
        "[sft step %03d] loss=%.4f eval_exec=%.1f%% eval_similarity=%.1f%%",
        local_step,
        loss,
        exec_match * 100,
        similarity * 100,
      )

  if len(losses) >= 2:
    logging.info("Completed SFT: loss_drop=%.1f%%", (losses[0] - losses[-1]) / (abs(losses[0]) or 1.0) * 100)
  return exec_match, similarity


async def run_rl_step(examples: list[dict[str, Any]], alias: str) -> dict[str, float | int]:
  """Sample N completions per prompt, score them, and take one GRPO gradient step."""
  sampler = await trainer.save_weights_and_get_sampling_client_async(name=alias)
  responses = await asyncio.gather(
    *[
      sampler.sample_async(
        prompt=types.ModelInput.from_ints(tokens=ex["prompt_tokens"]),
        num_samples=config.rl_samples_per_prompt,
        sampling_params=types.SamplingParams(max_tokens=config.rl_max_tokens, temperature=config.rl_temperature),
      )
      for ex in examples
    ]
  )

  datums: list[types.Datum] = []
  rollouts: list[dict[str, Any]] = []
  for example, response in zip(examples, responses):
    group = [score_rollout(example, seq) for seq in response.sequences]
    advantages = group_relative_advantages([r["reward"] for r in group])
    for rollout, advantage in zip(group, advantages):
      if abs(advantage) < 1e-8:
        continue
      if not rollout["completion_tokens"] or len(rollout["completion_tokens"]) != len(rollout["completion_logprobs"]):
        continue
      rollout["advantage"] = advantage
      rollouts.append(rollout)
      datums.append(make_rl_datum(rollout, advantage))

  if not datums:
    return {"loss": 0.0, "reward": 0.0, "compile_rate": 0.0, "execution_match": 0.0, "similarity": 0.0, "num_rollouts": 0}

  loss_fn_config = {"clip_range": config.rl_clip_range, "kl_coeff": config.rl_kl_coeff} if config.rl_loss_fn in {"ppo", "grpo"} else None
  fwdbwd_future = await trainer.forward_backward_async(datums, config.rl_loss_fn, loss_fn_config=loss_fn_config)
  optim_future = await trainer.optim_step_async(types.AdamParams(learning_rate=config.rl_learning_rate, grad_clip_norm=config.grad_clip_norm))
  fwdbwd = await fwdbwd_future
  await optim_future

  log_best_rollout(rollouts)

  def mean(k: str) -> float:
    return statistics.fmean(float(r[k]) for r in rollouts)

  return {
    "loss": float(fwdbwd.metrics.get("loss:mean", 0.0)),
    "num_rollouts": len(rollouts),
    "reward": mean("reward"),
    "compile_rate": mean("compile"),
    "execution_match": mean("execution_match"),
    "similarity": mean("similarity"),
  }


async def run_rl_phase(
  rl_examples: list[dict[str, Any]],
  eval_examples: list[dict[str, Any]],
  step_offset: int = 0,
) -> tuple[float, float]:
  if config.rl_steps <= 0:
    return await snapshot_eval("texttosql_rl_skip", eval_examples)

  batches = shuffled_batches(rl_examples, config.rl_prompts_per_step, config.seed + 1)
  exec_match, similarity = 0.0, 0.0
  logging.info(
    "Starting GRPO-style RL: steps=%s prompts_per_step=%s samples_per_prompt=%s lr=%g loss_fn=%s",
    config.rl_steps,
    min(config.rl_prompts_per_step, len(rl_examples)),
    config.rl_samples_per_prompt,
    config.rl_learning_rate,
    config.rl_loss_fn,
  )

  for local_step in range(1, config.rl_steps + 1):
    step_metrics = await run_rl_step(next(batches), alias=f"texttosql_rl_rollout_s{local_step}")
    global_step = step_offset + local_step
    ml_logger.log_metrics({"phase": "rl_train", **step_metrics}, step=global_step)
    logging.info(
      "[rl step %03d] reward=%.3f compile=%.1f%% exec=%.1f%% rollouts=%d",
      local_step,
      float(step_metrics["reward"]),
      float(step_metrics["compile_rate"]) * 100,
      float(step_metrics["execution_match"]) * 100,
      int(step_metrics["num_rollouts"]),
    )

    if local_step % config.rl_eval_every == 0 or local_step == config.rl_steps:
      exec_match, similarity = await snapshot_eval(f"texttosql_rl_s{local_step}", eval_examples)
      ml_logger.log_metrics(
        {"phase": "rl_eval", "execution_match": exec_match, "similarity": similarity},
        step=global_step,
      )
      logging.info(
        "[rl eval %03d] eval_exec=%.1f%% eval_similarity=%.1f%%",
        local_step,
        exec_match * 100,
        similarity * 100,
      )

  return exec_match, similarity


async def run_training(config_arg: Config, preset: str) -> dict[str, float | str]:
  """Initialize recipe globals, orchestrate the configured phases, and return summary metrics."""
  global config, ml_logger, service_client, tokenizer

  config = config_arg
  log_dir = Path(config.log_dir.replace("{preset}", preset))
  ml_logger = ml_log.setup_logging(log_dir=str(log_dir), config=config, do_configure_logging_module=True)
  service_client = tinker.ServiceClient(
    api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"),
    base_url=config.base_url,
    timeout=600.0,
  )
  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
  metrics_path = log_dir / "metrics.jsonl"

  server_model = await require_server(service_client, config.base_url)
  logging.info("Server ready at %s | model=%s", config.base_url, server_model or "unset")

  global trainer
  trainer = await service_client.create_lora_training_client_async(
    base_model=config.base_model,
    rank=config.rank,
    seed=config.seed,
    train_mlp=True,
    train_attn=True,
    train_unembed=False,
  )
  if config.resume_state_path:
    try:
      if config.resume_with_optimizer:
        await trainer.load_state_with_optimizer_async(config.resume_state_path)
      else:
        await trainer.load_state_async(config.resume_state_path)
    except Exception as exc:
      raise RuntimeError(f"Failed to resume trainer from {config.resume_state_path!r}") from exc

  sft_train, rl_train, eval_examples = load_example_splits(config, tokenizer)
  before_exec, before_sim = await snapshot_eval("texttosql_before", eval_examples)
  ml_logger.log_metrics({"phase": "eval_baseline", "execution_match": before_exec, "similarity": before_sim}, step=0)
  logging.info("baseline exec=%.1f%% sim=%.1f%%", before_exec * 100, before_sim * 100)

  after_sft_exec, after_sft_sim = before_exec, before_sim
  after_rl_exec, after_rl_sim = before_exec, before_sim
  sft_state_path = ""
  final_state_path = ""

  step_offset = 0
  if config.phase in {"full", "sft_only"}:
    logging.info(">>> Phase: SFT")
    after_sft_exec, after_sft_sim = await run_sft_phase(sft_train, eval_examples, step_offset=step_offset)
    step_offset += config.sft_steps
    if config.save_sft_state_name:
      sft_state_path = getattr(trainer.save_state(config.save_sft_state_name).result(), "path", "") or ""
      logging.info("Post-SFT state saved to %s", sft_state_path or "<empty path>")
  else:
    logging.info(">>> Skipping SFT (phase=%s)", config.phase)

  if config.phase in {"full", "rl_only"}:
    logging.info(">>> Phase: RL")
    after_rl_exec, after_rl_sim = await run_rl_phase(rl_train, eval_examples, step_offset=step_offset)
  else:
    logging.info(">>> Skipping RL (phase=%s)", config.phase)
    after_rl_exec, after_rl_sim = after_sft_exec, after_sft_sim

  if config.save_final_state_name:
    final_state_path = getattr(trainer.save_state(config.save_final_state_name).result(), "path", "") or ""
    logging.info("Final state saved to %s", final_state_path or "<empty path>")

  logging.info("Saved metrics to %s", metrics_path)
  logging.info(
    "[summary] execution=%.1f%%->%.1f%%->%.1f%% similarity=%.1f%%->%.1f%%->%.1f%%",
    before_exec * 100,
    after_sft_exec * 100,
    after_rl_exec * 100,
    before_sim * 100,
    after_sft_sim * 100,
    after_rl_sim * 100,
  )
  ml_logger.close()

  result: dict[str, float | str] = {
    "before_execution_match": before_exec,
    "after_sft_execution_match": after_sft_exec,
    "after_rl_execution_match": after_rl_exec,
    "before_similarity": before_sim,
    "after_sft_similarity": after_sft_sim,
    "after_rl_similarity": after_rl_sim,
    "metrics_path": str(metrics_path),
  }
  if sft_state_path:
    result["sft_state_path"] = sft_state_path
  if final_state_path:
    result["final_state_path"] = final_state_path
  return result


# ---------------------------------------------------------------------------
# Rollout scoring + datum construction (core RL mechanics)
# ---------------------------------------------------------------------------
def score_rollout(example: dict[str, Any], sequence: Any) -> dict[str, Any]:
  """Decode one sampled sequence, compute its SQL reward, return a flat rollout dict."""
  predicted_sql = clean_sql_for_execution(tokenizer.decode(sequence.tokens, skip_special_tokens=True))
  reward = compute_sql_reward(
    example,
    predicted_sql,
    compile_reward=config.compile_reward,
    match_reward=config.match_reward,
    compile_error_penalty=config.compile_error_penalty,
    similarity_reward=config.similarity_reward,
  )
  return {
    "question": example["question"],
    "target": example["target"],
    "predicted_sql": predicted_sql,
    "prompt_tokens": example["prompt_tokens"],
    "completion_tokens": list(sequence.tokens),
    "completion_logprobs": [float(v) for v in (sequence.logprobs or [])],
    "reward": reward["total"],
    "compile": reward["compile"],
    "execution_match": reward["execution_match"],
    "similarity": reward["similarity"],
    "sqlite_error": reward["sqlite_error"],
  }


def make_rl_datum(rollout: dict[str, Any], advantage: float) -> types.Datum:
  """Pack one scored rollout into a Datum with prompt-token weights/logprobs/advantages masked to 0."""
  prompt, completion, logprobs = rollout["prompt_tokens"], rollout["completion_tokens"], rollout["completion_logprobs"]
  prompt_pad = max(0, len(prompt) - 1)
  n = prompt_pad + len(completion)

  def masked_tensor(prompt_fill: float, completion_fill: list[float] | float) -> types.TensorData:
    tail = completion_fill if isinstance(completion_fill, list) else [completion_fill] * len(completion)
    return types.TensorData(data=[prompt_fill] * prompt_pad + tail, dtype="float32", shape=[n])

  full = prompt + completion
  return types.Datum(
    model_input=types.ModelInput.from_ints(tokens=full[:-1]),
    loss_fn_inputs=cast(
      Any,
      {
        "target_tokens": full[1:],
        "weights": masked_tensor(0.0, 1.0),
        "logprobs": masked_tensor(0.0, logprobs),
        "advantages": masked_tensor(0.0, advantage),
      },
    ),
  )


def group_relative_advantages(rewards: list[float]) -> list[float]:
  if len(rewards) < 2:
    return [0.0] * len(rewards)
  mean = statistics.fmean(rewards)
  std = statistics.pstdev(rewards)
  if std < 1e-8:
    return [0.0] * len(rewards)
  return [(r - mean) / std for r in rewards]


def log_best_rollout(rollouts: list[dict[str, Any]]) -> None:
  if not rollouts:
    return
  best = max(rollouts, key=lambda r: (r["reward"], r["execution_match"], r["compile"]))
  logging.info(
    "\n--- [RL Rollout Sample] ---\nQuestion: %s\nPredicted: %s\nTarget:    %s\nReward:    %.2f\nCompile:   %s\nExecution: %s%s\n",
    best["question"],
    normalize_sql(best["predicted_sql"]),
    normalize_sql(best["target"]),
    best["reward"],
    "YES" if best["compile"] else "NO",
    "MATCH" if best["execution_match"] else "NO MATCH",
    f"\nSQLite:    {best['sqlite_error']}" if best["sqlite_error"] else "",
  )


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
async def snapshot_eval(alias: str, eval_examples: list[dict[str, Any]]) -> tuple[float, float]:
  """Snapshot current weights to a sampler and run the eval loop against it."""
  sampler = await trainer.save_weights_and_get_sampling_client_async(name=alias)
  return await evaluate(sampler, tokenizer, alias, eval_examples, config)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  logging.getLogger("tinker").setLevel(logging.WARNING)

  preset = sys.argv[1]
  blueprint = PRESETS[preset].clone()
  config: Config = blueprint.make_from_argv(sys.argv[2:], allow_hyphens=True)
  asyncio.run(run_training(config, preset))
