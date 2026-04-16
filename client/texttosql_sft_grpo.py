from __future__ import annotations

"""Text-to-SQL SFT + RL recipe.

See `docs/guides/text-to-sql.md` for direct `uv run` commands.

Common phase modes:
- `phase=full`: run SFT, then RL
- `phase=sft_only`: stop after SFT
- `phase=rl_only`: skip SFT and run RL, optionally from `resume_state_path`
"""

import asyncio
import logging
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chz
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook.utils import ml_log

from texttosql_rewards import compute_sql_reward
from texttosql_sft import (
  BASE_URL,
  DATASET,
  build_examples,
  clean_sql_for_execution,
  evaluate,
  normalize_sql,
  require_server,
)

LOG_DIR = Path(__file__).resolve().parent / "artifacts" / "texttosql_sft_grpo_{preset}"

PHASE_FULL = "full"
PHASE_SFT_ONLY = "sft_only"
PHASE_RL_ONLY = "rl_only"
VALID_PHASES = {PHASE_FULL, PHASE_SFT_ONLY, PHASE_RL_ONLY}

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@dataclass
class EvalSnapshot:
  execution_match: float
  similarity: float


@dataclass
class ExampleSplits:
  sft_train: list[dict[str, Any]]
  rl_train: list[dict[str, Any]]
  eval: list[dict[str, Any]]


@dataclass
class TrainingSummary:
  baseline: EvalSnapshot
  after_sft: EvalSnapshot
  after_rl: EvalSnapshot
  sft_state_path: str = ""
  final_state_path: str = ""

  def to_dict(self, metrics_path: Path) -> dict[str, float | str]:
    result: dict[str, float | str] = {
      "before_execution_match": self.baseline.execution_match,
      "after_sft_execution_match": self.after_sft.execution_match,
      "after_rl_execution_match": self.after_rl.execution_match,
      "before_similarity": self.baseline.similarity,
      "after_sft_similarity": self.after_sft.similarity,
      "after_rl_similarity": self.after_rl.similarity,
      "metrics_path": str(metrics_path),
    }
    if self.sft_state_path:
      result["sft_state_path"] = self.sft_state_path
    if self.final_state_path:
      result["final_state_path"] = self.final_state_path
    return result


def runs_sft(phase: str) -> bool:
  return phase in {PHASE_FULL, PHASE_SFT_ONLY}


def runs_rl(phase: str) -> bool:
  return phase in {PHASE_FULL, PHASE_RL_ONLY}


def validate_config(config: Config) -> None:
  if config.phase not in VALID_PHASES:
    raise ValueError(f"phase must be one of: {', '.join(sorted(VALID_PHASES))}")
  if config.rl_loss_fn not in {"importance_sampling", "ppo"}:
    raise ValueError("rl_loss_fn must be importance_sampling or ppo")
  if config.resume_with_optimizer and not config.resume_state_path:
    raise ValueError("resume_with_optimizer requires resume_state_path")


def log_phase_header(name: str) -> None:
  logging.info("")
  logging.info("=" * 18)
  logging.info("Phase: %s", name)
  logging.info("=" * 18)


def group_relative_advantages(rewards: list[float]) -> list[float]:
  if len(rewards) < 2:
    return [0.0] * len(rewards)

  reward_mean = statistics.fmean(rewards)
  reward_std = statistics.pstdev(rewards)
  if reward_std < 1e-8:
    return [0.0] * len(rewards)
  return [(reward - reward_mean) / reward_std for reward in rewards]


def next_batch(
  examples: list[dict[str, Any]],
  order: list[int],
  batch_size: int,
  position: int,
  rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
  if position + batch_size > len(order):
    rng.shuffle(order)
    position = 0
  batch = [examples[order[idx]] for idx in range(position, position + batch_size)]
  return batch, position + batch_size


def make_rl_datum(
  prompt_tokens: list[int],
  completion_tokens: list[int],
  completion_logprobs: list[float],
  advantage: float,
) -> types.Datum:
  full_tokens = prompt_tokens + list(completion_tokens)
  target_tokens = full_tokens[1:]
  prompt_weight_count = max(0, len(prompt_tokens) - 1)
  completion_weight_count = len(completion_tokens)

  return types.Datum(
    model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
    loss_fn_inputs=cast(
      Any,
      {
        "target_tokens": target_tokens,
        "weights": types.TensorData(
          data=[0.0] * prompt_weight_count + [1.0] * completion_weight_count,
          dtype="float32",
          shape=[prompt_weight_count + completion_weight_count],
        ),
        "logprobs": types.TensorData(
          data=[0.0] * prompt_weight_count + list(completion_logprobs),
          dtype="float32",
          shape=[prompt_weight_count + len(completion_logprobs)],
        ),
        "advantages": types.TensorData(
          data=[0.0] * prompt_weight_count + [advantage] * completion_weight_count,
          dtype="float32",
          shape=[prompt_weight_count + completion_weight_count],
        ),
      },
    ),
  )


def build_rollout_rows(
  example: dict[str, Any],
  response: Any,
  tokenizer,
  config: Config,
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for sequence in response.sequences:
    predicted_sql = clean_sql_for_execution(tokenizer.decode(sequence.tokens, skip_special_tokens=True))
    reward = compute_sql_reward(
      example,
      predicted_sql,
      compile_reward=config.compile_reward,
      match_reward=config.match_reward,
      compile_error_penalty=config.compile_error_penalty,
      similarity_reward=config.similarity_reward,
    )
    rows.append(
      {
        "question": example["question"],
        "target": example["target"],
        "predicted_sql": predicted_sql,
        "prompt_tokens": example["prompt_tokens"],
        "completion_tokens": list(sequence.tokens),
        "completion_logprobs": [float(value) for value in (sequence.logprobs or [])],
        "reward": reward["total"],
        "compile": reward["compile"],
        "execution_match": reward["execution_match"],
        "similarity": reward["similarity"],
        "sqlite_error": reward["sqlite_error"],
      }
    )
  return rows


def build_rl_training_batch(group_rollouts: list[dict[str, Any]]) -> tuple[list[types.Datum], list[dict[str, Any]]]:
  datums: list[types.Datum] = []
  kept_rollouts: list[dict[str, Any]] = []
  advantages = group_relative_advantages([float(item["reward"]) for item in group_rollouts])

  for rollout, advantage in zip(group_rollouts, advantages):
    if abs(float(advantage)) < 1e-8:
      continue

    completion_tokens = rollout["completion_tokens"]
    completion_logprobs = rollout["completion_logprobs"]
    if not completion_tokens or len(completion_tokens) != len(completion_logprobs):
      continue

    datums.append(
      make_rl_datum(
        rollout["prompt_tokens"],
        completion_tokens,
        completion_logprobs,
        advantage,
      )
    )
    rollout["advantage"] = advantage
    kept_rollouts.append(rollout)

  return datums, kept_rollouts


def empty_rl_metrics() -> dict[str, float | int]:
  return {
    "loss": 0.0,
    "reward": 0.0,
    "compile_rate": 0.0,
    "execution_match": 0.0,
    "similarity": 0.0,
    "num_rollouts": 0,
  }


def summarize_rollouts(rollout_rows: list[dict[str, Any]], loss: float) -> dict[str, float | int]:
  if not rollout_rows:
    return empty_rl_metrics()

  return {
    "loss": loss,
    "reward": statistics.fmean(float(row["reward"]) for row in rollout_rows),
    "compile_rate": statistics.fmean(float(row["compile"]) for row in rollout_rows),
    "execution_match": statistics.fmean(float(row["execution_match"]) for row in rollout_rows),
    "similarity": statistics.fmean(float(row["similarity"]) for row in rollout_rows),
    "num_rollouts": len(rollout_rows),
  }


def log_best_rollout(rollout_rows: list[dict[str, Any]]) -> None:
  if not rollout_rows:
    return

  best_rollout = max(rollout_rows, key=lambda row: (row["reward"], row["execution_match"], row["compile"]))
  logging.info(
    "\n--- [RL Rollout Sample] ---\nQuestion: %s\nPredicted: %s\nTarget:    %s\nReward:    %.2f\nCompile:   %s\nExecution: %s%s\n",
    best_rollout["question"],
    normalize_sql(best_rollout["predicted_sql"]),
    normalize_sql(best_rollout["target"]),
    best_rollout["reward"],
    "YES" if best_rollout["compile"] else "NO",
    "MATCH" if best_rollout["execution_match"] else "NO MATCH",
    f"\nSQLite:    {best_rollout['sqlite_error']}" if best_rollout["sqlite_error"] else "",
  )


def load_example_splits(tokenizer, config: Config) -> ExampleSplits:
  dataset = load_dataset(config.dataset_name, split="train").shuffle(seed=config.seed)
  dataset = dataset.select(range(min(config.dataset_limit, len(dataset))))
  if len(dataset) < 10:
    raise RuntimeError("dataset_limit is too small to create train/eval splits")

  split = dataset.train_test_split(test_size=min(2_500, max(1, len(dataset) // 5)), shuffle=False)

  sft_examples = (
    build_examples(tokenizer, config.prompt_format, split["train"], config.train_limit)
    if runs_sft(config.phase)
    else []
  )
  rl_examples = (
    build_examples(
      tokenizer,
      config.prompt_format,
      split["train"],
      config.rl_train_limit,
      require_seed_data=True,
      require_target_rows=True,
    )
    if runs_rl(config.phase)
    else []
  )
  eval_examples = build_examples(
    tokenizer,
    config.prompt_format,
    split["test"],
    config.eval_limit,
    require_seed_data=True,
    require_target_rows=True,
  )

  if runs_sft(config.phase) and not sft_examples:
    raise RuntimeError("No SFT examples fit within the max sequence length.")
  if runs_rl(config.phase) and not rl_examples:
    raise RuntimeError("No RL examples with executable target rows were found.")
  if not eval_examples:
    raise RuntimeError("No evaluation examples with executable seed data were found.")

  logging.info(
    "Data: %s SFT train, %s RL train, %s eval",
    len(sft_examples),
    len(rl_examples),
    len(eval_examples),
  )
  return ExampleSplits(sft_train=sft_examples, rl_train=rl_examples, eval=eval_examples)


async def snapshot_eval(
  trainer,
  client: tinker.ServiceClient,
  tokenizer,
  alias: str,
  eval_examples: list[dict[str, Any]],
  config: Config,
) -> EvalSnapshot:
  sampler_path = trainer.save_weights_for_sampler(name=alias).result().path
  sampler = client.create_sampling_client(sampler_path)
  execution_match, similarity = await evaluate(sampler, tokenizer, alias, eval_examples, config)
  return EvalSnapshot(execution_match=execution_match, similarity=similarity)


async def create_or_resume_trainer(client: tinker.ServiceClient, config: Config):
  if config.resume_state_path:
    try:
      if config.resume_with_optimizer:
        return await client.create_training_client_from_state_with_optimizer_async(config.resume_state_path)
      return await client.create_training_client_from_state_async(config.resume_state_path)
    except Exception as exc:
      raise RuntimeError(
        "Failed to resume from resume_state_path. The hosted Tinker API supports this, "
        "but this local Open-RL backend may not have checkpoint restore wired yet."
      ) from exc

  if config.phase == PHASE_RL_ONLY:
    logging.info("phase=rl_only without resume_state_path: starting RL from a fresh adapter.")

  return await client.create_lora_training_client_async(
    base_model=config.base_model,
    rank=config.rank,
    seed=config.seed,
    train_mlp=True,
    train_attn=True,
    train_unembed=False,
  )


async def maybe_save_state(trainer, name: str, label: str) -> str:
  if not name:
    return ""
  save_result = trainer.save_state(name).result()
  save_path = getattr(save_result, "path", "") or ""
  logging.info("%s state saved to %s", label, save_path or "<empty path>")
  return save_path


async def run_sft_phase(
  *,
  trainer,
  client: tinker.ServiceClient,
  tokenizer,
  train_examples: list[dict[str, Any]],
  eval_examples: list[dict[str, Any]],
  config: Config,
  ml_logger,
  step_offset: int,
) -> EvalSnapshot:
  if config.sft_steps <= 0:
    return await snapshot_eval(trainer, client, tokenizer, "texttosql_sft_skip", eval_examples, config)

  batch_size = min(config.sft_batch_size, len(train_examples))
  rng = random.Random(config.seed)
  order = list(range(len(train_examples)))
  rng.shuffle(order)
  position = 0
  losses: list[float] = []
  latest_metrics = EvalSnapshot(execution_match=0.0, similarity=0.0)

  logging.info(
    "Starting SFT: steps=%s batch=%s lr=%g train_examples=%s",
    config.sft_steps,
    batch_size,
    config.sft_learning_rate,
    len(train_examples),
  )

  for local_step in range(1, config.sft_steps + 1):
    batch, position = next_batch(train_examples, order, batch_size, position, rng)
    datums = [example["datum"] for example in batch]
    active_tokens = sum(example["active_tokens"] for example in batch)

    fwdbwd_future = await trainer.forward_backward_async(datums, "cross_entropy")
    optim_future = await trainer.optim_step_async(
      types.AdamParams(
        learning_rate=config.sft_learning_rate,
        grad_clip_norm=config.grad_clip_norm,
      )
    )
    fwdbwd = await fwdbwd_future
    await optim_future

    loss = float(fwdbwd.metrics.get("loss:sum", 0.0)) / max(1, active_tokens)
    global_step = step_offset + local_step
    losses.append(loss)
    ml_logger.log_metrics({"phase": "sft_train", "loss": loss}, step=global_step)

    if local_step % config.sft_eval_every == 0 or local_step == config.sft_steps:
      latest_metrics = await snapshot_eval(
        trainer,
        client,
        tokenizer,
        f"texttosql_sft_s{local_step}",
        eval_examples,
        config,
      )
      ml_logger.log_metrics(
        {"phase": "sft_eval", "execution_match": latest_metrics.execution_match, "similarity": latest_metrics.similarity},
        step=global_step,
      )
      logging.info(
        "[sft step %03d] loss=%.4f eval_exec=%.1f%% eval_similarity=%.1f%%",
        local_step,
        loss,
        latest_metrics.execution_match * 100,
        latest_metrics.similarity * 100,
      )

  if len(losses) >= 2:
    loss_drop = (losses[0] - losses[-1]) / (abs(losses[0]) or 1.0)
    logging.info("Completed SFT: loss_drop=%.1f%%", loss_drop * 100)
  return latest_metrics


async def run_rl_step(
  *,
  trainer,
  client: tinker.ServiceClient,
  tokenizer,
  examples: list[dict[str, Any]],
  config: Config,
  alias: str,
) -> dict[str, float | int]:
  sampler_path = trainer.save_weights_for_sampler(name=alias).result().path
  sampler = client.create_sampling_client(sampler_path)
  futures = [
    sampler.sample_async(
      prompt=types.ModelInput.from_ints(tokens=example["prompt_tokens"]),
      num_samples=config.rl_samples_per_prompt,
      sampling_params=types.SamplingParams(
        max_tokens=config.rl_max_tokens,
        temperature=config.rl_temperature,
      ),
    )
    for example in examples
  ]
  responses = await asyncio.gather(*futures)

  datums: list[types.Datum] = []
  rollout_rows: list[dict[str, Any]] = []
  for example, response in zip(examples, responses):
    group_rollouts = build_rollout_rows(example, response, tokenizer, config)
    group_datums, kept_rollouts = build_rl_training_batch(group_rollouts)
    datums.extend(group_datums)
    rollout_rows.extend(kept_rollouts)

  if not datums:
    return empty_rl_metrics()

  if config.rl_loss_fn == "ppo":
    loss_fn_config = {"clip_range": config.rl_clip_range, "kl_coeff": config.rl_kl_coeff}
  else:
    loss_fn_config = None
  fwdbwd_future = await trainer.forward_backward_async(datums, config.rl_loss_fn, loss_fn_config=loss_fn_config)
  optim_future = await trainer.optim_step_async(
    types.AdamParams(
      learning_rate=config.rl_learning_rate,
      grad_clip_norm=config.grad_clip_norm,
    )
  )
  fwdbwd = await fwdbwd_future
  await optim_future

  log_best_rollout(rollout_rows)
  loss = float(fwdbwd.metrics.get("loss:mean", 0.0))
  return summarize_rollouts(rollout_rows, loss)


async def run_rl_phase(
  *,
  trainer,
  client: tinker.ServiceClient,
  tokenizer,
  rl_examples: list[dict[str, Any]],
  eval_examples: list[dict[str, Any]],
  config: Config,
  ml_logger,
  step_offset: int,
) -> EvalSnapshot:
  if config.rl_steps <= 0:
    return await snapshot_eval(trainer, client, tokenizer, "texttosql_rl_skip", eval_examples, config)

  batch_size = min(config.rl_prompts_per_step, len(rl_examples))
  rng = random.Random(config.seed + 1)
  order = list(range(len(rl_examples)))
  rng.shuffle(order)
  position = 0
  latest_metrics = EvalSnapshot(execution_match=0.0, similarity=0.0)

  logging.info(
    "Starting GRPO-style RL: steps=%s prompts_per_step=%s samples_per_prompt=%s lr=%g loss_fn=%s",
    config.rl_steps,
    batch_size,
    config.rl_samples_per_prompt,
    config.rl_learning_rate,
    config.rl_loss_fn,
  )

  for local_step in range(1, config.rl_steps + 1):
    batch, position = next_batch(rl_examples, order, batch_size, position, rng)
    step_metrics = await run_rl_step(
      trainer=trainer,
      client=client,
      tokenizer=tokenizer,
      examples=batch,
      config=config,
      alias=f"texttosql_rl_rollout_s{local_step}",
    )
    global_step = step_offset + local_step
    ml_logger.log_metrics({"phase": "rl_train", **step_metrics}, step=global_step)
    logging.info(
      "[rl step %03d] reward=%.3f compile=%.1f%% exec=%.1f%% rollouts=%s",
      local_step,
      float(step_metrics["reward"]),
      float(step_metrics["compile_rate"]) * 100,
      float(step_metrics["execution_match"]) * 100,
      int(step_metrics["num_rollouts"]),
    )

    if local_step % config.rl_eval_every == 0 or local_step == config.rl_steps:
      latest_metrics = await snapshot_eval(
        trainer,
        client,
        tokenizer,
        f"texttosql_rl_s{local_step}",
        eval_examples,
        config,
      )
      ml_logger.log_metrics(
        {"phase": "rl_eval", "execution_match": latest_metrics.execution_match, "similarity": latest_metrics.similarity},
        step=global_step,
      )
      logging.info(
        "[rl eval %03d] eval_exec=%.1f%% eval_similarity=%.1f%%",
        local_step,
        latest_metrics.execution_match * 100,
        latest_metrics.similarity * 100,
      )

  return latest_metrics


async def run_training(config: Config, preset: str) -> dict[str, float | str]:
  validate_config(config)

  log_dir = Path(config.log_dir.replace("{preset}", preset))
  ml_logger = ml_log.setup_logging(log_dir=str(log_dir), config=config, do_configure_logging_module=True)
  metrics_path = log_dir / "metrics.jsonl"

  client = tinker.ServiceClient(
    api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"),
    base_url=config.base_url,
    timeout=600.0,
  )
  server_model = await require_server(client, config.base_url)
  logging.info("Server ready at %s | model=%s", config.base_url, server_model or "unset")

  trainer = await create_or_resume_trainer(client, config)

  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
  examples = load_example_splits(tokenizer, config)

  baseline = await snapshot_eval(trainer, client, tokenizer, "texttosql_before", examples.eval, config)
  ml_logger.log_metrics(
    {"phase": "eval_baseline", "execution_match": baseline.execution_match, "similarity": baseline.similarity},
    step=0,
  )
  summary = TrainingSummary(baseline=baseline, after_sft=baseline, after_rl=baseline)
  step_offset = 0

  if runs_sft(config.phase):
    log_phase_header("SFT")
    summary.after_sft = await run_sft_phase(
      trainer=trainer,
      client=client,
      tokenizer=tokenizer,
      train_examples=examples.sft_train,
      eval_examples=examples.eval,
      config=config,
      ml_logger=ml_logger,
      step_offset=step_offset,
    )
    step_offset += config.sft_steps
    summary.sft_state_path = await maybe_save_state(trainer, config.save_sft_state_name, "Post-SFT")
  else:
    logging.info("Skipping SFT phase (phase=%s)", config.phase)

  if runs_rl(config.phase):
    log_phase_header("RL")
    summary.after_rl = await run_rl_phase(
      trainer=trainer,
      client=client,
      tokenizer=tokenizer,
      rl_examples=examples.rl_train,
      eval_examples=examples.eval,
      config=config,
      ml_logger=ml_logger,
      step_offset=step_offset,
    )
  else:
    logging.info("Skipping RL phase (phase=%s)", config.phase)
    summary.after_rl = summary.after_sft

  summary.final_state_path = await maybe_save_state(trainer, config.save_final_state_name, "Final")

  logging.info("Saved metrics to %s", metrics_path)
  logging.info(
    "[summary] execution=%.1f%%->%.1f%%->%.1f%% similarity=%.1f%%->%.1f%%->%.1f%%",
    summary.baseline.execution_match * 100,
    summary.after_sft.execution_match * 100,
    summary.after_rl.execution_match * 100,
    summary.baseline.similarity * 100,
    summary.after_sft.similarity * 100,
    summary.after_rl.similarity * 100,
  )
  ml_logger.close()
  return summary.to_dict(metrics_path)


# Advanced tuning knobs live near the CLI so the training logic above can read
# top-to-bottom without starting with a large config block.
@chz.chz
class Config:
  # Model / runtime
  base_model: str
  tokenizer_name: str
  rank: int
  phase: str = PHASE_FULL
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
  rl_loss_fn: str = "ppo"
  rl_clip_range: float = 0.2

  # Reward / checkpointing
  compile_reward: float = 0.25
  match_reward: float = 1.0
  compile_error_penalty: float = -0.5
  similarity_reward: float = 0.0
  rl_kl_coeff: float = 0.1
  resume_state_path: str = ""
  resume_with_optimizer: bool = False
  save_sft_state_name: str = ""
  save_final_state_name: str = ""


GEMMA4_E2B_PRESET = {
  "base_model": "google/gemma-4-e2b",
  "tokenizer_name": "google/gemma-4-e2b",
  "rank": 32,
  "train_limit": 100,
  "rl_train_limit": 64,
  "eval_limit": 25,
  "sft_steps": 100,
  "sft_batch_size": 1,
  "sft_learning_rate": 5e-5,
  "sft_eval_every": 100,
  "rl_steps": 40,
  "rl_prompts_per_step": 4,
  "rl_samples_per_prompt": 4,
  "rl_learning_rate": 1e-5,
  "rl_temperature": 0.8,
  "rl_eval_every": 10,
  "rl_loss_fn": "ppo",
}

GEMMA4_E2B_RL_RECIPE_PRESET = {
  **GEMMA4_E2B_PRESET,
  "rl_eval_every": 4,
  "rl_loss_fn": "importance_sampling",
  "rl_samples_per_prompt": 6,
  # Keep a small compile signal so RL does not collapse into pure string mimicry.
  "compile_reward": 0.25,
  "match_reward": 2.0,
  "compile_error_penalty": -0.25,
}

GEMMA4_E2B_SMOKE_PRESET = {
  **GEMMA4_E2B_PRESET,
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
}


def build_presets() -> dict[str, Any]:
  preset_overrides = {
    "gemma4_e2b": GEMMA4_E2B_PRESET,
    "gemma4_e2b_rl_recipe": GEMMA4_E2B_RL_RECIPE_PRESET,
    "gemma4_e2b_smoke": GEMMA4_E2B_SMOKE_PRESET,
  }
  return {
    name: chz.Blueprint(Config).apply(overrides, layer_name=name)
    for name, overrides in preset_overrides.items()
  }


PRESETS = build_presets()


@chz.blueprint._entrypoint.exit_on_entrypoint_error
def cli() -> None:
  import sys

  logging.getLogger("tinker").setLevel(logging.WARNING)

  preset = sys.argv[1]
  blueprint = PRESETS[preset].clone()
  config = blueprint.make_from_argv(sys.argv[2:], allow_hyphens=True)
  asyncio.run(run_training(config, preset))


if __name__ == "__main__":
  cli()
