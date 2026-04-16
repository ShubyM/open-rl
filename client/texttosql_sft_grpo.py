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
from pathlib import Path
from typing import Any, cast

import chz
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook.utils import ml_log
from transformers import PreTrainedTokenizerBase

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

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def group_relative_advantages(rewards: list[float]) -> list[float]:
  if len(rewards) < 2:
    return [0.0] * len(rewards)

  reward_mean = statistics.fmean(rewards)
  reward_std = statistics.pstdev(rewards)
  if reward_std < 1e-8:
    return [0.0] * len(rewards)
  return [(reward - reward_mean) / reward_std for reward in rewards]


def shuffled_batches(examples: list[dict[str, Any]], batch_size: int, seed: int):
  """Yield shuffled mini-batches forever, reshuffling when the pool is exhausted."""
  rng = random.Random(seed)
  order = list(range(len(examples)))
  rng.shuffle(order)
  pos = 0
  batch_size = min(batch_size, len(examples))
  while True:
    if pos + batch_size > len(order):
      rng.shuffle(order)
      pos = 0
    yield [examples[order[i]] for i in range(pos, pos + batch_size)]
    pos += batch_size


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


def build_rollout_rows(example: dict[str, Any], response: Any) -> list[dict[str, Any]]:
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


def summarize_rollouts(rollout_rows: list[dict[str, Any]], loss: float) -> dict[str, float | int]:
  n = len(rollout_rows)
  if not n:
    return {"loss": 0.0, "reward": 0.0, "compile_rate": 0.0, "execution_match": 0.0, "similarity": 0.0, "num_rollouts": 0}
  mean = lambda k: statistics.fmean(float(r[k]) for r in rollout_rows)
  return {
    "loss": loss, "num_rollouts": n,
    "reward": mean("reward"), "compile_rate": mean("compile"),
    "execution_match": mean("execution_match"), "similarity": mean("similarity"),
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


def load_example_splits() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
  """Returns (sft_train, rl_train, eval)."""
  do_sft = config.phase in {PHASE_FULL, PHASE_SFT_ONLY}
  do_rl = config.phase in {PHASE_FULL, PHASE_RL_ONLY}

  dataset = load_dataset(config.dataset_name, split="train").shuffle(seed=config.seed)
  dataset = dataset.select(range(min(config.dataset_limit, len(dataset))))
  if len(dataset) < 10:
    raise RuntimeError("dataset_limit is too small to create train/eval splits")

  split = dataset.train_test_split(test_size=min(2_500, max(1, len(dataset) // 5)), shuffle=False)

  sft_examples = (
    build_examples(tokenizer, config.prompt_format, split["train"], config.train_limit)
    if do_sft
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
    if do_rl
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

  if do_sft and not sft_examples:
    raise RuntimeError("No SFT examples fit within the max sequence length.")
  if do_rl and not rl_examples:
    raise RuntimeError("No RL examples with executable target rows were found.")
  if not eval_examples:
    raise RuntimeError("No evaluation examples with executable seed data were found.")

  logging.info(
    "Data: %s SFT train, %s RL train, %s eval",
    len(sft_examples),
    len(rl_examples),
    len(eval_examples),
  )
  return sft_examples, rl_examples, eval_examples


async def snapshot_eval(alias: str, eval_examples: list[dict[str, Any]]) -> tuple[float, float]:
  """Returns (execution_match, similarity)."""
  sampler_path = trainer.save_weights_for_sampler(name=alias).result().path
  sampler = service_client.create_sampling_client(sampler_path)
  return await evaluate(sampler, tokenizer, alias, eval_examples, config)


async def load_trainer(resume_from: str | None = None, with_optimizer: bool = False) -> tinker.TrainingClient:
  """Create a fresh LoRA adapter, or resume one from a saved state path."""
  if resume_from:
    try:
      if with_optimizer:
        return await service_client.create_training_client_from_state_with_optimizer_async(resume_from)
      return await service_client.create_training_client_from_state_async(resume_from)
    except Exception as exc:
      raise RuntimeError(f"Failed to resume trainer from {resume_from!r}") from exc
  return await service_client.create_lora_training_client_async(
    base_model=config.base_model, rank=config.rank, seed=config.seed,
    train_mlp=True, train_attn=True, train_unembed=False,
  )


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
    config.sft_steps, min(config.sft_batch_size, len(train_examples)), config.sft_learning_rate, len(train_examples),
  )

  for local_step in range(1, config.sft_steps + 1):
    batch = next(batches)
    datums = [ex["datum"] for ex in batch]
    active_tokens = sum(ex["active_tokens"] for ex in batch)

    fwdbwd_future = await trainer.forward_backward_async(datums, "cross_entropy")
    optim_future = await trainer.optim_step_async(
      types.AdamParams(learning_rate=config.sft_learning_rate, grad_clip_norm=config.grad_clip_norm)
    )
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
        local_step, loss, exec_match * 100, similarity * 100,
      )

  if len(losses) >= 2:
    logging.info("Completed SFT: loss_drop=%.1f%%", (losses[0] - losses[-1]) / (abs(losses[0]) or 1.0) * 100)
  return exec_match, similarity


async def run_rl_step(examples: list[dict[str, Any]], alias: str) -> dict[str, float | int]:
  sampler_path = trainer.save_weights_for_sampler(name=alias).result().path
  sampler = service_client.create_sampling_client(sampler_path)
  futures = [
    sampler.sample_async(
      prompt=types.ModelInput.from_ints(tokens=ex["prompt_tokens"]),
      num_samples=config.rl_samples_per_prompt,
      sampling_params=types.SamplingParams(max_tokens=config.rl_max_tokens, temperature=config.rl_temperature),
    )
    for ex in examples
  ]
  responses = await asyncio.gather(*futures)

  datums: list[types.Datum] = []
  rollout_rows: list[dict[str, Any]] = []
  for example, response in zip(examples, responses):
    group_datums, kept = build_rl_training_batch(build_rollout_rows(example, response))
    datums.extend(group_datums)
    rollout_rows.extend(kept)

  if not datums:
    return summarize_rollouts([], 0.0)

  loss_fn_config = (
    {"clip_range": config.rl_clip_range, "kl_coeff": config.rl_kl_coeff} if config.rl_loss_fn == "ppo" else None
  )
  fwdbwd_future = await trainer.forward_backward_async(datums, config.rl_loss_fn, loss_fn_config=loss_fn_config)
  optim_future = await trainer.optim_step_async(
    types.AdamParams(learning_rate=config.rl_learning_rate, grad_clip_norm=config.grad_clip_norm)
  )
  fwdbwd = await fwdbwd_future
  await optim_future

  log_best_rollout(rollout_rows)
  return summarize_rollouts(rollout_rows, float(fwdbwd.metrics.get("loss:mean", 0.0)))


async def run_rl_phase(
  rl_examples: list[dict[str, Any]],
  eval_examples: list[dict[str, Any]],
  step_offset: int = 0,
  resume_from: str | None = None,
) -> tuple[float, float]:
  """Run the GRPO-style RL phase.

  If `resume_from` is set, (re)load the trainer from that checkpoint first so
  you can do `run_rl_phase(..., resume_from="post-sft")` as a standalone RL
  continuation without running SFT in the same process.
  """
  if resume_from:
    global trainer
    trainer = await load_trainer(resume_from=resume_from, with_optimizer=config.resume_with_optimizer)

  if config.rl_steps <= 0:
    return await snapshot_eval("texttosql_rl_skip", eval_examples)

  batches = shuffled_batches(rl_examples, config.rl_prompts_per_step, config.seed + 1)
  exec_match, similarity = 0.0, 0.0
  logging.info(
    "Starting GRPO-style RL: steps=%s prompts_per_step=%s samples_per_prompt=%s lr=%g loss_fn=%s",
    config.rl_steps, min(config.rl_prompts_per_step, len(rl_examples)),
    config.rl_samples_per_prompt, config.rl_learning_rate, config.rl_loss_fn,
  )

  for local_step in range(1, config.rl_steps + 1):
    step_metrics = await run_rl_step(next(batches), alias=f"texttosql_rl_rollout_s{local_step}")
    global_step = step_offset + local_step
    ml_logger.log_metrics({"phase": "rl_train", **step_metrics}, step=global_step)
    logging.info(
      "[rl step %03d] reward=%.3f compile=%.1f%% exec=%.1f%% rollouts=%d",
      local_step, float(step_metrics["reward"]),
      float(step_metrics["compile_rate"]) * 100, float(step_metrics["execution_match"]) * 100,
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
        local_step, exec_match * 100, similarity * 100,
      )

  return exec_match, similarity


async def run_training(metrics_path: Path) -> dict[str, float | str]:
  """Run the configured phases against the already-initialized module globals."""
  global trainer

  server_model = await require_server(service_client, config.base_url)
  logging.info("Server ready at %s | model=%s", config.base_url, server_model or "unset")

  # RL-only resumes lazily inside run_rl_phase so the same code path handles
  # both "resume from a specific checkpoint" and "just continue the current run".
  trainer = await load_trainer(
    resume_from=config.resume_state_path if config.phase != PHASE_RL_ONLY else None,
    with_optimizer=config.resume_with_optimizer,
  )

  sft_train, rl_train, eval_examples = load_example_splits()
  before_exec, before_sim = await snapshot_eval("texttosql_before", eval_examples)
  ml_logger.log_metrics(
    {"phase": "eval_baseline", "execution_match": before_exec, "similarity": before_sim},
    step=0,
  )
  logging.info("baseline exec=%.1f%% sim=%.1f%%", before_exec * 100, before_sim * 100)

  after_sft_exec, after_sft_sim = before_exec, before_sim
  after_rl_exec, after_rl_sim = before_exec, before_sim
  sft_state_path = ""
  final_state_path = ""

  step_offset = 0
  if config.phase in {PHASE_FULL, PHASE_SFT_ONLY}:
    logging.info(">>> Phase: SFT")
    after_sft_exec, after_sft_sim = await run_sft_phase(sft_train, eval_examples, step_offset=step_offset)
    step_offset += config.sft_steps
    if config.save_sft_state_name:
      sft_state_path = getattr(trainer.save_state(config.save_sft_state_name).result(), "path", "") or ""
      logging.info("Post-SFT state saved to %s", sft_state_path or "<empty path>")
  else:
    logging.info(">>> Skipping SFT (phase=%s)", config.phase)

  if config.phase in {PHASE_FULL, PHASE_RL_ONLY}:
    logging.info(">>> Phase: RL")
    after_rl_exec, after_rl_sim = await run_rl_phase(
      rl_train, eval_examples, step_offset=step_offset,
      resume_from=config.resume_state_path if config.phase == PHASE_RL_ONLY else None,
    )
  else:
    logging.info(">>> Skipping RL (phase=%s)", config.phase)
    after_rl_exec, after_rl_sim = after_sft_exec, after_sft_sim

  if config.save_final_state_name:
    final_state_path = getattr(trainer.save_state(config.save_final_state_name).result(), "path", "") or ""
    logging.info("Final state saved to %s", final_state_path or "<empty path>")

  logging.info("Saved metrics to %s", metrics_path)
  logging.info(
    "[summary] execution=%.1f%%->%.1f%%->%.1f%% similarity=%.1f%%->%.1f%%->%.1f%%",
    before_exec * 100, after_sft_exec * 100, after_rl_exec * 100,
    before_sim * 100, after_sft_sim * 100, after_rl_sim * 100,
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


# Advanced tuning knobs live near the CLI so the training logic above can read
# top-to-bottom without starting with a large config block.
@chz.chz
class Config:
  # Model / runtime
  base_model: str
  tokenizer_name: str
  rank: int = 32
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


GEMMA4_E2B = {"base_model": "google/gemma-4-e2b", "tokenizer_name": "google/gemma-4-e2b"}

PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
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
}

PRESETS = {
  name: chz.Blueprint(Config).apply(overrides, layer_name=name)
  for name, overrides in PRESET_OVERRIDES.items()
}


if __name__ == "__main__":
  import sys

  from transformers import AutoTokenizer

  logging.getLogger("tinker").setLevel(logging.WARNING)

  preset = sys.argv[1]
  blueprint = PRESETS[preset].clone()
  config: Config = blueprint.make_from_argv(sys.argv[2:], allow_hyphens=True)

  log_dir = Path(config.log_dir.replace("{preset}", preset))
  ml_logger: ml_log.Logger = ml_log.setup_logging(
    log_dir=str(log_dir), config=config, do_configure_logging_module=True
  )
  metrics_path = log_dir / "metrics.jsonl"

  service_client: tinker.ServiceClient = tinker.ServiceClient(
    api_key=os.getenv("TINKER_API_KEY", "tml-dummy-key"),
    base_url=config.base_url,
    timeout=600.0,
  )
  tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.tokenizer_name)

  asyncio.run(run_training(metrics_path))
