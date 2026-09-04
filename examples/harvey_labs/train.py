"""Train on Harvey LAB with live tool-use rollouts."""

from __future__ import annotations

import asyncio
import io
import json
import re
import subprocess
from pathlib import Path

import chz
import tinker
from tinker.lib.public_interfaces import training_client as tinker_training_client
from tinker_cookbook import checkpoint_utils

from .env import LabDatasetBuilder
from .tasks import BOOTSTRAP_TASKS, EVAL_TASKS, family_task_split, random_task_split, three_way_task_split

# 5MB chunks put one long-context datum per request, collapsing DP sharding
# to rank 0. ~30MB carries several datums so ranks get real shards, while
# staying small enough for the HTTP/JSON path (a single giant request
# stalled the gateway loop and reset connections).
tinker_training_client.MAX_CHUNK_BYTES_COUNT = 30_000_000

from tinker_cookbook.rl import message_env as rl_message_env
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.stores.storage import LocalStorage
from tinker_cookbook.stores.training_store import TrainingRunStore
from tinker_utils import force_rich_log_colors, resolve_base_url

# Give optim_step a gradient clip. The cookbook builds its AdamParams as
# AdamParams(learning_rate=..., beta1=.9, beta2=.95, eps=1e-8) at two call sites
# and never passes grad_clip_norm, which defaults to 0.0; lora_trainer_worker
# reads that as `adam_params.get("grad_clip_norm") or math.inf`, so nothing is
# ever clipped. The knob is implemented on both ends -- it is simply not
# reachable from a recipe, and rl_train.Config has no field for it either, so a
# patch here is the only place to set it without forking a pinned dependency.
#
# The threshold sits just above the healthy range rather than at the usual O(1).
# Clipping is only useful against spikes anyway -- Adam is scale-invariant to a
# *constant* gradient rescale (m and sqrt(v) scale together), so a clip low
# enough to bind on every step would buy nothing and only discard the
# relative-magnitude signal Adam runs on. run26 hit 7.6e5 and 1.3e6 on the two
# steps where it came apart, which is what this is meant to catch.
#
# run27 measured the healthy distribution and 1e5 looked too low: it bound on 11
# of 21 steps, i.e. most of the ordinary body (5e4-2e5) rather than the tail,
# while the run's real outliers were ~9.8e5. run28 raised it to 3e5 to clear the
# body -- and that was the wrong read. Binding often was doing useful work.
# Over 40 steps at 3e5, entropy went 0.21 -> 0.55 (run27 held 0.15-0.31 for its
# whole length), KL(sample||train) went 7.7e-4 -> 6.8e-3, and train reward drifted
# down 0.217 -> 0.200. The looser clip bought drift, not signal, so this is back
# at run27's value.
GRAD_CLIP_NORM = 1e5

_BaseAdamParams = tinker.AdamParams


class _AdamParamsWithClip(_BaseAdamParams):
  def __init__(self, **kwargs):
    kwargs.setdefault("grad_clip_norm", GRAD_CLIP_NORM)
    super().__init__(**kwargs)


tinker.AdamParams = _AdamParamsWithClip

# Report the terminal-condition flags as rates instead of the constant 1.0.
# message_env emits context_overflow / parse_error / max_tokens_reached only on
# the transition that trips them, and only ever as 1.0. dict_mean averages a key
# over just the dicts that carry it ("missing keys are not treated as zero"), so
# each one reads exactly 1.000 whenever >=1 episode trips it and is absent
# otherwise -- it encodes presence, never frequency. That hid the failure mode
# in run26: max_tokens_reached went 6% -> 54% of episodes across seven steps
# while logging 1.000 throughout, and context_overflow logged the same 1.000
# while falling 33% -> 0%.
#
# Defaulting them to 0.0 makes the mean a real rate. Note the denominator is
# transitions, not episodes: these flags all end the episode, so each episode
# trips at most one once, and the per-episode rate is the logged value times
# total_turns / total_episodes (both already in the metrics).
_TERMINAL_FLAG_METRICS = ("context_overflow", "parse_error", "max_tokens_reached")
_base_env_step = rl_message_env.EnvFromMessageEnv.step

# Keep the generation behind a parse_error, because nothing else does.
# message_env's MALFORMED branch returns metrics={"parse_error": 1.0} and an
# empty next_observation, dropping the text; StepResult.logs is left empty on
# exactly those steps, so the failure survives nowhere -- not the rollout
# summaries, not the logtree, not the HTML. run27 could establish that parse
# errors land on generations ~13x longer than clean turns (median ac_len 3.4k vs
# 270, in every one of 21 steps) and that they cost ~16% of episodes, but not
# *why* they fail: a hallucinated tool name, a channel order outside the response
# schema, and a truncated JSON body are indistinguishable after the fact.
#
# Gemma4ToolRenderer flags MALFORMED on exactly one condition -- "<|tool_call>"
# present in the decoded text and no call surviving _valid_tool_calls -- so the
# decoded text is the entire diagnosis. Log the tail preferentially: the call
# sits at the end, after whatever prose ran long.
PARSE_ERROR_SAMPLE_CHARS = 2000
_TOOL_CALL_NAME = re.compile(r"<\|tool_call>call:(\w+)")


def _parse_error_logs(env, action: list[int]) -> dict[str, str | int]:
  try:
    text = env.renderer.tokenizer.decode(action, skip_special_tokens=False)
  except Exception as exc:  # a decode failure is itself the diagnosis
    return {"parse_error_decode_failed": repr(exc)}
  if len(text) <= PARSE_ERROR_SAMPLE_CHARS:
    sample = text
  else:
    head = PARSE_ERROR_SAMPLE_CHARS // 4
    tail = PARSE_ERROR_SAMPLE_CHARS - head
    sample = f"{text[:head]}\n...[{len(text) - PARSE_ERROR_SAMPLE_CHARS} chars elided]...\n{text[-tail:]}"
  return {
    "parse_error_text": sample,
    "parse_error_chars": len(text),
    "parse_error_call_tokens": text.count("<|tool_call>"),
    # Names the renderer would have had to recognise. Any name here that is not
    # a live tool means the model invented one and the fallback dropped it.
    "parse_error_call_names": ",".join(sorted(set(_TOOL_CALL_NAME.findall(text)))),
  }


# Same hole as parse_error, and worse. A generation that trips max_tokens ends
# the episode from message_env's cap branch, and StepResult.logs comes back
# empty on exactly those steps -- run38's 32,768-token blowups recorded
# `logs: {}`, nothing in the logtree, nothing in the HTML. That is the one
# failure mode the run was actually dying of: the median generation held at
# ~200 tokens from step 0 to step 14 while the share of all generated tokens
# sitting inside capped generations went 2.9% -> 84.5%. The distribution is
# bimodal, so the aggregate ac_tokens_per_turn understates it badly and the text
# that would say *why* was the text being dropped.
MAX_TOKENS_SAMPLE_CHARS = 2000
# Tokens off the end to measure repetition over. Long enough to span a loop,
# short enough that a legitimately long answer still looks varied.
REPETITION_WINDOW_TOKENS = 512


def max_tokens_logs(env, action: list[int]) -> dict[str, str | int | float]:
  try:
    text = env.renderer.tokenizer.decode(action, skip_special_tokens=False)
  except Exception as exc:
    return {"max_tokens_decode_failed": repr(exc)}
  # Keep both ends: these start coherent and degenerate, so the head says what
  # the model was trying to do and the tail says what it collapsed into.
  head = MAX_TOKENS_SAMPLE_CHARS // 2
  if len(text) <= MAX_TOKENS_SAMPLE_CHARS:
    sample = text
  else:
    tail = MAX_TOKENS_SAMPLE_CHARS - head
    sample = f"{text[:head]}\n...[{len(text) - MAX_TOKENS_SAMPLE_CHARS} chars elided]...\n{text[-tail:]}"
  # One number that separates the two ways to hit the cap. A repetition loop
  # cycles a handful of tokens and lands near 0; a long but varied answer that
  # simply ran out of budget stays high.
  window = action[-REPETITION_WINDOW_TOKENS:]
  distinct = len(set(window)) / len(window) if window else 0.0
  return {
    "max_tokens_text": sample,
    "max_tokens_chars": len(text),
    "max_tokens_tokens": len(action),
    "max_tokens_distinct_frac": round(distinct, 4),
  }


async def _step_with_flag_rates(self, action, *, extra=None):
  result = await _base_env_step(self, action, extra=extra)
  metrics = dict(result.metrics or {})
  if metrics.get("parse_error"):
    result.logs = {**(result.logs or {}), **_parse_error_logs(self, action)}
  if metrics.get("max_tokens_reached"):
    result.logs = {**(result.logs or {}), **max_tokens_logs(self, action)}
  for key in _TERMINAL_FLAG_METRICS:
    metrics.setdefault(key, 0.0)
  result.metrics = metrics
  return result


rl_message_env.EnvFromMessageEnv.step = _step_with_flag_rates

MODEL_NAME = "google/gemma-4-E4B-it"
COMMAND_TIMEOUT = 60


def default_judge_parallel(judge_model: str) -> int:
  # Self-hosted GLM absorbs concurrent grading; Gemini rate limits force 1.
  return 16 if "glm" in judge_model else 1


NUM_GROUPS_TO_LOG = 1


def print_group_summary(traj_group, tokenizer) -> None:
  rewards = traj_group.get_total_rewards()
  buf = io.StringIO()
  buf.write("====== Trajectory Group ======\n")
  for idx, traj in enumerate(traj_group.trajectories_G):
    metrics = traj_group.metrics_G[idx] or {}
    ac_tokens = sum(len(t.ac.tokens) for t in traj.transitions)
    last_ac = len(traj.transitions[-1].ac.tokens) if traj.transitions else 0
    extras = "".join(
      f" {key.rsplit('/', 1)[-1]}={metrics[key]:.3g}"
      for key in ("lab/criteria_pass_fraction", "lab/document_coverage", "lab/reward_error")
      if isinstance(metrics.get(key), (int, float))
    )
    buf.write(f"  rollout {idx}: reward={rewards[idx]:.3f} turns={len(traj.transitions)} ac_tokens={ac_tokens} last_ac={last_ac}{extras}\n")
  buf.write("====== End Trajectory Group ======")
  rl_train.logger.info(buf.getvalue())


def print_group_responses(traj_group, tokenizer) -> None:
  rewards = traj_group.get_total_rewards()
  buf = io.StringIO()
  buf.write("\n====== Trajectory Group (model responses only) ======\n")
  for idx, traj in enumerate(traj_group.trajectories_G):
    buf.write(f"****** trajectory idx={idx}, reward={rewards[idx]:.3g} ******\n")
    for key, value in (traj_group.metrics_G[idx] or {}).items():
      buf.write(f"  {key}: {value}\n")
    for turn, transition in enumerate(traj.transitions):
      buf.write(f"---- turn {turn} | ob_len={transition.ob.length} ac_len={len(transition.ac.tokens)} reward={transition.reward:.3f} ----\n")
      buf.write(tokenizer.decode(transition.ac.tokens).rstrip() + "\n")
  buf.write("====== End Trajectory Group ======")
  rl_train.logger.info(buf.getvalue())


_print_group_responses = print_group_responses


@chz.chz
class RunConfig:
  """Small set of knobs for the LAB RL experiment."""

  base_url: str | None = None
  model_name: str = MODEL_NAME
  renderer_name: str | None = None
  learning_rate: float = 3e-6
  lora_rank: int = 32
  lab_root: Path = Path(__file__).resolve().parent / "harvey-labs"
  # Single-task override; otherwise task_set picks the pools: "random" is a
  # seeded disjoint split of the whole runnable pool, "bootstrap" is the
  # curated 100-train/20-eval lists earlier runs used (comparable numbers).
  task: str | None = None
  task_set: str = "random"
  # Only used by task_set=disjoint: tasks reserved for SFT trace collection,
  # family-disjoint from the RL train pool (must match collect_traces.py).
  sft_tasks: int = 100
  train_tasks: int = 300
  eval_tasks: int = 50
  task_split_seed: int = 0
  eval_every: int = 20
  batch_size: int = 1
  rollouts_per_example: int = 4
  # Rollouts per held-out eval task, averaged. At 1 the benchmark cannot see a
  # 10pp change (see env.py); at 4 the floor is ~4.9pp. Eval cost scales with
  # it, so widen eval_every to pay for it rather than evaluating as often.
  eval_rollouts_per_task: int = 4
  max_steps: int = 40
  max_turns: int = 40
  max_tokens: int = 3072
  max_trajectory_tokens: int = 128 * 1024
  max_tool_result_tokens: int = 8 * 1024
  judge_model: str = "gemini-3.5-flash"
  # Criteria graded concurrently within one episode. 0 = auto by judge model.
  judge_parallel: int = 0
  max_reward_criteria: int | None = None
  # Full-state checkpoint cadence (weights + optimizer, resumable). 0 = off.
  save_every: int = 5
  # Overlap training with sampling: forward_backward runs on each trajectory
  # group as its rollouts finish instead of waiting for the whole batch.
  # Gradient math is unchanged at num_substeps=1 (one optim_step per batch).
  stream_minibatches: bool = False
  num_substeps: int = 1
  # Anchor the policy to the base model by penalizing per-token KL against it.
  # run38 measured what actually goes wrong, and it is not verbosity. The median
  # generation was 225 tokens at step 0 and 193 at step 14 -- flat the whole run,
  # as was p90. What grows is a bimodal tail that never terminates and burns to
  # max_tokens: 1 such generation out of 1074 at step 0, 29 out of 466 at step
  # 14, by which point 84.5% of every token generated was inside a capped one.
  # Capped episodes end at the -0.1 floor ungraded, so once most of a GRPO group
  # lands there the group's rewards are identical, every advantage is zero, and
  # the gradient vanishes -- the lock that froze run37 for five hours.
  # A KL term is the right lever precisely because it is per-token: the penalty
  # is coef * (avg_kl - per_token_kl), which still varies across tokens when
  # every episode in the group scores the same. It restores a gradient pointing
  # back at base exactly where the reward signal has gone flat. 0.0 = off,
  # which is the cookbook default and every run through run38.
  # The reference is a plain sampling client on base weights: the gateway maps a
  # base_model session to lora_path=None, so the samplers serve the base model,
  # and compute_logprobs is just sample(max_tokens=1, prompt_logprobs=True) over
  # the endpoint they already implement. No server change is needed.
  kl_penalty_coef: float = 0.0
  # Spread each token's penalty over the tokens that follow it. 0 = undiscounted.
  kl_discount_factor: float = 0.0
  # Warm-start: initialize adapter weights from an existing snapshot
  # (tinker://<model_id>/sampler_weights/<label>) with a fresh optimizer.
  # The batch counter restarts at 0 — this begins a NEW run from those
  # weights, it does not resume the old run's step position.
  load_checkpoint_path: str | None = None
  log_path: str = "artifacts/harvey-labs"
  log_full_rollouts: bool = False
  # The in-loop evals measure the model BEFORE an optimizer step (batch 0 is
  # the untrained baseline); this one runs after training, on the final
  # checkpoint. The cookbook always evals once more at the top of the last
  # iteration, even with eval_every=0.
  final_eval: bool = True
  # Batch 0's eval is the untrained baseline. It is a full test-set pass before
  # a single optimizer step, and it re-measures a number we already have
  # whenever the benchmark has not moved. Turn it back on when the task set,
  # split seed, or base model changes and the baseline is genuinely unknown.
  eval_at_step_0: bool = False


def resolve_renderer_name(config: RunConfig) -> str:
  if config.renderer_name:
    return config.renderer_name
  name = config.model_name.lower()
  if "qwen" in name:
    return "qwen3_5"
  if "gemma" in name:
    return "gemma4"
  raise ValueError(f"Cannot infer a renderer for model {config.model_name!r}; pass renderer_name explicitly.")


def preflight_grading(config: RunConfig) -> None:
  """Fail before step 0 on the grading-environment rot that poisoned runs 3/4.

  Run 4 lost 38% of its gradings to a missing LAB venv (reward.py silently
  fell back to the recipe interpreter, which cannot import anthropic) and 58%
  to a stale judge without the schema fix. Both are detectable up front.
  """
  lab_python = config.lab_root / ".venv" / "bin" / "python"
  if not lab_python.exists():
    raise RuntimeError(
      f"LAB venv not found at {lab_python}. Run setup_lab.sh so grading uses the "
      "LAB environment; without it every reward silently falls back to the recipe venv."
    )
  probe = subprocess.run(
    [str(lab_python), "-c", "from evaluation.judge import Judge; Judge._salvage_verdict"],
    cwd=str(config.lab_root),
    capture_output=True,
    text=True,
  )
  if probe.returncode != 0:
    raise RuntimeError(
      "LAB grading preflight failed — every episode would score 0 as reward_error. "
      "Missing deps mean setup_lab.sh didn't finish; a missing Judge._salvage_verdict "
      "means the LAB checkout predates the judge fix (git pull in the LAB checkout).\n"
      f"{probe.stderr.strip()}"
    )


def build_dataset_builder(config: RunConfig) -> LabDatasetBuilder:
  if config.task:
    train_names, eval_names = [config.task], []
  elif config.task_set == "bootstrap":
    train_names, eval_names = list(BOOTSTRAP_TASKS), list(EVAL_TASKS)
  elif config.task_set == "random":
    train_names, eval_names = random_task_split(
      config.lab_root, config.train_tasks, config.eval_tasks, config.task_split_seed
    )
  elif config.task_set == "family":
    train_names, eval_names = family_task_split(config.lab_root, config.train_tasks, config.eval_tasks, config.task_split_seed)
  elif config.task_set == "disjoint":
    _, train_names, eval_names = three_way_task_split(
      config.lab_root, config.sft_tasks, config.train_tasks, config.eval_tasks, config.task_split_seed
    )
  else:
    raise ValueError(f"Unknown task_set {config.task_set!r} (use 'random', 'family', 'disjoint', or 'bootstrap').")
  return LabDatasetBuilder(
    lab_root=config.lab_root,
    task_names=train_names,
    eval_task_names=eval_names,
    train_limit=None,
    eval_limit=len(eval_names) or None,
    batch_size=config.batch_size,
    group_size=config.rollouts_per_example,
    eval_group_size=config.eval_rollouts_per_task,
    model_name=config.model_name,
    renderer_name=resolve_renderer_name(config),
    max_turns=config.max_turns,
    command_timeout=COMMAND_TIMEOUT,
    judge_model=config.judge_model,
    judge_parallel=config.judge_parallel or default_judge_parallel(config.judge_model),
    max_reward_criteria=config.max_reward_criteria,
    max_trajectory_tokens=config.max_trajectory_tokens,
    max_generation_tokens=config.max_tokens,
    max_tool_result_tokens=config.max_tool_result_tokens,
  )


async def run_final_eval(train_config: rl_train.Config) -> None:
  record = checkpoint_utils.get_last_checkpoint(train_config.log_path, required_key="sampler_path")
  if record is None:
    raise RuntimeError(f"No sampler checkpoint in {train_config.log_path}/checkpoints.jsonl; cannot run the final eval.")
  _, test_dataset = await train_config.dataset_builder()
  if test_dataset is None:
    return
  batch = record.batch if record.batch is not None else train_config.max_steps or 0
  service_client = tinker.ServiceClient(base_url=train_config.base_url)
  sampling_client = service_client.create_sampling_client(model_path=record.sampler_path)
  evaluator = RLTestSetEvaluator(test_dataset, max_tokens=train_config.max_tokens)
  store = TrainingRunStore(LocalStorage(Path(train_config.log_path)))
  metrics = await rl_train.run_single_evaluation(evaluator, train_config, batch, sampling_client, "test", store=store)
  with open(Path(train_config.log_path) / "metrics.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps({"progress/batch": batch, **metrics}) + "\n")
  passed = metrics.get("test/env/harvey-labs/lab/criteria_passed")
  total = metrics.get("test/env/harvey-labs/lab/criteria_total")
  episodes = metrics.get("test/env/harvey-labs/total_episodes")
  if passed is not None and total and episodes:
    rl_train.logger.info(f"Final eval after {batch} steps: pooled criteria {passed * episodes:.0f}/{total * episodes:.0f} ({passed / total:.1%})")


def skip_batch_0_eval(run_evals):
  """Wrap run_evaluations_parallel so the batch-0 baseline pass is a no-op."""

  async def wrapped(evaluators, sampling_client, config, i_batch, store=None):
    if i_batch == 0:
      return {}
    return await run_evals(evaluators, sampling_client, config, i_batch, store=store)

  return wrapped


async def run(config: RunConfig) -> None:
  preflight_grading(config)
  rl_train.print_group = print_group_responses if config.log_full_rollouts else print_group_summary
  # All three cookbook training loops gate their in-loop evals on
  # `i_batch % eval_every == 0`, so batch 0 always evals and there is no config
  # knob for it. All three also route through run_evaluations_parallel, so
  # wrapping that one name covers every path without forking the cookbook.
  # run_final_eval calls run_single_evaluation directly and is unaffected.
  if not config.eval_at_step_0:
    rl_train.run_evaluations_parallel = skip_batch_0_eval(rl_train.run_evaluations_parallel)
  train_config = rl_train.Config(
    learning_rate=config.learning_rate,
    lora_rank=config.lora_rank,
    dataset_builder=build_dataset_builder(config),
    model_name=config.model_name,
    recipe_name="harvey_labs",
    renderer_name=resolve_renderer_name(config),
    max_tokens=config.max_tokens,
    log_path=config.log_path,
    base_url=resolve_base_url(config.base_url),
    eval_every=config.eval_every,
    save_every=config.save_every,
    max_steps=config.max_steps,
    num_groups_to_log=NUM_GROUPS_TO_LOG,
    load_checkpoint_path=config.load_checkpoint_path,
    num_substeps=config.num_substeps,
    kl_penalty_coef=config.kl_penalty_coef,
    kl_discount_factor=config.kl_discount_factor,
    # Required whenever the coefficient is on: the cookbook raises rather than
    # defaulting to the training model's base, despite what its docstring says.
    kl_reference_config=(
      rl_train.KLReferenceConfig(base_model=config.model_name)
      if config.kl_penalty_coef > 0
      else None
    ),
    stream_minibatch_config=(
      rl_train.StreamMinibatchConfig(
        groups_per_batch=config.batch_size,
        num_minibatches=config.batch_size // config.num_substeps,
      )
      if config.stream_minibatches
      else None
    ),
  )
  await rl_train.main(train_config)
  if config.final_eval:
    await run_final_eval(train_config)


def main() -> None:
  force_rich_log_colors()
  config = chz.entrypoint(RunConfig, allow_hyphens=True)
  asyncio.run(run(config))


if __name__ == "__main__":
  main()
