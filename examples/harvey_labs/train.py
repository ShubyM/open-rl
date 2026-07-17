"""Train on Harvey LAB with live tool-use rollouts."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import chz
from env import LabDatasetBuilder
from tasks import BOOTSTRAP_TASKS, EVAL_TASKS
from tinker_cookbook.rl import train as rl_train
from tinker_utils import force_rich_log_colors, resolve_base_url

MODEL_NAME = "google/gemma-4-E4B-it"
COMMAND_TIMEOUT = 60
JUDGE_PARALLEL = 1
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
  task: str | None = None
  train_limit: int | None = None
  eval_limit: int | None = 20
  eval_every: int = 20
  batch_size: int = 1
  rollouts_per_example: int = 4
  max_steps: int = 40
  max_turns: int = 40
  max_tokens: int = 3072
  max_trajectory_tokens: int = 128 * 1024
  max_tool_result_tokens: int = 8 * 1024
  judge_model: str = "gemini-3.5-flash"
  max_reward_criteria: int | None = None
  log_path: str = "artifacts/harvey-labs"
  log_full_rollouts: bool = False


def resolve_renderer_name(config: RunConfig) -> str:
  if config.renderer_name:
    return config.renderer_name
  name = config.model_name.lower()
  if "qwen" in name:
    return "qwen3_5"
  if "gemma" in name:
    return "gemma4"
  raise ValueError(f"Cannot infer a renderer for model {config.model_name!r}; pass renderer_name explicitly.")


def build_dataset_builder(config: RunConfig) -> LabDatasetBuilder:
  return LabDatasetBuilder(
    lab_root=config.lab_root,
    task_names=[config.task] if config.task else list(BOOTSTRAP_TASKS),
    eval_task_names=list(EVAL_TASKS),
    train_limit=config.train_limit,
    eval_limit=config.eval_limit,
    batch_size=config.batch_size,
    group_size=config.rollouts_per_example,
    model_name=config.model_name,
    renderer_name=resolve_renderer_name(config),
    max_turns=config.max_turns,
    command_timeout=COMMAND_TIMEOUT,
    judge_model=config.judge_model,
    judge_parallel=JUDGE_PARALLEL,
    max_reward_criteria=config.max_reward_criteria,
    max_trajectory_tokens=config.max_trajectory_tokens,
    max_generation_tokens=config.max_tokens,
    max_tool_result_tokens=config.max_tool_result_tokens,
  )


async def run(config: RunConfig) -> None:
  rl_train.print_group = print_group_responses if config.log_full_rollouts else print_group_summary
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
    save_every=0,
    max_steps=config.max_steps,
    num_groups_to_log=NUM_GROUPS_TO_LOG,
  )
  await rl_train.main(train_config)


def main() -> None:
  force_rich_log_colors()
  config = chz.entrypoint(RunConfig, allow_hyphens=True)
  asyncio.run(run(config))


if __name__ == "__main__":
  main()
