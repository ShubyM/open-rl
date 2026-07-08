"""Train Gemma 4 on Harvey LAB with live tool-use rollouts."""

from __future__ import annotations

import asyncio
from pathlib import Path

import chz
from tinker_cookbook.rl import train as rl_train
from tinker_utils import LimitedDatasetBuilder, force_rich_log_colors, resolve_base_url

from recipes.harvey_labs.env import LabDatasetBuilder
from recipes.harvey_labs.renderer import register_gemma4_tool_renderer

MODEL_NAME = "gemma-4-e4b"
RENDERER_NAME = "gemma4"
LORA_RANK = 32
LEARNING_RATE = 3e-6
TEMPERATURE = 1.0
LOSS_FN = "importance_sampling"
MAX_TRAJECTORY_TOKENS = 128 * 1024
COMMAND_TIMEOUT = 60
JUDGE_PARALLEL = 1
NUM_GROUPS_TO_LOG = 1


@chz.chz
class RunConfig:
    """Small set of knobs for the LAB RL experiment."""

    base_url: str | None = None
    lab_root: Path = Path("experiments/lab-traces/harvey-labs")
    split_path: Path | None = None
    task: str | None = None
    train_limit: int | None = 2
    eval_limit: int | None = 1
    batch_size: int = 1
    rollouts_per_example: int = 2
    max_steps: int = 1
    max_turns: int = 40
    max_tokens: int = 1024
    judge_model: str = "gemini-3.5-flash"
    max_reward_criteria: int | None = None
    log_path: str = "artifacts/harvey-labs"


def build_dataset_builder(config: RunConfig) -> LabDatasetBuilder:
    return LabDatasetBuilder(
        lab_root=config.lab_root,
        split_path=config.split_path,
        task_names=[config.task] if config.task else None,
        train_limit=config.train_limit,
        eval_limit=config.eval_limit,
        batch_size=config.batch_size,
        group_size=config.rollouts_per_example,
        model_name=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        max_turns=config.max_turns,
        command_timeout=COMMAND_TIMEOUT,
        judge_model=config.judge_model,
        judge_parallel=JUDGE_PARALLEL,
        max_reward_criteria=config.max_reward_criteria,
        max_trajectory_tokens=MAX_TRAJECTORY_TOKENS,
        max_generation_tokens=config.max_tokens,
    )


async def run(config: RunConfig) -> None:
    register_gemma4_tool_renderer(RENDERER_NAME)
    builder = LimitedDatasetBuilder(
        build_dataset_builder(config),
        max_batches=config.max_steps,
        max_eval_batches=config.eval_limit,
    )
    train_config = rl_train.Config(
        learning_rate=LEARNING_RATE,
        dataset_builder=builder,
        model_name=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        lora_rank=LORA_RANK,
        max_tokens=config.max_tokens,
        temperature=TEMPERATURE,
        log_path=config.log_path,
        base_url=resolve_base_url(config.base_url),
        eval_every=0,
        save_every=0,
        max_steps=config.max_steps,
        loss_fn=LOSS_FN,
        num_substeps=1,
        kl_penalty_coef=0.0,
        kl_discount_factor=0.0,
        remove_constant_reward_groups=False,
        num_groups_to_log=NUM_GROUPS_TO_LOG,
    )
    await rl_train.main(train_config)


def main() -> None:
    force_rich_log_colors()
    config = chz.entrypoint(RunConfig, allow_hyphens=True)
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
