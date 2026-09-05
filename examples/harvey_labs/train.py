"""Train on Harvey LAB with live tool-use rollouts."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import chz
import tinker
from common.tinker_utils import force_rich_log_colors, resolve_base_url
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.stores.storage import LocalStorage
from tinker_cookbook.stores.training_store import TrainingRunStore

from .cookbook_compat import recipe_runtime
from .env import LabDatasetBuilder
from .results import write_report
from .sandbox import SandboxFactory, podman_sandbox_factory
from .tasks import BOOTSTRAP_TASKS, EVAL_TASKS, family_task_split, random_task_split

MODEL_NAME = "google/gemma-4-E4B-it"
COMMAND_TIMEOUT = 60


def default_judge_parallel(judge_model: str) -> int:
  # Self-hosted GLM absorbs concurrent grading; Gemini rate limits force 1.
  return 16 if "glm" in judge_model else 1


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
  # Optional per-token KL to the base model. This can retain a learning signal
  # when long, capped episodes all receive the same reward. 0 disables it.
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


def build_dataset_builder(config: RunConfig, sandbox_factory: SandboxFactory = podman_sandbox_factory) -> LabDatasetBuilder:
  if config.task:
    train_names, eval_names = [config.task], []
  elif config.task_set == "bootstrap":
    train_names, eval_names = list(BOOTSTRAP_TASKS), list(EVAL_TASKS)
  elif config.task_set == "random":
    train_names, eval_names = random_task_split(config.lab_root, config.train_tasks, config.eval_tasks, config.task_split_seed)
  elif config.task_set == "family":
    train_names, eval_names = family_task_split(config.lab_root, config.train_tasks, config.eval_tasks, config.task_split_seed)
  else:
    raise ValueError(f"Unknown task_set {config.task_set!r} (use 'random', 'family', or 'bootstrap').")
  return LabDatasetBuilder(
    lab_root=config.lab_root,
    sandbox_factory=sandbox_factory,
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
    f.write(json.dumps({**metrics, "step": batch, "progress/batch": batch, "eval_phase": "final"}) + "\n")
  passed = metrics.get("test/env/harvey-labs/lab/criteria_passed")
  total = metrics.get("test/env/harvey-labs/lab/criteria_total")
  episodes = metrics.get("test/env/harvey-labs/total_episodes")
  if passed is not None and total and episodes:
    rl_train.logger.info(f"Final eval after {batch} steps: pooled criteria {passed * episodes:.0f}/{total * episodes:.0f} ({passed / total:.1%})")


async def run(config: RunConfig, *, sandbox_factory: SandboxFactory = podman_sandbox_factory) -> None:
  preflight_grading(config)
  print(f"Read this run: uv --project examples run --no-sync python -m harvey_labs.results {config.log_path} --json", flush=True)
  train_config = rl_train.Config(
    learning_rate=config.learning_rate,
    lora_rank=config.lora_rank,
    dataset_builder=build_dataset_builder(config, sandbox_factory),
    model_name=config.model_name,
    recipe_name="harvey_labs",
    renderer_name=resolve_renderer_name(config),
    max_tokens=config.max_tokens,
    log_path=config.log_path,
    base_url=resolve_base_url(config.base_url),
    eval_every=config.eval_every,
    save_every=config.save_every,
    max_steps=config.max_steps,
    num_groups_to_log=1,
    load_checkpoint_path=config.load_checkpoint_path,
    num_substeps=config.num_substeps,
    kl_penalty_coef=config.kl_penalty_coef,
    kl_discount_factor=config.kl_discount_factor,
    # Required whenever the coefficient is on: the cookbook raises rather than
    # defaulting to the training model's base, despite what its docstring says.
    kl_reference_config=(rl_train.KLReferenceConfig(base_model=config.model_name) if config.kl_penalty_coef > 0 else None),
    stream_minibatch_config=(
      rl_train.StreamMinibatchConfig(
        groups_per_batch=config.batch_size,
        num_minibatches=config.batch_size // config.num_substeps,
      )
      if config.stream_minibatches
      else None
    ),
  )
  with recipe_runtime(config):
    try:
      await rl_train.main(train_config)
      if config.final_eval:
        await run_final_eval(train_config)
    finally:
      # Keep partial runs inspectable, and never hide a training failure behind
      # an error while creating its report.
      if (Path(config.log_path) / "metrics.jsonl").exists():
        try:
          write_report(Path(config.log_path))
        except Exception:
          rl_train.logger.exception("Could not generate the run report; read metrics with harvey_labs.results")


def main() -> None:
  force_rich_log_colors()
  config = chz.entrypoint(RunConfig, allow_hyphens=True)
  asyncio.run(run(config))


if __name__ == "__main__":
  main()
