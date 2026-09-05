"""LAB task environments for tinker-cookbook RL training."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import chz
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env

from .episode import LabEpisodeEnv
from .prompts import copy_skill_scripts, default_skills, initial_messages, lab_renderer, lab_system_prompt
from .reward import LabRubricReward
from .sandbox import LabSandbox, SandboxFactory, SandboxRequest, podman_sandbox_factory
from .tasks import LabTask, load_lab_tasks, task_slug
from .tools import LabTool

logger = logging.getLogger(__name__)


@dataclass
class LabEnvGroupBuilder(EnvGroupBuilder):
  task: LabTask
  lab_root: Path
  model_name: str
  renderer_name: str | None
  group_size: int
  max_turns: int
  command_timeout: int
  judge_model: str
  judge_parallel: int
  max_reward_criteria: int | None
  max_trajectory_tokens: int
  max_generation_tokens: int | None
  max_tool_result_tokens: int
  sandbox_factory: SandboxFactory = podman_sandbox_factory
  sandboxes: list[LabSandbox] = field(default_factory=list, init=False, repr=False)

  async def make_envs(self) -> Sequence[Env]:
    renderer = lab_renderer(self.model_name, self.renderer_name)
    system_prompt = lab_system_prompt(self.lab_root)
    skills = default_skills(self.lab_root)

    async def start_sandbox() -> tuple[str, LabSandbox]:
      run_id = f"open-rl-harvey-labs/{task_slug(self.task.name)}/{uuid.uuid4().hex[:12]}"
      run_dir = self.lab_root / "results" / run_id
      output_dir = run_dir / "output"
      workspace_dir = run_dir / "workspace"
      output_dir.mkdir(parents=True, exist_ok=True)
      workspace_dir.mkdir(parents=True, exist_ok=True)
      await asyncio.to_thread(copy_skill_scripts, self.lab_root, workspace_dir)
      sandbox = await self.sandbox_factory(
        SandboxRequest(
          lab_root=self.lab_root,
          run_id=run_id,
          documents_dir=self.task.documents_dir,
          workspace_dir=workspace_dir,
          output_dir=output_dir,
          command_timeout=self.command_timeout,
        )
      )
      self.sandboxes.append(sandbox)
      return run_id, sandbox

    # TaskGroup waits for siblings to settle on failure/cancellation. Register
    # each returned sandbox immediately, so partial startup can be cleaned up.
    try:
      async with asyncio.TaskGroup() as group:
        starts = [group.create_task(start_sandbox()) for _ in range(self.group_size)]
      return self._build_envs([task.result() for task in starts], renderer, system_prompt, skills)
    except BaseException:
      await self.cleanup()
      raise

  def _build_envs(self, started, renderer, system_prompt, skills) -> Sequence[Env]:
    criteria_count = self.task.criteria_count
    if self.max_reward_criteria is not None:
      criteria_count = min(criteria_count, self.max_reward_criteria)

    envs: list[Env] = []
    for run_id, sandbox in started:
      tool_definitions = sandbox.tool_definitions
      prefix_messages = initial_messages(self.task, renderer, system_prompt, tool_definitions)
      tools = [
        LabTool(spec=dict(spec), sandbox=sandbox, tokenizer=renderer.tokenizer, max_result_tokens=self.max_tool_result_tokens)
        for spec in tool_definitions
      ]
      reward = LabRubricReward(
        lab_root=self.lab_root,
        run_id=run_id,
        task_name=self.task.name,
        judge_model=self.judge_model,
        task_instructions=self.task.instructions,
        judge_parallel=self.judge_parallel,
        max_criteria=self.max_reward_criteria,
        criteria_count=criteria_count,
        tool_metrics=sandbox.tool_metrics,
        collect_outputs=sandbox.collect_outputs,
        config={
          "model": self.model_name,
          "renderer": self.renderer_name,
          "max_turns": self.max_turns,
          "skills": skills,
        },
      )
      envs.append(
        LabEpisodeEnv(
          build_agent_tool_env(
            renderer=renderer,
            tools=tools,
            initial_messages=prefix_messages,
            reward_fn=reward,
            max_turns=self.max_turns,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_generation_tokens=self.max_generation_tokens,
          ),
          criteria_count,
          renderer.tokenizer,
        )
      )
    return envs

  async def cleanup(self) -> None:
    for sandbox in self.sandboxes:
      try:
        await sandbox.cleanup()
      except Exception as exc:
        logger.warning("LAB sandbox cleanup failed: %s", exc)
    self.sandboxes.clear()

  def logging_tags(self) -> list[str]:
    return ["harvey-labs"]


@dataclass(frozen=True)
class LabDataset(RLDataset):
  groups: list[LabEnvGroupBuilder]
  batch_size: int

  def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
    start = index * self.batch_size
    return self.groups[start : start + self.batch_size]

  def __len__(self) -> int:
    return (len(self.groups) + self.batch_size - 1) // self.batch_size


@chz.chz
class LabDatasetBuilder(RLDatasetBuilder):
  """All knobs are required; defaults live in train.py's RunConfig."""

  lab_root: Path
  task_names: list[str]
  eval_task_names: list[str]
  train_limit: int | None
  eval_limit: int | None
  batch_size: int
  group_size: int
  eval_group_size: int
  model_name: str
  renderer_name: str | None
  max_turns: int
  command_timeout: int
  judge_model: str
  judge_parallel: int
  max_reward_criteria: int | None
  max_trajectory_tokens: int
  max_generation_tokens: int | None
  max_tool_result_tokens: int
  sandbox_factory: SandboxFactory = podman_sandbox_factory

  async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
    lab_root = self.lab_root.resolve()
    train_tasks = load_lab_tasks(lab_root, self.task_names, limit=self.train_limit)
    if not train_tasks:
      raise ValueError("No LAB train tasks selected")

    train = LabDataset([self._env_group(task, lab_root, self.group_size) for task in train_tasks], self.batch_size)
    if not self.eval_limit or not self.eval_task_names:
      return train, None
    # Held-out progress evals on the dedicated eval tasks, graded with the same
    # rubric settings as training. eval_group_size rollouts per task, averaged:
    # one rollout per task made the benchmark unreadable. Two evals of the same
    # untrained model on these same 50 tasks (run39 and run40, step 0) scored
    # 0.3267 and 0.3923, and per-task they correlated only 0.41 -- a task scored
    # twice varied as much as two different tasks. The paired difference had
    # sd 0.245, so the smallest detectable change was 9.7pp while run39's entire
    # eval range was 5.4pp. Averaging k rollouts per task divides that variance
    # by k, taking the floor to 9.7/sqrt(k) pp for k times the eval cost.
    eval_tasks = load_lab_tasks(lab_root, self.eval_task_names, limit=self.eval_limit)
    eval_dataset = LabDataset([self._env_group(task, lab_root, self.eval_group_size) for task in eval_tasks], self.batch_size)
    return train, eval_dataset

  def _env_group(self, task: LabTask, lab_root: Path, group_size: int) -> LabEnvGroupBuilder:
    return LabEnvGroupBuilder(
      task=task,
      lab_root=lab_root,
      model_name=self.model_name,
      renderer_name=self.renderer_name,
      group_size=group_size,
      max_turns=self.max_turns,
      command_timeout=self.command_timeout,
      judge_model=self.judge_model,
      judge_parallel=self.judge_parallel,
      max_reward_criteria=self.max_reward_criteria,
      max_trajectory_tokens=self.max_trajectory_tokens,
      max_generation_tokens=self.max_generation_tokens,
      max_tool_result_tokens=self.max_tool_result_tokens,
      sandbox_factory=self.sandbox_factory,
    )
