"""LAB task environments for tinker-cookbook RL training."""

from __future__ import annotations

import json
import logging
import shutil
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chz
from renderer import register_gemma4_tool_renderer
from reward import LabRubricReward
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env
from tinker_cookbook.tool_use.types import Tool
from tools import build_lab_tools

logger = logging.getLogger(__name__)

DEFAULT_LAB_ROOT = Path("experiments/lab-traces/harvey-labs")
OUTPUT_INSTRUCTIONS_SUFFIX = (
    "\n\nSave every deliverable to /workspace/output using exactly the filename "
    "specified in the instructions. When finished, call the submit tool."
)
CONTEXT_OVERFLOW_REWARD = -0.1


@dataclass(frozen=True)
class LabTask:
    name: str
    instructions: str
    task_dir: Path
    documents_dir: Path


def task_slug(task_name: str) -> str:
    return task_name.replace("/", "__")


def add_lab_to_path(lab_root: Path) -> None:
    resolved = str(lab_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def discover_tasks(lab_root: Path) -> list[str]:
    tasks_root = lab_root / "tasks"
    return sorted(str(path.parent.relative_to(tasks_root)) for path in tasks_root.glob("**/task.json"))


def load_task(lab_root: Path, task_name: str) -> LabTask:
    task_dir = lab_root / "tasks" / Path(*task_name.split("/"))
    config = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    instructions = config.get("instructions")
    if not instructions:
        instructions = (task_dir / "instructions.md").read_text(encoding="utf-8")
    return LabTask(
        name=task_name,
        instructions=instructions,
        task_dir=task_dir,
        documents_dir=task_dir / "documents",
    )


def load_lab_tasks(
    lab_root: Path,
    *,
    split_path: Path | None = None,
    subset: str = "train",
    task_names: list[str] | None = None,
    limit: int | None = None,
) -> list[LabTask]:
    lab_root = lab_root.resolve()
    if task_names:
        names = task_names
    elif split_path and split_path.exists():
        names = json.loads(split_path.read_text(encoding="utf-8"))[subset]
    else:
        names = discover_tasks(lab_root)
    return [load_task(lab_root, task_name) for task_name in list(names)[:limit]]


def default_skills(lab_root: Path) -> list[str]:
    return sorted(path.parent.name for path in (lab_root / "harness" / "skills").glob("*/SKILL.md"))


def lab_system_prompt(lab_root: Path) -> str:
    prompt = (lab_root / "harness" / "system_prompt.md").read_text(encoding="utf-8")
    for skill_name in default_skills(lab_root):
        skill_path = lab_root / "harness" / "skills" / skill_name / "SKILL.md"
        prompt += f"\n\n## Skill: {skill_name}\n\n{skill_path.read_text(encoding='utf-8')}"
    return prompt


def copy_skill_scripts(lab_root: Path, workspace_dir: Path) -> None:
    for skill_name in default_skills(lab_root):
        scripts_dir = lab_root / "harness" / "skills" / skill_name / "scripts"
        if scripts_dir.exists():
            shutil.copytree(
                scripts_dir,
                workspace_dir / "skills" / skill_name / "scripts",
                dirs_exist_ok=True,
            )


def initial_messages(
    task: LabTask,
    renderer: Renderer,
    system_prompt: str,
    tools: list[Tool],
) -> list[Message]:
    return renderer.create_conversation_prefix_with_tools(
        tools=[tool.to_spec() for tool in tools],
        system_prompt=system_prompt,
    ) + [{"role": "user", "content": task.instructions + OUTPUT_INSTRUCTIONS_SUFFIX}]


def lab_renderer(model_name: str, renderer_name: str | None) -> Renderer:
    register_gemma4_tool_renderer()
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    resolved_name = renderer_name or model_info.get_recommended_renderer_name(model_name)
    return get_renderer(resolved_name, tokenizer, model_name=model_name)


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
    sandboxes: list[Any] = field(default_factory=list, init=False, repr=False)

    async def make_envs(self) -> Sequence[Env]:
        add_lab_to_path(self.lab_root)
        from harness.tools import ToolExecutor, get_all_tool_definitions
        from sandbox.sandbox import DEFAULT_IMAGE, Sandbox

        renderer = lab_renderer(self.model_name, self.renderer_name)
        system_prompt = lab_system_prompt(self.lab_root)
        lab_tool_definitions = get_all_tool_definitions()
        envs: list[Env] = []

        for _ in range(self.group_size):
            run_id = f"open-rl-harvey-labs/{task_slug(self.task.name)}/{uuid.uuid4().hex[:12]}"
            run_dir = self.lab_root / "results" / run_id
            output_dir = run_dir / "output"
            workspace_dir = run_dir / "workspace"
            output_dir.mkdir(parents=True, exist_ok=True)
            workspace_dir.mkdir(parents=True, exist_ok=True)
            copy_skill_scripts(self.lab_root, workspace_dir)

            sandbox = Sandbox(
                documents_dir=self.task.documents_dir,
                output_dir=output_dir,
                workspace_dir=workspace_dir,
                image=DEFAULT_IMAGE,
                default_timeout=self.command_timeout,
            )
            sandbox.start()
            self.sandboxes.append(sandbox)

            executor = ToolExecutor(sandbox=sandbox, shell_timeout=self.command_timeout)
            tools = build_lab_tools(executor, lab_tool_definitions)
            reward = LabRubricReward(
                lab_root=self.lab_root,
                run_id=run_id,
                run_dir=run_dir,
                task_name=self.task.name,
                judge_model=self.judge_model,
                judge_parallel=self.judge_parallel,
                max_criteria=self.max_reward_criteria,
                tool_metrics=executor.get_metrics,
                config={
                    "model": self.model_name,
                    "renderer": self.renderer_name,
                    "max_turns": self.max_turns,
                    "skills": default_skills(self.lab_root),
                },
            )
            envs.append(
                build_agent_tool_env(
                    renderer=renderer,
                    tools=tools,
                    initial_messages=initial_messages(self.task, renderer, system_prompt, tools),
                    reward_fn=reward,
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                    context_overflow_reward=CONTEXT_OVERFLOW_REWARD,
                )
            )
        return envs

    async def cleanup(self) -> None:
        for sandbox in self.sandboxes:
            try:
                sandbox.stop()
            except Exception as exc:
                logger.warning("LAB sandbox cleanup failed: %s", exc)
        self.sandboxes.clear()

    def logging_tags(self) -> list[str]:
        return ["harvey-labs", self.task.name.split("/")[0], task_slug(self.task.name)]


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
    lab_root: Path = DEFAULT_LAB_ROOT
    split_path: Path | None = None
    task_names: list[str] | None = None
    train_limit: int | None = None
    eval_limit: int | None = 1
    batch_size: int = 1
    group_size: int = 2
    model_name: str = "google/gemma-4-e4b"
    renderer_name: str | None = "gemma4"
    max_turns: int = 40
    command_timeout: int = 60
    judge_model: str = "gemini-3.5-flash"
    judge_parallel: int = 1
    max_reward_criteria: int | None = None
    max_trajectory_tokens: int = 128 * 1024
    max_generation_tokens: int | None = None

    def env_groups(self, tasks: list[LabTask], group_size: int, lab_root: Path) -> list[LabEnvGroupBuilder]:
        return [
            LabEnvGroupBuilder(
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
            )
            for task in tasks
        ]

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        lab_root = self.lab_root.resolve()
        train_tasks = load_lab_tasks(
            lab_root,
            split_path=self.split_path,
            subset="train",
            task_names=self.task_names,
            limit=self.train_limit,
        )
        if not train_tasks:
            raise ValueError("No LAB train tasks selected")

        train = LabDataset(self.env_groups(train_tasks, self.group_size, lab_root), self.batch_size)
        if not self.eval_limit:
            return train, None

        eval_tasks = load_lab_tasks(
            lab_root,
            split_path=self.split_path,
            subset="eval",
            task_names=None if self.task_names is None else self.task_names[: self.eval_limit],
            limit=self.eval_limit,
        )
        eval_dataset = LabDataset(self.env_groups(eval_tasks, 1, lab_root), self.batch_size) if eval_tasks else None
        return train, eval_dataset
