#!/usr/bin/env python3
"""Collect teacher traces for SFT: run LAB tasks under this recipe's exact
scaffold (system prompt, tools, sandbox, output contract) with an
OpenAI-compatible chat model as the policy, grade each episode with the
usual judge, and write message-level traces to jsonl.

  uv --project examples run python examples/harvey_labs/collect_traces.py \
    endpoint=http://<glm-box>:8000/v1 teacher=zai-org/GLM-5.2-FP8 \
    split=eval rollouts_per_task=1 out=traces/glm_eval.jsonl

Traces are stored as OpenAI-style messages; sft.py re-renders them
with the student renderer, so the teacher's chat format never leaks into
training data. Doubles as a teacher eval: the summary line reports the
pooled criterion pass rate of the collected episodes.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import chz
import httpx

RECIPE_DIR = Path(__file__).resolve().parent


@chz.chz
class TraceConfig:
  endpoint: str
  teacher: str
  out: str
  lab_root: Path = RECIPE_DIR / "harvey-labs"
  api_key: str = "dummy"
  split: str = "eval"  # eval | train | sft (sft needs task_set=disjoint) | task=<name> via `task`
  task: str | None = None
  task_set: str = "random"  # random | family | disjoint (family-disjoint sft/train/eval)
  sft_tasks: int = 100
  train_tasks: int = 300
  eval_tasks: int = 50
  task_split_seed: int = 0
  # Dataset repo to auto-upload the finished traces to (gzipped, under jsonl/).
  hf_repo: str | None = None
  limit: int | None = None
  rollouts_per_task: int = 1
  parallel: int = 6
  temperature: float = 1.0
  max_turns: int = 40
  command_timeout: int = 60
  max_tokens: int = 32768
  max_trajectory_tokens: int = 163840
  max_tool_result_tokens: int = 16384
  judge_model: str = "gpt-glm-5.2"
  judge_parallel: int = 0
  # Tokenizer used for tool-result truncation, matched to the student so the
  # observations the student later trains on are truncated identically.
  student_model: str = "Qwen/Qwen3.5-9B"


def select_tasks(config: TraceConfig) -> list[str]:
  from tasks import family_task_split, random_task_split, three_way_task_split

  if config.task:
    return [config.task]
  if config.task_set == "disjoint":
    sft_names, train_names, eval_names = three_way_task_split(
      config.lab_root, config.sft_tasks, config.train_tasks, config.eval_tasks, config.task_split_seed
    )
    names = {"sft": sft_names, "train": train_names, "eval": eval_names}[config.split]
  else:
    if config.split == "sft":
      raise ValueError("split=sft requires task_set=disjoint")
    split_fn = {"random": random_task_split, "family": family_task_split}[config.task_set]
    train_names, eval_names = split_fn(config.lab_root, config.train_tasks, config.eval_tasks, config.task_split_seed)
    names = eval_names if config.split == "eval" else train_names
  return names[: config.limit] if config.limit else names


def upload_traces(repo_id: str, out_path: Path) -> None:
  import gzip
  import shutil

  from huggingface_hub import HfApi

  gz_path = out_path.with_name(out_path.name + ".gz")
  with open(out_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
    shutil.copyfileobj(src, dst)
  HfApi().upload_file(
    path_or_fileobj=str(gz_path),
    path_in_repo=f"jsonl/{gz_path.name}",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message=f"Add {gz_path.name} from collect_traces.py",
  )
  print(f"[traces] uploaded hf datasets:{repo_id}/jsonl/{gz_path.name}")


async def run_episode(config: TraceConfig, client: httpx.AsyncClient, task, tokenizer, system_prompt: str, rollout: int) -> dict:
  from harness.tools import ToolExecutor, get_all_tool_definitions
  from prompts import artifact_path_prompt, copy_skill_scripts
  from reward import LabRubricReward
  from sandbox.sandbox import DEFAULT_IMAGE, Sandbox
  from tasks import task_slug
  from tinker_cookbook.tool_use.types import ToolInput
  from tools import LabTool

  run_id = f"open-rl-traces/{task_slug(task.name)}/{uuid.uuid4().hex[:12]}"
  run_dir = config.lab_root / "results" / run_id
  output_dir = run_dir / "output"
  workspace_dir = run_dir / "workspace"
  output_dir.mkdir(parents=True, exist_ok=True)
  workspace_dir.mkdir(parents=True, exist_ok=True)
  copy_skill_scripts(config.lab_root, workspace_dir)
  sandbox = Sandbox(
    documents_dir=task.documents_dir,
    output_dir=output_dir,
    workspace_dir=workspace_dir,
    image=DEFAULT_IMAGE,
    default_timeout=config.command_timeout,
  )
  await asyncio.to_thread(sandbox.start)
  try:
    executor = ToolExecutor(sandbox=sandbox, shell_timeout=config.command_timeout)
    tools = {
      spec["name"]: LabTool(spec=dict(spec), executor=executor, tokenizer=tokenizer, max_result_tokens=config.max_tool_result_tokens)
      for spec in get_all_tool_definitions()
    }
    openai_tools = [{"type": "function", "function": tool.to_spec()} for tool in tools.values()]
    messages: list[dict] = [
      {"role": "system", "content": system_prompt + artifact_path_prompt(task)},
      {"role": "user", "content": task.instructions},
    ]
    context_tokens = sum(len(tokenizer.encode(m["content"], add_special_tokens=False)) for m in messages)
    turns = 0
    stop_reason = "no_tool_call"
    while turns < config.max_turns:
      if context_tokens + config.max_tokens > config.max_trajectory_tokens:
        stop_reason = "context_budget"
        break
      response = await client.post(
        "/chat/completions",
        json={
          "model": config.teacher,
          "messages": messages,
          "tools": openai_tools,
          "max_tokens": config.max_tokens,
          "temperature": config.temperature,
        },
      )
      response.raise_for_status()
      choice = response.json()["choices"][0]
      msg = choice["message"]
      tool_calls = msg.get("tool_calls") or []
      assistant: dict = {"role": "assistant", "content": msg.get("content") or ""}
      if tool_calls:
        assistant["tool_calls"] = [
          {
            "id": call["id"],
            "type": "function",
            "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]},
          }
          for call in tool_calls
        ]
      messages.append(assistant)
      context_tokens += len(tokenizer.encode(json.dumps(assistant), add_special_tokens=False))
      turns += 1
      if choice.get("finish_reason") == "length":
        stop_reason = "max_tokens"
        break
      if not tool_calls:
        stop_reason = "no_tool_call"
        break
      for call in assistant["tool_calls"]:
        tool = tools.get(call["function"]["name"])
        if tool is None:
          result = f"Unknown tool: {call['function']['name']}"
        else:
          try:
            arguments = json.loads(call["function"]["arguments"] or "{}")
          except json.JSONDecodeError:
            arguments = {}
          result = await asyncio.to_thread(tool._execute_bounded, ToolInput(arguments=arguments, call_id=call["id"]))
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        context_tokens += len(tokenizer.encode(result, add_special_tokens=False))
    else:
      stop_reason = "max_turns"

    judge_parallel = config.judge_parallel or (16 if "glm" in config.judge_model else 1)
    reward = LabRubricReward(
      lab_root=config.lab_root,
      run_id=run_id,
      task_name=task.name,
      judge_model=config.judge_model,
      task_instructions=task.instructions,
      judge_parallel=judge_parallel,
      max_criteria=None,
      criteria_count=task.criteria_count,
      tool_metrics=executor.get_metrics,
      config={"teacher": config.teacher, "collector": "collect_traces.py"},
    )
    history = [{"role": m["role"], "content": m.get("content", "")} for m in messages]
    total_reward, metrics = await reward(history)
    return {
      "task": task.name,
      "rollout": rollout,
      "run_id": run_id,
      "teacher": config.teacher,
      "reward": total_reward,
      "metrics": metrics,
      "stop_reason": stop_reason,
      "turns": turns,
      "context_tokens": context_tokens,
      "tool_specs": [tool.to_spec() for tool in tools.values()],
      "messages": messages,
    }
  finally:
    try:
      await asyncio.to_thread(sandbox.stop)
    except Exception as exc:
      print(f"[traces] sandbox cleanup failed for {run_id}: {exc}")


async def collect(config: TraceConfig) -> None:
  import sys

  sys.path.insert(0, str(RECIPE_DIR))
  sys.path.insert(0, str(config.lab_root.resolve()))
  from prompts import lab_system_prompt
  from tasks import load_lab_tasks
  from tinker_cookbook import tokenizer_utils

  names = select_tasks(config)
  tasks = load_lab_tasks(config.lab_root.resolve(), names, limit=None)
  tokenizer = tokenizer_utils.get_tokenizer(config.student_model)
  system_prompt = lab_system_prompt(config.lab_root.resolve())
  client = httpx.AsyncClient(
    base_url=config.endpoint.rstrip("/"),
    headers={"Authorization": f"Bearer {config.api_key}"},
    timeout=httpx.Timeout(connect=30.0, read=1800.0, write=120.0, pool=60.0),
  )
  out_path = Path(config.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  semaphore = asyncio.Semaphore(config.parallel)
  write_lock = asyncio.Lock()
  done = 0
  total = len(tasks) * config.rollouts_per_task
  started = time.monotonic()

  async def one(task, rollout: int) -> tuple[float, float, float]:
    nonlocal done
    async with semaphore:
      record = await run_episode(config, client, task, tokenizer, system_prompt, rollout)
    async with write_lock:
      with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
      done += 1
      print(
        f"[traces {done}/{total}] {task.name} r{rollout}: reward={record['reward']:.2f} "
        f"turns={record['turns']} stop={record['stop_reason']} ({(time.monotonic() - started) / 60:.0f}m)",
        flush=True,
      )
    m = record["metrics"]
    return record["reward"], m.get("lab/criteria_passed", 0.0), m.get("lab/criteria_total", 0.0)

  results = await asyncio.gather(*(one(task, r) for task in tasks for r in range(config.rollouts_per_task)))
  rewards = [r for r, _, _ in results]
  passed = sum(p for _, p, _ in results)
  total_criteria = sum(t for _, _, t in results)
  print(
    f"\n[traces] {len(results)} episodes -> {out_path}\n"
    f"[traces] teacher mean reward {sum(rewards) / len(rewards):.3f}, "
    f"pooled criteria {passed:.0f}/{total_criteria:.0f} = {passed / max(total_criteria, 1):.1%}"
  )
  if config.hf_repo:
    upload_traces(config.hf_repo, out_path)


if __name__ == "__main__":
  asyncio.run(collect(chz.entrypoint(TraceConfig, allow_hyphens=True)))
