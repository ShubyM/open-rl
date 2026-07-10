#!/usr/bin/env python3
"""Collect all-pass Harvey LAB teacher traces with retry feedback."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

STOP_REQUESTED = threading.Event()
MANIFEST_LOCK = threading.Lock()
FILTERED_WORDS = re.compile(r"\b(criterion|criteria|rubric|judge|verdict|pass|passed|fail|failed)\b", re.IGNORECASE)
OUTPUT_INSTRUCTIONS_SUFFIX = "\n\nSave every deliverable to /workspace/output using exactly the filename specified in the instructions."


class AttemptEvaluationError(Exception):
  def __init__(self, run_dir: Path, metrics: dict[str, Any], cause: Exception):
    super().__init__(repr(cause))
    self.run_dir = run_dir
    self.metrics = metrics
    self.cause = cause


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Collect Harvey LAB teacher traces.")
  parser.add_argument("--lab-root", default="./harvey-labs")
  parser.add_argument("--split", default="split.json")
  parser.add_argument("--subset", choices=("train", "eval"), default="train")
  parser.add_argument("--task", default=None, help="Single task, e.g. banking-finance/some-task")
  parser.add_argument("--model", default="gemini-3.5-flash")
  parser.add_argument("--judge-model", default="gemini-3.5-flash")
  parser.add_argument("--max-attempts", type=int, default=4)
  parser.add_argument("--max-turns", type=int, default=120)
  parser.add_argument("--reasoning-effort", default="medium")
  parser.add_argument("--out-dir", default="./traces")
  parser.add_argument("--parallel", type=int, default=4)
  parser.add_argument("--limit", type=int, default=None)
  return parser.parse_args()


def install_sigint_handler() -> None:
  previous = signal.getsignal(signal.SIGINT)

  def handler(signum: int, frame: Any) -> None:
    STOP_REQUESTED.set()
    if callable(previous):
      previous(signum, frame)

  signal.signal(signal.SIGINT, handler)


def import_lab(lab_root: Path):
  sys.path.insert(0, str(lab_root.resolve()))
  from evaluation.judge import Judge
  from evaluation.run_eval import evaluate_run
  from harness.agent_loop import run_agent
  from harness.run import (
    DEFAULT_SKILLS,
    SYSTEM_PROMPT_PREAMBLE,
    Sandbox,
    ToolExecutor,
    _load_env,
    create_adapter,
    get_all_tool_definitions,
    load_skills,
    load_task,
    setup_skill_scripts,
  )
  from sandbox.sandbox import DEFAULT_IMAGE

  return SimpleNamespace(
    DEFAULT_IMAGE=DEFAULT_IMAGE,
    DEFAULT_SKILLS=DEFAULT_SKILLS,
    Judge=Judge,
    Sandbox=Sandbox,
    SYSTEM_PROMPT_PREAMBLE=SYSTEM_PROMPT_PREAMBLE,
    ToolExecutor=ToolExecutor,
    _load_env=_load_env,
    create_adapter=create_adapter,
    evaluate_run=evaluate_run,
    get_all_tool_definitions=get_all_tool_definitions,
    load_skills=load_skills,
    load_task=load_task,
    run_agent=run_agent,
    setup_skill_scripts=setup_skill_scripts,
  )


def load_tasks(args: argparse.Namespace) -> list[str]:
  if args.task:
    return [args.task]
  split = json.loads(Path(args.split).read_text(encoding="utf-8"))
  tasks = list(split[args.subset])
  if args.limit is not None:
    tasks = tasks[: args.limit]
  return tasks


def task_slug(task: str) -> str:
  return task.replace("/", "__")


def read_kept_tasks(manifest_path: Path) -> set[str]:
  kept = set()
  if not manifest_path.exists():
    return kept
  for line in manifest_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    try:
      entry = json.loads(line)
    except json.JSONDecodeError:
      continue
    if entry.get("status") == "kept":
      kept.add(entry.get("task"))
  return kept


def append_manifest(manifest_path: Path, entry: dict[str, Any]) -> None:
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  line = json.dumps(entry, sort_keys=True)
  with MANIFEST_LOCK, manifest_path.open("a", encoding="utf-8") as f:
    f.write(line + "\n")
    f.flush()


def sanitize_feedback(text: str) -> str:
  text = FILTERED_WORDS.sub("", text)
  text = re.sub(r"[ \t]{2,}", " ", text)
  return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_feedback(failed: list[dict[str, Any]]) -> str:
  lines = ["Before finalizing, ensure the following points are fully addressed:"]
  for item in failed:
    title = sanitize_feedback(str(item.get("title", "")))
    reasoning = sanitize_feedback(str(item.get("reasoning", "")))
    if title and reasoning:
      lines.append(f"- {title}: {reasoning}")
    elif title:
      lines.append(f"- {title}")
    elif reasoning:
      lines.append(f"- {reasoning}")
  return "\n\n" + "\n".join(lines)


def read_run_metrics(run_dir: Path) -> dict[str, Any]:
  metrics_path = run_dir / "metrics.json"
  if not metrics_path.exists():
    return {}
  try:
    return json.loads(metrics_path.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
    return {}


def token_counts(metrics: dict[str, Any]) -> tuple[int, int]:
  return (
    int(metrics.get("input_tokens", 0) or 0),
    int(metrics.get("output_tokens", 0) or 0),
  )


def evaluate_with_retry(lab, run_id: str, task_name: str, judge) -> dict[str, Any]:
  last_err: Exception | None = None
  for attempt in range(2):
    try:
      return lab.evaluate_run(run_id=run_id, task=task_name, judge=judge, parallel=1)
    except Exception as exc:
      last_err = exc
      if attempt == 0:
        time.sleep(30)
  assert last_err is not None
  raise last_err


def run_attempt(
  lab,
  lab_root: Path,
  task_name: str,
  instructions_suffix: str,
  attempt: int,
  args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
  lab._load_env()
  task = lab.load_task(task_name)
  run_id = f"lab-traces/{task_slug(task_name)}/attempt-{attempt}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
  results_dir = lab_root / "results" / run_id
  output_dir = results_dir / "output"
  workspace_dir = results_dir / "workspace"
  output_dir.mkdir(parents=True, exist_ok=True)
  workspace_dir.mkdir(parents=True, exist_ok=True)

  skill_names = lab.DEFAULT_SKILLS
  sandbox = lab.Sandbox(
    documents_dir=Path(task["docs_dir"]),
    output_dir=output_dir,
    workspace_dir=workspace_dir,
    image=lab.DEFAULT_IMAGE,
    default_timeout=60,
  )
  sandbox.start()
  try:
    config = {
      "model": args.model,
      "task": task_name,
      "run_id": run_id,
      "max_turns": args.max_turns,
      "temperature": 0.0,
      "shell_timeout": 60,
      "reasoning_effort": args.reasoning_effort,
      "skills": skill_names,
      "sandbox_image": lab.DEFAULT_IMAGE,
      "started_at": datetime.now(UTC).isoformat(),
      "teacher_retry_attempt": attempt,
      "instructions_augmented": bool(instructions_suffix),
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    adapter = lab.create_adapter(
      model=args.model,
      temperature=0.0,
      reasoning_effort=args.reasoning_effort,
    )
    tool_executor = lab.ToolExecutor(sandbox=sandbox, shell_timeout=60)
    system_prompt = lab.SYSTEM_PROMPT_PREAMBLE
    if skill_names:
      system_prompt += lab.load_skills(skill_names)
      lab.setup_skill_scripts(skill_names, workspace_dir)

    result = lab.run_agent(
      adapter=adapter,
      system_prompt=system_prompt,
      user_prompt=task["instructions"] + OUTPUT_INSTRUCTIONS_SUFFIX + instructions_suffix,
      tool_executor=tool_executor,
      tools=lab.get_all_tool_definitions(),
      max_turns=args.max_turns,
      transcript_path=str(results_dir / "transcript.jsonl"),
    )
  finally:
    sandbox.stop()

  metrics = {
    "model": args.model,
    "task": task_name,
    "run_id": run_id,
    "turn_count": result["turn_count"],
    "input_tokens": result["input_tokens"],
    "output_tokens": result["output_tokens"],
    "total_tokens": result["input_tokens"] + result["output_tokens"],
    "wall_clock_seconds": result["wall_clock_seconds"],
    "finished_cleanly": result["finished_cleanly"],
    "completed_at": datetime.now(UTC).isoformat(),
    **result["tool_metrics"],
  }
  (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

  judge = lab.Judge(model=args.judge_model)
  try:
    scores = evaluate_with_retry(lab, run_id, task_name, judge)
  except Exception as exc:
    raise AttemptEvaluationError(results_dir, read_run_metrics(results_dir) or metrics, exc) from exc
  return results_dir, metrics, scores


def keep_attempt(task_name: str, run_dir: Path, scores: dict[str, Any], args: argparse.Namespace, attempts_used: int) -> None:
  dest = Path(args.out_dir) / task_slug(task_name)
  dest.mkdir(parents=True, exist_ok=True)
  required = ["full_transcript.jsonl", "config.json", "metrics.json", "scores.json"]
  for name in required:
    src = run_dir / name
    if not src.exists():
      raise FileNotFoundError(f"kept run is missing {src}")
    shutil.copy2(src, dest / name)

  meta = {
    "task": task_name,
    "attempts_used": attempts_used,
    "model": args.model,
    "judge_model": args.judge_model,
    "criteria_total": scores.get("n_criteria", 0),
    "criteria_passed": scores.get("n_passed", 0),
    "kept_attempt_run_id": str(run_dir.relative_to(Path(args.lab_root) / "results")),
    "augmented": attempts_used > 1,
  }
  (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def collect_one(lab, lab_root: Path, task_name: str, args: argparse.Namespace, manifest_path: Path) -> None:
  start = time.time()
  attempts = 0
  input_tokens = 0
  output_tokens = 0
  feedback = ""
  try:
    for attempt in range(1, args.max_attempts + 1):
      if STOP_REQUESTED.is_set():
        break
      attempts = attempt
      run_dir, metrics, scores = run_attempt(lab, lab_root, task_name, feedback, attempt, args)
      metrics = read_run_metrics(run_dir) or metrics
      attempt_input_tokens, attempt_output_tokens = token_counts(metrics)
      input_tokens += attempt_input_tokens
      output_tokens += attempt_output_tokens
      if scores.get("all_pass"):
        keep_attempt(task_name, run_dir, scores, args, attempt)
        append_manifest(
          manifest_path,
          {
            "task": task_name,
            "status": "kept",
            "attempts": attempts,
            "wall_s": round(time.time() - start, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
          },
        )
        return
      failed = [c for c in scores.get("criteria_results", []) if c.get("verdict") != "pass"]
      feedback = build_feedback(failed)

    append_manifest(
      manifest_path,
      {
        "task": task_name,
        "status": "exhausted",
        "attempts": attempts,
        "wall_s": round(time.time() - start, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
      },
    )
  except AttemptEvaluationError as exc:
    attempt_input_tokens, attempt_output_tokens = token_counts(exc.metrics)
    input_tokens += attempt_input_tokens
    output_tokens += attempt_output_tokens
    append_manifest(
      manifest_path,
      {
        "task": task_name,
        "status": "error",
        "attempts": attempts,
        "wall_s": round(time.time() - start, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": repr(exc.cause),
      },
    )
  except Exception as exc:
    append_manifest(
      manifest_path,
      {
        "task": task_name,
        "status": "error",
        "attempts": attempts,
        "wall_s": round(time.time() - start, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": repr(exc),
      },
    )


def main() -> int:
  args = parse_args()
  install_sigint_handler()
  lab_root = Path(args.lab_root)
  out_dir = Path(args.out_dir)
  manifest_path = out_dir / "manifest.jsonl"
  kept = read_kept_tasks(manifest_path)
  tasks = [task for task in load_tasks(args) if task not in kept]
  if not tasks:
    print("no tasks to collect")
    return 0

  lab = import_lab(lab_root)
  out_dir.mkdir(parents=True, exist_ok=True)
  print(f"collecting {len(tasks)} task(s) with parallel={args.parallel}")
  with ThreadPoolExecutor(max_workers=max(args.parallel, 1)) as pool:
    futures = [pool.submit(collect_one, lab, lab_root, task, args, manifest_path) for task in tasks]
    try:
      for future in as_completed(futures):
        future.result()
        if STOP_REQUESTED.is_set():
          break
    except KeyboardInterrupt:
      STOP_REQUESTED.set()
      for future in futures:
        future.cancel()
      raise
  return 130 if STOP_REQUESTED.is_set() else 0


if __name__ == "__main__":
  raise SystemExit(main())
