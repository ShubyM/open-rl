"""LAB rubric reward for terminal tool-use episodes."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tinker_cookbook.renderers.base import Message


@dataclass
class LabRubricReward:
  lab_root: Path
  run_id: str
  run_dir: Path
  task_name: str
  judge_model: str
  documents_dir: Path
  task_instructions: str
  judge_parallel: int = 1
  max_criteria: int | None = None
  tool_metrics: Callable[[], dict[str, Any]] | None = None
  config: dict[str, Any] = field(default_factory=dict)
  timeout_seconds: int = 900
  process_reward_weight: float = 0.2

  async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
    return await asyncio.to_thread(self.score, history)

  def score(self, history: list[Message]) -> tuple[float, dict[str, float]]:
    self.write_metadata(history)
    process_reward, process_metrics = self.process_reward(history)
    if not process_metrics["lab/has_output"]:
      return self.combine_rewards(0.0, process_reward), {
        **process_metrics,
        "lab/no_output": 1.0,
        "lab/criteria_pass_fraction": 0.0,
      }

    scores_path = self.run_dir / "scores.reward.json"
    cmd = [
      str(self.lab_python()),
      str(Path(__file__).with_name("score_lab_run.py")),
      "--lab-root",
      str(self.lab_root),
      "--run-id",
      self.run_id,
      "--task",
      self.task_name,
      "--judge-model",
      self.judge_model,
      "--parallel",
      str(self.judge_parallel),
      "--scores-out",
      str(scores_path),
    ]
    if self.max_criteria is not None:
      cmd += ["--max-criteria", str(self.max_criteria)]

    result = subprocess.run(
      cmd,
      cwd=str(self.lab_root),
      capture_output=True,
      text=True,
      encoding="utf-8",
      errors="replace",
      env={**os.environ, "PYTHONUNBUFFERED": "1"},
      timeout=self.timeout_seconds,
    )
    if result.returncode != 0:
      (self.run_dir / "reward_error.log").write_text(
        "COMMAND:\n" + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr,
        encoding="utf-8",
      )
      return self.combine_rewards(0.0, process_reward), {
        **process_metrics,
        "lab/reward_error": 1.0,
        "lab/criteria_pass_fraction": 0.0,
      }

    rubric_reward, rubric_metrics = reward_from_scores(json.loads(scores_path.read_text(encoding="utf-8")))
    return self.combine_rewards(rubric_reward, process_reward), {
      **process_metrics,
      **rubric_metrics,
      "lab/rubric_reward": rubric_reward,
    }

  def combine_rewards(self, rubric_reward: float, process_reward: float) -> float:
    weight = self.process_reward_weight
    return (1.0 - weight) * rubric_reward + weight * process_reward

  def process_reward(self, history: list[Message]) -> tuple[float, dict[str, float]]:
    """Give bounded credit for grounded progress without rewarding tool loops."""
    document_names = {path.name for path in self.documents_dir.rglob("*") if path.is_file()}
    read_names = set()
    for message in history:
      for tool_call in message.get("tool_calls") or []:
        if tool_call.function.name != "read":
          continue
        try:
          arguments = json.loads(tool_call.function.arguments)
        except (json.JSONDecodeError, TypeError):
          continue
        if file_path := arguments.get("file_path"):
          read_names.add(Path(str(file_path)).name)

    coverage = len(read_names & document_names) / len(document_names) if document_names else 0.0
    output_files = [path for path in (self.run_dir / "output").rglob("*") if path.is_file() and path.stat().st_size > 0]
    expected_extensions = {
      f".{extension.lower()}" for extension in re.findall(r"\.(docx|xlsx|pptx|pdf|md|txt)\b", self.task_instructions, re.IGNORECASE)
    }
    has_output = bool(output_files)
    has_valid_expected_output = any(path.suffix.lower() in expected_extensions and valid_artifact(path) for path in output_files)
    process_reward = 0.5 * coverage + 0.25 * float(has_output) + 0.25 * float(has_valid_expected_output)
    return process_reward, {
      "lab/document_coverage": coverage,
      "lab/has_output": float(has_output),
      "lab/has_valid_expected_output": float(has_valid_expected_output),
      "lab/process_reward": process_reward,
    }

  def lab_python(self) -> Path:
    candidate = self.lab_root / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)

  def write_metadata(self, history: list[Message]) -> None:
    self.run_dir.mkdir(parents=True, exist_ok=True)
    with (self.run_dir / "tinker_history.jsonl").open("w", encoding="utf-8") as f:
      for message in history:
        f.write(json.dumps(jsonable_message(message), sort_keys=True) + "\n")

    config = {
      "task": self.task_name,
      "run_id": self.run_id,
      "judge_model": self.judge_model,
      **self.config,
    }
    (self.run_dir / "config.json").write_text(
      json.dumps(config, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )
    metrics = self.tool_metrics() if self.tool_metrics else {}
    (self.run_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
    )


def reward_from_scores(scores: dict[str, Any]) -> tuple[float, dict[str, float]]:
  n_criteria = int(scores.get("n_criteria", 0) or 0)
  n_passed = int(scores.get("n_passed", 0) or 0)
  reward = (n_passed / n_criteria) if n_criteria else 0.0
  return (
    float(reward),
    {
      "lab/criteria_total": float(n_criteria),
      "lab/criteria_passed": float(n_passed),
      "lab/criteria_pass_fraction": float(reward),
      "lab/all_pass": float(bool(scores.get("all_pass"))),
    },
  )


def valid_artifact(path: Path) -> bool:
  suffix = path.suffix.lower()
  if suffix in {".md", ".txt"}:
    return path.stat().st_size > 0
  if suffix == ".pdf":
    return path.read_bytes()[:5] == b"%PDF-"
  office_roots = {
    ".docx": "word/document.xml",
    ".xlsx": "xl/workbook.xml",
    ".pptx": "ppt/presentation.xml",
  }
  if root := office_roots.get(suffix):
    try:
      with zipfile.ZipFile(path) as archive:
        return root in archive.namelist()
    except (OSError, zipfile.BadZipFile):
      return False
  return False


def jsonable_message(message: Message) -> dict[str, Any]:
  out: dict[str, Any] = {"role": message["role"], "content": message["content"]}
  for key in ("tool_call_id", "name"):
    if key in message:
      out[key] = message[key]
  if "tool_calls" in message:
    out["tool_calls"] = [tool_call.model_dump(mode="json") for tool_call in message["tool_calls"]]
  if "unparsed_tool_calls" in message:
    out["unparsed_tool_calls"] = [tool_call.model_dump(mode="json") for tool_call in message["unparsed_tool_calls"]]
  return out
