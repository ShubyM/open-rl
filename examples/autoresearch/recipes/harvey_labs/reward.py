"""LAB rubric reward for terminal tool-use episodes."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
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
    judge_parallel: int = 1
    max_criteria: int | None = None
    tool_metrics: Callable[[], dict[str, Any]] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 900

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        return await asyncio.to_thread(self.score, history)

    def score(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        self.write_metadata(history)
        if not any(path.is_file() for path in (self.run_dir / "output").rglob("*")):
            return 0.0, {"lab/no_output": 1.0, "lab/criteria_pass_fraction": 0.0}

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
                "COMMAND:\n"
                + " ".join(cmd)
                + "\n\nSTDOUT:\n"
                + result.stdout
                + "\n\nSTDERR:\n"
                + result.stderr,
                encoding="utf-8",
            )
            return 0.0, {"lab/reward_error": 1.0, "lab/criteria_pass_fraction": 0.0}

        return reward_from_scores(json.loads(scores_path.read_text(encoding="utf-8")))

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


def jsonable_message(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message["role"], "content": message["content"]}
    for key in ("tool_call_id", "name"):
        if key in message:
            out[key] = message[key]
    if "tool_calls" in message:
        out["tool_calls"] = [tool_call.model_dump(mode="json") for tool_call in message["tool_calls"]]
    if "unparsed_tool_calls" in message:
        out["unparsed_tool_calls"] = [
            tool_call.model_dump(mode="json") for tool_call in message["unparsed_tool_calls"]
        ]
    return out
