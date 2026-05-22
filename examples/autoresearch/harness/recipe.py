"""Recipe config loading for the autoresearch harness."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recipe:
  task: str
  command: str
  editable: list[Path]
  metric: str
  metric_label: str
  metric_mode: str


def load_recipe(path: Path) -> Recipe:
  raw = tomllib.loads(path.read_text(encoding="utf-8"))
  mode = str(raw["metric_mode"])
  if mode not in {"max", "min"}:
    raise ValueError("metric_mode must be max or min")
  return Recipe(
    task=str(raw["task"]),
    command=str(raw["command"]),
    editable=[Path(value) for value in raw["editable"]],
    metric=str(raw["metric"]),
    metric_label=str(raw["metric_label"]),
    metric_mode=mode,
  )
