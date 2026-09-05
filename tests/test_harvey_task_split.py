"""Unit tests for harvey_labs task splitting.

Runs the seeded split over a synthetic lab_root, so it needs no GPU, judge, or
LAB checkout. The property that matters is that the held-out eval set is a pure
function of the seed and never moves with the train count.

Run with the examples venv: examples/.venv/bin/python -m unittest tests.test_harvey_task_split
"""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("harvey_labs", reason="Harvey recipe tests require the examples environment")

from harvey_labs import tasks


def write_task(lab_root: Path, name: str) -> None:
  task_dir = lab_root / "tasks" / Path(name)
  (task_dir / "documents").mkdir(parents=True, exist_ok=True)
  (task_dir / "documents" / "doc.txt").write_text("x", encoding="utf-8")
  (task_dir / "task.json").write_text(json.dumps({"instructions": f"do {name}", "criteria": ["c1"]}), encoding="utf-8")


def build_lab_root(lab_root: Path) -> None:
  # Scenario families (siblings share a family) plus singletons, across areas,
  # so the family-disjointness check has something to bite on.
  for area in ("alpha", "beta", "gamma", "delta"):
    for fam in range(3):
      for scenario in (1, 2):
        write_task(lab_root, f"{area}/multi-{fam}/scenario-0{scenario}")
    for solo in range(4):
      write_task(lab_root, f"{area}/solo-{solo}")


class TaskSplitTest(unittest.TestCase):
  def setUp(self) -> None:
    self._tmp = tempfile.TemporaryDirectory()
    self.lab_root = Path(self._tmp.name)
    build_lab_root(self.lab_root)
    # Small offset so a modest synthetic pool exercises the real slice logic.
    patcher = patch.object(tasks, "EVAL_SLICE_OFFSET", 5)
    patcher.start()
    self.addCleanup(patcher.stop)

  def tearDown(self) -> None:
    self._tmp.cleanup()

  def split(self, num_train: int, num_eval: int = 3, seed: int = 0):
    return tasks.random_task_split(self.lab_root, num_train, num_eval, seed)

  def test_eval_is_deterministic_for_a_seed(self):
    train_a, eval_a = self.split(num_train=6)
    train_b, eval_b = self.split(num_train=6)
    self.assertEqual(eval_a, eval_b)
    self.assertEqual(train_a, train_b)

  def test_eval_does_not_move_with_train_count(self):
    _, eval_small = self.split(num_train=4)
    _, eval_large = self.split(num_train=9)
    # The whole point of freezing the offset: growing train leaves eval fixed.
    self.assertEqual(eval_small, eval_large)

  def test_train_and_eval_are_family_disjoint(self):
    train, eval_names = self.split(num_train=9)
    self.assertEqual(set(train) & set(eval_names), set())
    eval_families = {tasks.task_family(name) for name in eval_names}
    train_families = {tasks.task_family(name) for name in train}
    self.assertEqual(eval_families & train_families, set())

  def test_eval_slice_honours_the_offset(self):
    _, eval_names = self.split(num_train=6, num_eval=3, seed=0)
    names, _ = tasks.discover_lab_tasks(self.lab_root)
    shuffled = list(names)
    random.Random(0).shuffle(shuffled)
    self.assertEqual(eval_names, shuffled[5:8])

  def test_raises_when_pool_too_small(self):
    with self.assertRaises(ValueError):
      self.split(num_train=1000)
    with patch.object(tasks, "EVAL_SLICE_OFFSET", 10_000), self.assertRaises(ValueError):
      self.split(num_train=1)


if __name__ == "__main__":
  unittest.main()
