import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))
sys.modules.setdefault(
  "chz",
  types.SimpleNamespace(chz=lambda cls: cls, entrypoint=lambda *_args, **_kwargs: None),
)

from rl.autoresearch.ui import observer  # noqa: E402

UI_EVENTS_FILE = "ui_events.jsonl"


def write_events(event_dir: Path, events: list[dict]) -> None:
  event_dir.mkdir(parents=True, exist_ok=True)
  with (event_dir / UI_EVENTS_FILE).open("a", encoding="utf-8") as f:
    for event in events:
      f.write(json.dumps(event) + "\n")


class TestAutoresearchUiObserver(unittest.TestCase):
  def test_agent_only_event_creates_live_row(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      agent_log = root / "researcher-a" / "agent.log"
      agent_log.parent.mkdir()
      agent_log.write_text('{"type":"init","model":"gemini"}\n', encoding="utf-8")
      write_events(
        root / "researcher-a",
        [
          {
            "time": 1,
            "work_id": "researcher-a",
            "researcher": "researcher-a",
            "kind": "activity",
            "status": "running",
            "tab": "agent",
            "path": str(agent_log),
          }
        ],
      )

      data = observer.payload(root / "runs", root / "ui")
      row = data["researchers"][0]["views"]["all"]["table_rows"][0]

      self.assertEqual(row["kind"], "live")
      self.assertTrue(row["live"])
      self.assertEqual(row["tab_order"], ["agent", "logs"])
      self.assertIn("gemini", row["tabs"]["agent"]["tail"])

  def test_attempt_events_populate_logs_diff_metric_and_chart(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_dir = root / "runs" / "researcher-a-attempt-1-baseline"
      (run_dir / "logs.log").parent.mkdir(parents=True)
      (run_dir / "logs.log").write_text("training\n", encoding="utf-8")
      (run_dir / "agent.log").write_text("agent\n", encoding="utf-8")
      (run_dir / "code.diff").write_text("diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-old\n+new\n", encoding="utf-8")
      (run_dir / "code_full.diff").write_text("full diff\n", encoding="utf-8")
      write_events(
        run_dir,
        [
          {
            "time": 1,
            "work_id": run_dir.name,
            "researcher": "researcher-a",
            "kind": "attempt",
            "status": "running",
            "order": 1,
            "description": "baseline",
            "git": {"commit": "abc1234"},
            "experiment": {"attempt": 1},
            "tab": "logs",
            "path": "logs.log",
          },
          {"time": 2, "work_id": run_dir.name, "researcher": "researcher-a", "tab": "agent", "path": "agent.log"},
          {"time": 3, "work_id": run_dir.name, "researcher": "researcher-a", "tab": "diff", "path": "code.diff"},
          {"time": 4, "work_id": run_dir.name, "researcher": "researcher-a", "tab": "diff_full", "path": "code_full.diff"},
          {
            "time": 5,
            "work_id": run_dir.name,
            "researcher": "researcher-a",
            "metric": {"name": "dev/accuracy", "label": "dev accuracy", "value": 0.5, "step": 1},
          },
          {"time": 6, "work_id": run_dir.name, "researcher": "researcher-a", "status": "completed"},
        ],
      )

      view = observer.payload(root / "runs", root / "ui")["researchers"][0]["views"]["all"]
      row = view["table_rows"][0]

      self.assertEqual(row["kind"], "attempt")
      self.assertEqual(row["number"], 1)
      self.assertEqual(row["score"], 0.5)
      self.assertEqual(row["score_label"], "Dev Accuracy")
      self.assertIn("training", row["tabs"]["logs"]["tail"])
      self.assertIn("diff --git", row["tabs"]["diff"]["compact"])
      self.assertEqual(view["chart_points"][0]["id"], run_dir.name)

  def test_running_attempt_hides_metric_until_done(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      run_dir = root / "runs" / "researcher-a-attempt-1"
      write_events(
        run_dir,
        [
          {"time": 1, "work_id": run_dir.name, "researcher": "researcher-a", "kind": "attempt", "status": "running", "order": 1},
          {
            "time": 2,
            "work_id": run_dir.name,
            "researcher": "researcher-a",
            "metric": {"name": "dev/accuracy", "label": "dev accuracy", "value": 0.5, "step": 8},
          },
        ],
      )

      view = observer.payload(root / "runs", root / "ui")["researchers"][0]["views"]["all"]
      row = view["table_rows"][0]

      self.assertIsNone(row["score"])
      self.assertEqual(row["history"], [])
      self.assertEqual(view["chart_points"], [])

      write_events(run_dir, [{"time": 3, "work_id": run_dir.name, "researcher": "researcher-a", "status": "completed"}])
      view = observer.payload(root / "runs", root / "ui")["researchers"][0]["views"]["all"]
      row = view["table_rows"][0]

      self.assertEqual(row["score"], 0.5)
      self.assertEqual(row["history"], [{"step": 8, "score": 0.5}])
      self.assertEqual(view["chart_points"][0]["score"], 0.5)

  def test_live_row_uses_running_attempt_logs(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      activity = root / "researcher-a"
      run_dir = root / "runs" / "researcher-a-attempt-1"
      (activity / "agent.log").parent.mkdir(parents=True)
      (activity / "agent.log").write_text("agent\n", encoding="utf-8")
      (run_dir / "logs.log").parent.mkdir(parents=True)
      (run_dir / "logs.log").write_text("training\n", encoding="utf-8")
      write_events(
        activity,
        [
          {
            "time": 1,
            "work_id": "researcher-a-activity",
            "researcher": "researcher-a",
            "kind": "activity",
            "status": "running",
            "tab": "agent",
            "path": "agent.log",
          }
        ],
      )
      write_events(
        run_dir,
        [
          {
            "time": 2,
            "work_id": run_dir.name,
            "researcher": "researcher-a",
            "kind": "attempt",
            "status": "running",
            "order": 1,
            "tab": "logs",
            "path": "logs.log",
          }
        ],
      )

      live = observer.payload(root / "runs", root / "ui")["researchers"][0]["views"]["all"]["table_rows"][0]

      self.assertEqual(live["kind"], "live")
      self.assertEqual(live["tab_order"], ["agent", "logs"])
      self.assertIn("agent", live["tabs"]["agent"]["tail"])
      self.assertIn("training", live["tabs"]["logs"]["tail"])
      self.assertTrue(live["tabs"]["logs"]["live"])

  def test_researchers_are_stably_numbered_by_first_seen(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      write_events(
        root / "runs" / "b-attempt-1",
        [
          {"time": 2, "work_id": "b-attempt-1", "researcher": "b", "kind": "attempt", "status": "completed", "order": 2},
        ],
      )
      write_events(
        root / "runs" / "a-attempt-1",
        [
          {"time": 1, "work_id": "a-attempt-1", "researcher": "a", "kind": "attempt", "status": "completed", "order": 1},
        ],
      )

      researchers = observer.payload(root / "runs", root / "ui")["researchers"]

      self.assertEqual([(row["id"], row["label"]) for row in researchers], [("a", "Researcher 1"), ("b", "Researcher 2")])

  def test_stale_attempt_stops_live_state(self) -> None:
    with tempfile.TemporaryDirectory() as tmp, patch.object(observer.time, "time", return_value=10000):
      root = Path(tmp)
      write_events(
        root / "runs" / "researcher-a-attempt-1",
        [
          {
            "time": 1,
            "work_id": "researcher-a-attempt-1",
            "researcher": "researcher-a",
            "kind": "attempt",
            "status": "running",
            "attempt_timeout_minutes": 0.001,
          }
        ],
      )

      researcher = observer.payload(root / "runs", root / "ui")["researchers"][0]
      row = researcher["views"]["all"]["table_rows"][0]

      self.assertEqual(row["status"], "stale")
      self.assertFalse(row["live"])
      self.assertEqual(researcher["status"], "complete")

  def test_improvements_view_only_keeps_score_improvements(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      for attempt, score in [(1, 0.1), (2, 0.05), (3, 0.2)]:
        run_dir = root / "runs" / f"researcher-a-attempt-{attempt}"
        write_events(
          run_dir,
          [
            {"time": attempt, "work_id": run_dir.name, "researcher": "researcher-a", "kind": "attempt", "order": attempt, "status": "completed"},
            {"time": attempt, "work_id": run_dir.name, "researcher": "researcher-a", "metric": {"name": "score", "label": "score", "value": score}},
          ],
        )

      view = observer.payload(root / "runs", root / "ui")["researchers"][0]["views"]["improvements"]

      self.assertEqual([point["number"] for point in view["chart_points"]], [1, 3])
      self.assertEqual([row["number"] for row in view["table_rows"]], [1, 3])

  def test_chart_keeps_attempt_count_when_middle_attempt_has_no_score(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      for attempt, score in [(1, 0.1), (2, None), (3, 0.2)]:
        run_dir = root / "runs" / f"researcher-a-attempt-{attempt}"
        events = [
          {"time": attempt, "work_id": run_dir.name, "researcher": "researcher-a", "kind": "attempt", "order": attempt, "status": "completed"}
        ]
        if score is not None:
          events.append(
            {"time": attempt, "work_id": run_dir.name, "researcher": "researcher-a", "metric": {"name": "score", "label": "score", "value": score}}
          )
        write_events(run_dir, events)

      view = observer.payload(root / "runs", root / "ui")["researchers"][0]["views"]["all"]

      self.assertEqual(view["attempt_count"], 3)
      self.assertEqual([point["number"] for point in view["chart_points"]], [1, 3])


if __name__ == "__main__":
  unittest.main()
