import asyncio
import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("harvey_labs", reason="Harvey recipe tests require the examples environment")

from harvey_labs import train
from harvey_labs.cookbook_compat import StreamingProgressFilter, recipe_runtime
from harvey_labs.plot_run import plot_results
from harvey_labs.results import eval_result, load_metrics, read_results
from harvey_labs.train import RunConfig, rl_train, tinker


def write_metrics(path, rows):
  path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_pre_update_and_final_eval_share_the_plot_and_json_reader(tmp_path):
  prefix = "test/env/harvey-labs/"
  write_metrics(
    tmp_path / "metrics.jsonl",
    [
      {
        "step": 0,
        "env/harvey-labs/reward/total": 0.2,
        prefix + "lab/criteria_passed": 1,
        prefix + "lab/criteria_total": 4,
        prefix + "total_episodes": 2,
      },
      {"step": 1, "env/all/reward/total": 0.4},
      # Older final-eval records have no 'step' or explicit phase marker.
      {"progress/batch": 2, prefix + "lab/criteria_passed": 3, prefix + "lab/criteria_total": 4, prefix + "total_episodes": 2},
    ],
  )
  result = read_results(tmp_path)
  assert [row["step"] for row in result["train"]] == [1, 2]
  assert [row["step"] for row in result["evaluations"]] == [0, 2]
  assert result["latest_eval"]["pass_rate"] == 0.75
  assert result["latest_eval"]["criteria_passed"] == 6
  assert result["latest_eval"]["criteria_total"] == 8
  assert result["final_eval"] is None  # A last evaluation alone doesn't prove a completed run.
  assert plot_results(tmp_path, results=result).read_bytes().startswith(b"\x89PNG")


def test_final_eval_zero_scores_and_specific_namespace(tmp_path):
  write_metrics(
    tmp_path / "metrics.jsonl",
    [
      {
        "step": 8,
        "progress/batch": 8,
        "eval_phase": "final",
        "test/env/all/lab/criteria_pass_fraction": 0.9,
        "test/env/harvey-labs/lab/criteria_passed": 0,
        "test/env/harvey-labs/lab/criteria_total": 5,
        "test/env/harvey-labs/total_episodes": 4,
      }
    ],
  )
  result = read_results(tmp_path)
  assert result["final_eval"]["pass_rate"] == 0
  assert result["final_eval"]["criteria_total"] == 20
  assert result["last_train"] is None


def test_legacy_episode_average_is_labeled():
  result = eval_result({"test/env/all/lab/criteria_pass_fraction": 0.5})
  assert result["aggregation"] == "mean_episode_fraction"
  assert result["criteria_total"] is None


def test_live_partial_record_is_ignored_but_complete_corruption_is_not(tmp_path):
  path = tmp_path / "metrics.jsonl"
  path.write_text('{"step": 0}\n{"step":')
  assert load_metrics(path) == [{"step": 0}]
  path.write_text('{"step": 0}\n{"step":\n')
  with pytest.raises(json.JSONDecodeError):
    load_metrics(path)
  with pytest.raises(FileNotFoundError):
    read_results(Path(tmp_path / "missing"))


def test_streaming_progress_reaches_eight_of_eight():
  progress = StreamingProgressFilter()
  for index in range(8):
    record = logging.makeLogRecord({"msg": "[stream_minibatch] Step 0, Substep 0/1, Minibatch %s/8: Will train on minibatch", "args": (index,)})
    assert progress.filter(record)
    assert record.getMessage() == f"[stream_minibatch] Step 1, Substep 1/1, Minibatch {index + 1}/8: Will train on minibatch"
  record = logging.makeLogRecord({"msg": "progress/batch: 0"})
  progress.filter(record)
  assert record.getMessage() == "progress/batch: 0"


def test_recipe_patches_are_restored_after_failure():
  adam = tinker.AdamParams
  evaluations = rl_train.run_evaluations_parallel
  filters = list(rl_train.logger.filters)
  with pytest.raises(RuntimeError), recipe_runtime(RunConfig()):
    assert tinker.AdamParams(learning_rate=1e-4).grad_clip_norm == 1e5
    assert rl_train.run_evaluations_parallel is not evaluations
    raise RuntimeError("training failed")
  assert tinker.AdamParams is adam
  assert rl_train.run_evaluations_parallel is evaluations
  assert rl_train.logger.filters == filters
  with recipe_runtime(RunConfig(eval_at_step_0=True)):
    assert rl_train.run_evaluations_parallel is evaluations


def test_training_failure_still_writes_a_consumable_report(tmp_path, monkeypatch):
  monkeypatch.setattr(train, "preflight_grading", lambda config: None)
  monkeypatch.setattr(train, "build_dataset_builder", lambda config, sandbox_factory: None)
  monkeypatch.setattr(train, "resolve_renderer_name", lambda config: "qwen3_5")

  async def fail_after_one_batch(config):
    write_metrics(tmp_path / "metrics.jsonl", [{"step": 0, "env/all/reward/total": 0.25}])
    raise RuntimeError("original training failure")

  monkeypatch.setattr(rl_train, "main", fail_after_one_batch)
  with pytest.raises(RuntimeError, match="original training failure"):
    asyncio.run(train.run(RunConfig(log_path=str(tmp_path))))
  report = json.loads((tmp_path / "results.json").read_text())
  assert report["last_train"] == {"step": 1, "reward": 0.25}
  assert report["final_eval"] is None
  assert (tmp_path / "run_plot.png").read_bytes().startswith(b"\x89PNG")
