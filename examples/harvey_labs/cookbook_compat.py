"""Scoped workarounds for cookbook 0.5.7 knobs without public configuration hooks."""

import io
import logging
import re
from contextlib import contextmanager

import tinker
from tinker.lib.public_interfaces import training_client as tinker_training_client
from tinker_cookbook.rl import train as rl_train

# Cookbook has no Config knob for AdamParams.grad_clip_norm. Scope this
# override to the run; the threshold matches the recipe's stable run27 setting.
GRAD_CLIP_NORM = 1e5

_BaseAdamParams = tinker.AdamParams


class _AdamParamsWithClip(_BaseAdamParams):
  def __init__(self, **kwargs):
    kwargs.setdefault("grad_clip_norm", GRAD_CLIP_NORM)
    super().__init__(**kwargs)


def print_group_summary(traj_group, tokenizer) -> None:
  rewards = traj_group.get_total_rewards()
  buf = io.StringIO()
  buf.write("====== Trajectory Group ======\n")
  for idx, traj in enumerate(traj_group.trajectories_G):
    metrics = traj_group.metrics_G[idx] or {}
    ac_tokens = sum(len(t.ac.tokens) for t in traj.transitions)
    last_ac = len(traj.transitions[-1].ac.tokens) if traj.transitions else 0
    extras = "".join(
      f" {key.rsplit('/', 1)[-1]}={metrics[key]:.3g}"
      for key in ("lab/criteria_pass_fraction", "lab/document_coverage", "lab/reward_error")
      if isinstance(metrics.get(key), int | float)
    )
    buf.write(f"  rollout {idx}: reward={rewards[idx]:.3f} turns={len(traj.transitions)} ac_tokens={ac_tokens} last_ac={last_ac}{extras}\n")
  buf.write("====== End Trajectory Group ======")
  rl_train.logger.info(buf.getvalue())


def skip_batch_0_eval(run_evals):
  """Wrap run_evaluations_parallel so the batch-0 baseline pass is a no-op."""

  async def wrapped(evaluators, sampling_client, config, i_batch, store=None):
    if i_batch == 0:
      return {}
    return await run_evals(evaluators, sampling_client, config, i_batch, store=store)

  return wrapped


class StreamingProgressFilter(logging.Filter):
  """Display one-based item ordinals; cookbook's metric/trace IDs stay zero-based.

  This message precedes enqueueing, so 8/8 means 'starting minibatch eight',
  not that eight forward/backward calls have completed.
  """

  pattern = re.compile(r"^(\[stream_minibatch\] Step )(\d+)(, Substep )(\d+)/(\d+)(, Minibatch )(\d+)/(\d+)(:.*)$")

  def filter(self, record: logging.LogRecord) -> bool:
    match = self.pattern.match(record.getMessage())
    if match:
      a, batch, b, substep, substeps, c, minibatch, minibatches, rest = match.groups()
      record.msg = f"{a}{int(batch) + 1}{b}{int(substep) + 1}/{substeps}{c}{int(minibatch) + 1}/{minibatches}{rest}"
      record.args = ()
    return True


@contextmanager
def recipe_runtime(config):
  """Install cookbook workarounds only for the lifetime of this recipe run."""
  previous = (
    tinker.AdamParams,
    tinker_training_client.MAX_CHUNK_BYTES_COUNT,
    rl_train.print_group,
    rl_train.run_evaluations_parallel,
  )
  progress = StreamingProgressFilter()
  try:
    tinker.AdamParams = _AdamParamsWithClip
    # Batch several long-context datums per request for data-parallel sharding.
    tinker_training_client.MAX_CHUNK_BYTES_COUNT = 30_000_000
    if not config.log_full_rollouts:
      rl_train.print_group = print_group_summary
    if not config.eval_at_step_0:
      rl_train.run_evaluations_parallel = skip_batch_0_eval(rl_train.run_evaluations_parallel)
    rl_train.logger.addFilter(progress)
    yield
  finally:
    rl_train.logger.removeFilter(progress)
    (
      tinker.AdamParams,
      tinker_training_client.MAX_CHUNK_BYTES_COUNT,
      rl_train.print_group,
      rl_train.run_evaluations_parallel,
    ) = previous
