"""The two storage roots, resolved in one place.

open-rl writes two kinds of weights and they want opposite things from a
filesystem:

  snapshots (peft/)      Adapter-only handoffs from the trainer to the sampler.
                         Written every optim step, read immediately, pruned to
                         the last SNAPSHOT_KEEP. Nothing here is worth keeping
                         once the next one lands. Wants: fast, don't care if a
                         reboot eats it.

  checkpoints/           Adapter plus optimizer state. This is what
                         get_last_checkpoint() resumes from, so losing one
                         costs however many steps have run since. Wants:
                         durable, survives the machine.

Both defaulted to OPEN_RL_TMP_DIR (/tmp/open-rl), which gives the second kind
the durability guarantees of the first. On a spot VM that is a real data-loss
path: the instance is preempted, /tmp is cleared at boot, and checkpoints.jsonl
is left full of state_paths pointing at directories that no longer exist. Split
so each root can be pointed somewhere appropriate:

    OPEN_RL_SNAPSHOT_DIR=/dev/shm/open-rl/peft        # tmpfs, RAM speed
    OPEN_RL_CHECKPOINT_DIR=$HOME/open-rl-checkpoints  # persistent disk

Defaults are unchanged, so an unconfigured deployment behaves exactly as before.

CAUTION on /dev/shm: it is node-local. The snapshot root is the trainer ->
sampler handoff, so putting it on tmpfs is only correct when the trainer and
every sampler share a kernel. The moment samplers run in their own pods the
snapshot root must go back to a shared filesystem (RWX volume, NFS) or the
sampler will fail to find adapters the trainer swears it wrote.
"""

import os

DEFAULT_TMP_DIR = "/tmp/open-rl"


def tmp_dir() -> str:
  """Root for everything without a dedicated knob (fft/, sampler_full/)."""
  return os.getenv("OPEN_RL_TMP_DIR", DEFAULT_TMP_DIR)


def snapshot_root() -> str:
  """Sampler adapter snapshots. Ephemeral by design."""
  return os.getenv("OPEN_RL_SNAPSHOT_DIR") or os.path.join(tmp_dir(), "peft")


def checkpoint_root() -> str:
  """Training state with optimizer. Must outlive the process and the machine."""
  return os.getenv("OPEN_RL_CHECKPOINT_DIR") or os.path.join(tmp_dir(), "checkpoints")


def describe_roots() -> str:
  """One-line summary for startup logs.

  A misconfigured checkpoint root is invisible until something goes wrong, and
  by then the checkpoints that would have proved it are the missing evidence.
  Cheap to print it once at startup instead.
  """
  return f"storage roots: snapshots={snapshot_root()} checkpoints={checkpoint_root()}"
