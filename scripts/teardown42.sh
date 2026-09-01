#!/usr/bin/env bash
# Tear the Gemma-4-12B stack down so the Qwen3.5-9B one can take the box.
#
# Both halves go together. The samplers hold NCCL weight-transfer engines that
# were established with the trainer ranks; those engines outlive a dead trainer
# and the next trainer to come up hangs trying to rendezvous with them.
#
# PID-based throughout: a pkill -f pattern here would also match the ssh command
# string carrying this script and kill the connection mid-teardown.
set -uo pipefail

echo "=== before ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "train.py drivers: $(pgrep -cf '[t]rain.py')"

if [ "$(pgrep -cf '[t]rain.py')" != "0" ]; then
  echo "ABORT: a train.py driver is running -- this stack is not idle." >&2
  exit 1
fi

# tmux owns every server pane; killing the session sends HUP to all of them.
tmux kill-session -t work 2>/dev/null && echo "killed tmux session work" || echo "no tmux session"
sleep 5

# Anything tmux did not take with it. vLLM forks EngineCore children that
# survive their parent shell, and torchrun's ranks survive their launcher.
STRAGGLERS="42914 43278 43279 43280 43281 42860 42862 42865 42868 \
42917 42931 42937 42938 43869 44022 43945 43952 46994 46998"
for pid in $STRAGGLERS; do kill "$pid" 2>/dev/null && echo "TERM $pid"; done

# 30s to release CUDA contexts before escalating. A SIGKILL'd vLLM can leave the
# device busy for longer than a clean shutdown would.
for _ in $(seq 1 30); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ] && break
  sleep 1
done

n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$n" -ne 0 ]; then
  echo "=== $n procs still holding GPUs after 30s, escalating to KILL ==="
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read -r p; do
    kill -9 "$p" 2>/dev/null && echo "KILL $p"
  done
  sleep 15
fi

# Sandbox containers leaked by episodes that died with run41. Each holds a
# kernel keyring, and the keyring quota is what killed run41 at launch.
LEAKED=$(podman ps -aq 2>/dev/null | wc -l)
if [ "$LEAKED" != "0" ]; then
  echo "=== removing $LEAKED leaked sandbox container(s) ==="
  podman rm -f $(podman ps -aq) 2>/dev/null | tail -3
fi

echo "=== after ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "gpu procs remaining: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
echo "containers remaining: $(podman ps -aq 2>/dev/null | wc -l)"
echo "keyring maxkeys: $(cat /proc/sys/kernel/keys/maxkeys)"
echo "tmux: $(tmux ls 2>&1)"
