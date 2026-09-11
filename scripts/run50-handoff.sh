#!/usr/bin/env bash
# Hand-off from run49 to run50 on h200-vm; lives at ~/automodel/run50-handoff.sh there.# Wait for run49 to finish, then bring up run50 on the same stack with a fresh trainer.
log() { echo "$(date -u +%FT%TZ) $*"; }
log "handoff waiting for the run49 driver to exit"
while pgrep -f "[h]arvey-train" >/dev/null; do sleep 60; done
log "run49 driver gone; last log lines:"; tail -3 $HOME/open-rl/artifacts/box-logs/run49-qwen38-27b-automodel.log | cut -c1-160
log "installing run50 recipe files"
cp -v $HOME/run50-recipe/config.py $HOME/run50-recipe/env.py $HOME/run50-recipe/episode.py $HOME/run50-recipe/reward.py $HOME/open-rl-k8s/examples/harvey_labs/
log "restarting the trainer so run50 starts from fresh adapters"
pkill -f "[t]orch.distributed.run --standalone --nproc-per-node=4 -m server.training_requests_processor"; sleep 5
pkill -f "[s]erver.training_requests_processor"; sleep 10
pkill -9 -f "[s]erver.training_requests_processor" 2>/dev/null; sleep 5
for i in $(seq 1 30); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3 | sort -n | tail -1); [ "$used" -lt 2000 ] && break; sleep 5; done
log "trainer GPUs freed (max used ${used} MiB); relaunching in tmux work:trainer"
tmux send-keys -t work:trainer C-c; sleep 1
tmux send-keys -t work:trainer "bash scripts/trainer49.sh" C-m
for i in $(seq 1 60); do n=$(pgrep -fc "[s]erver.training_requests_processor"); [ "$n" -ge 4 ] && break; sleep 5; done
log "trainer ranks up: $n; waiting 90s for the processor loops"; sleep 90
log "starting the run50 driver in tmux work:train"
tmux send-keys -t work:train C-c; sleep 1
tmux send-keys -t work:train "bash scripts/run50-driver.sh" C-m
sleep 30; tmux capture-pane -p -t work:train | grep -v "^\s*$" | tail -5 | cut -c1-160
log "handoff done"
