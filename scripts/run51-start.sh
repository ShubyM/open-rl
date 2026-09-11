#!/usr/bin/env bash
# Bring up run51 end to end on h200-vm: fresh tmux stack from scripts/launch51.sh,
# wait for the four samplers and the trainer ranks, then start the driver in the
# train window. Run under nohup; it logs to ~/open-rl/artifacts/box-logs/run51-start.log.
log() { echo "$(date -u +%FT%TZ) $*"; }
cd "$HOME/open-rl" || exit 1
set -a; source "$HOME/open-rl/.env.judge"; set +a
curl -sf -m 10 "$OPENAI_BASE_URL/models" >/dev/null || { log "judge at $OPENAI_BASE_URL is not answering; not launching"; exit 1; }
[ -f "$HOME/open-rl-k8s/examples/harvey_labs/renderers/gemma4_renderer.py" ] || { log "gemma4 renderer missing in the box recipe"; exit 1; }
if pgrep -f "[v]llm serve|[s]erver.training_requests_processor|[h]arvey-train" >/dev/null; then
  log "stack processes still alive; not launching"; pgrep -fa "[v]llm serve|[s]erver.training_requests_processor|[h]arvey-train" | cut -c1-120; exit 1
fi
tmux kill-session -t work 2>/dev/null && log "killed the old work session"
log "launching the stack"
bash scripts/launch51.sh 2>&1 | tail -5
for i in $(seq 1 240); do
  up=0
  for port in 8000 8001 8002 8003; do curl -sf -m 5 "http://127.0.0.1:$port/v1/models" >/dev/null && up=$((up + 1)); done
  ranks=$(pgrep -fc "[s]erver.training_requests_processor")
  [ "$up" -eq 4 ] && [ "$ranks" -ge 4 ] && break
  sleep 15
done
log "samplers up: $up/4, trainer ranks: $ranks"
[ "$up" -eq 4 ] || { log "samplers did not all come up; driver not started"; exit 1; }
log "waiting 120s for the trainer to finish loading"; sleep 120
tmux send-keys -t work:train C-c; sleep 1
tmux send-keys -t work:train "bash scripts/run51-driver.sh" C-m
sleep 45; tmux capture-pane -p -t work:train | grep -v "^\s*$" | tail -6 | cut -c1-160
log "run51 driver started"
