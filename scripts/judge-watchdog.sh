#!/usr/bin/env bash
# Stop the training driver when the judge stops answering, instead of letting
# the run grade every episode as a reward error and train on the -0.1/0 floor
# (run51 lost nine steps that way when the spot judge VM was preempted).
# Usage: judge-watchdog.sh <tmux window with the driver>   e.g. work:train
set -u
WINDOW=${1:-work:train}
cd "$HOME/open-rl" || exit 1
set -a; source "$HOME/open-rl/.env.judge"; set +a
failures=0
while true; do
  if curl -sf -m 20 "$OPENAI_BASE_URL/models" >/dev/null; then
    failures=0
  else
    failures=$((failures + 1))
    echo "$(date -u +%FT%TZ) judge at $OPENAI_BASE_URL not answering ($failures)"
  fi
  if [ "$failures" -ge 5 ] && pgrep -f "[h]arvey-train" >/dev/null; then
    echo "$(date -u +%FT%TZ) judge down for 5 checks; stopping the driver in $WINDOW"
    tmux send-keys -t "$WINDOW" C-c
    sleep 30
    pkill -INT -f "[h]arvey-train"
  fi
  sleep 60
done
