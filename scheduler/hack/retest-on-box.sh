#!/usr/bin/env bash
# Dev-loop helper: wait for the dev box to be reachable, sync the controller
# over, and run the Go suite there (the Mac kills freshly built binaries).
set -u
controller=$(cd "$(dirname "$0")/.." && pwd)

for _ in $(seq "${TRIES:-40}"); do
  if ssh -o ConnectTimeout=15 box 'echo alive' >/dev/null 2>&1; then
    rsync -a --delete --exclude bin "$controller/" box:~/sched/ || exit 1
    exec ssh box 'export PATH=$PATH:/usr/local/go/bin && cd ~/sched && go vet ./... && go test -count=1 ./... 2>&1 | tail -6'
  fi
  sleep 30
done
echo "box never came back"
exit 1
