#!/usr/bin/env bash
set -euo pipefail

RESEARCHER_ID="${RESEARCHER_ID:-researcher-$(date +%Y%m%d-%H%M%S)-$$}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
LOG_ROOT="${LOG_ROOT:-artifacts/autoresearch/runs}"
WORK_DIR="${LOG_ROOT}/${RESEARCHER_ID}-activity"
RESEARCHER_LOG_PATH="${WORK_DIR}/agent.log"
LAUNCHER_LOG_PATH="${WORK_DIR}/launcher.log"
RECIPE="${RECIPE:?Set RECIPE=path/to/autoresearch.toml}"
PROGRAM_FILE="${PROGRAM_FILE:?Set PROGRAM_FILE=path/to/program.md}"
ATTEMPT_TIMEOUT_MINUTES="${ATTEMPT_TIMEOUT_MINUTES:-5}"
AGENT_TIMEOUT_MINUTES="${AGENT_TIMEOUT_MINUTES:-10}"
AGENT_MODEL="${AGENT_MODEL:-gemini-3.1-pro-preview}"
AGENT_FLAGS="${AGENT_FLAGS:---skip-trust --approval-mode yolo --output-format stream-json}"
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
READY_URLS="${READY_URLS:-}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-1200}"
RUN_ATTEMPT_COMMAND="uv run --no-sync --package open-rl-client python -m rl.autoresearch.run_attempt recipe=\"${RECIPE}\" researcher=\"${RESEARCHER_ID}\" attempt_timeout_minutes=\"${ATTEMPT_TIMEOUT_MINUTES}\" name=short-slug log_root=\"${LOG_ROOT}\""
SESSION_STATUS="running"

export RESEARCHER_ID ATTEMPT_TIMEOUT_MINUTES AGENT_TIMEOUT_MINUTES LOG_ROOT WORK_DIR REPO_DIR
export RESEARCHER_LOG_PATH RECIPE PROGRAM_FILE RUN_ATTEMPT_COMMAND READY_URLS READY_TIMEOUT_SECONDS
export GEMINI_CLI_TRUST_WORKSPACE=true PYTHONDONTWRITEBYTECODE=1

die() { echo "$*" >&2; exit 2; }
log() { echo "$*" | tee -a "${LAUNCHER_LOG_PATH}" "${RESEARCHER_LOG_PATH}"; }

emit_activity() {
  local event_file log_path
  event_file="${WORK_DIR}/ui_events.jsonl"
  mkdir -p "$(dirname "${RESEARCHER_LOG_PATH}")" "$(dirname "${LAUNCHER_LOG_PATH}")" "$(dirname "${event_file}")"
  touch "${RESEARCHER_LOG_PATH}" "${LAUNCHER_LOG_PATH}"
  log_path="$(readlink -f "${RESEARCHER_LOG_PATH}")"
  printf '{"kind":"activity","order":0,"path":"%s","researcher":"%s","status":"%s","tab":"agent","time":%s,"work_id":"%s-activity"}\n' \
    "${log_path}" "${RESEARCHER_ID}" "$1" "$(date +%s)" "${RESEARCHER_ID}" >>"${event_file}"
}

finish() {
  local code=$?
  if [ "${SESSION_STATUS}" = "running" ]; then
    [ "${code}" -eq 0 ] && SESSION_STATUS="completed" || SESSION_STATUS="failed"
  fi
  emit_activity "${SESSION_STATUS}" || true
  log "Autoresearch session exiting with code ${code}." >/dev/null || true
}

trap finish EXIT
trap 'SESSION_STATUS=stopped; log "Autoresearch session received shutdown signal." >/dev/null || true; exit 143' INT TERM

exclude_artifacts_from_git() {
  local artifact_path exclude_file git_path repo_path
  repo_path="$(readlink -f "$(git rev-parse --show-toplevel)")"
  exclude_file="$(git rev-parse --git-path info/exclude)"
  mkdir -p "$(dirname "${exclude_file}")"
  for git_path in "__pycache__/" "*.pyc"; do
    grep -Fxq "${git_path}" "${exclude_file}" 2>/dev/null || printf "%s\n" "${git_path}" >>"${exclude_file}"
  done
  artifact_path="$(readlink -f "${LOG_ROOT}")"
  case "${artifact_path}" in
    "${repo_path}"/*)
      git_path="/${artifact_path#"${repo_path}/"}/"
      grep -Fxq "${git_path}" "${exclude_file}" 2>/dev/null || printf "%s\n" "${git_path}" >>"${exclude_file}"
      ;;
  esac
}

prepare_git() {
  git config --global --add safe.directory "${REPO_DIR}" || true
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    [ -f /app/.gitignore ] && cp /app/.gitignore ./ || touch .gitignore
    git init
    git config user.email "agent@open-rl.local"
    git config user.name "Autoresearch Agent"
    exclude_artifacts_from_git
    git add .
    git commit -m "initial baseline commit"
    return
  fi

  log "Using existing git repo at $(git rev-parse --show-toplevel)."
  git config user.email "agent@open-rl.local"
  git config user.name "Autoresearch Agent"
  exclude_artifacts_from_git
  if [ "${ALLOW_DIRTY_REPO}" != "1" ] && [ -n "$(git status --porcelain)" ]; then
    die "Existing git repo has local changes. Run from an isolated copy or set ALLOW_DIRTY_REPO=1."
  fi
  return 0
}

run_baseline() {
  local code
  log "Running baseline attempt."
  set +e
  uv run --no-sync --package open-rl-client python -m rl.autoresearch.run_attempt \
    recipe="${RECIPE}" \
    researcher="${RESEARCHER_ID}" \
    attempt_timeout_minutes="${ATTEMPT_TIMEOUT_MINUTES}" \
    name=baseline \
    log_root="${LOG_ROOT}"
  code=$?
  set -e
  [ "${code}" -eq 0 ] || die "Baseline attempt failed with code ${code}."
}

python_cmd() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    die "python not found; cannot run dependency readiness checks."
  fi
}

wait_for_url() {
  local url deadline last_log code python
  url="$1"
  python="$(python_cmd)"
  deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  last_log=0
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    set +e
    "${python}" - "${url}" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import urlopen

with urlopen(sys.argv[1], timeout=5) as response:
  raise SystemExit(0 if 200 <= response.status < 400 else 1)
PY
    code=$?
    set -e
    if [ "${code}" -eq 0 ]; then
      log "Dependency ready: ${url}"
      return 0
    fi
    if [ $((SECONDS - last_log)) -ge 30 ]; then
      log "Waiting for dependency: ${url}"
      last_log="${SECONDS}"
    fi
    sleep 5
  done
  die "Timed out waiting for dependency after ${READY_TIMEOUT_SECONDS}s: ${url}"
}

wait_for_dependencies() {
  local url
  [ -n "${READY_URLS}" ] || return 0
  log "Waiting for dependencies before baseline: ${READY_URLS}"
  for url in ${READY_URLS//,/ }; do
    [ -n "${url}" ] && wait_for_url "${url}"
  done
}

cd "${REPO_DIR}"
REPO_DIR="$(pwd)"
[ -f "${PROGRAM_FILE}" ] || die "PROGRAM_FILE=${PROGRAM_FILE} does not exist"
mkdir -p "${WORK_DIR}" "${LOG_ROOT}" "$(dirname "${RESEARCHER_LOG_PATH}")" "$(dirname "${LAUNCHER_LOG_PATH}")"
cp "${PROGRAM_FILE}" "${WORK_DIR}/program.md"
touch "${RESEARCHER_LOG_PATH}" "${LAUNCHER_LOG_PATH}"
emit_activity running

prepare_git
wait_for_dependencies
run_baseline
command -v gemini >/dev/null 2>&1 || die "gemini not found; install @google/gemini-cli in the image or local environment."
emit_activity running
log "Starting ${AGENT_TIMEOUT_MINUTES} minute agent timeout."

set +e
timeout "${AGENT_TIMEOUT_MINUTES}m" gemini --model "${AGENT_MODEL}" ${AGENT_FLAGS} --prompt "$(cat "${WORK_DIR}/program.md")" 2>&1 | tee -a "${RESEARCHER_LOG_PATH}"
code=${PIPESTATUS[0]}
set -e

if [ "${code}" -eq 124 ] || [ "${code}" -eq 137 ]; then
  SESSION_STATUS="timed_out"
  log "Agent timed out after ${AGENT_TIMEOUT_MINUTES} minutes."
  exit 0
fi
exit "${code}"
