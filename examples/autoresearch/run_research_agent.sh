#!/usr/bin/env bash
set -euo pipefail

RESEARCHER_ID="${RESEARCHER_ID:?Set RESEARCHER_ID}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
LOG_ROOT="${LOG_ROOT:-artifacts/autoresearch/runs}"
WORK_DIR="${LOG_ROOT}/${RESEARCHER_ID}-activity"
RESEARCHER_LOG_PATH="${WORK_DIR}/agent.log"
LAUNCHER_LOG_PATH="${WORK_DIR}/launcher.log"
RECIPE="${RECIPE:-autoresearch.toml}"
ATTEMPT_TIMEOUT_MINUTES="${ATTEMPT_TIMEOUT_MINUTES:-5}"
AGENT_TIMEOUT_MINUTES="${AGENT_TIMEOUT_MINUTES:-10}"
AGENT_MODEL="${AGENT_MODEL:-gemini-3.1-pro-preview}"
AGENT_FLAGS="${AGENT_FLAGS:---yolo --output-format stream-json}"
READY_URLS="${READY_URLS:-}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-1200}"
RUN_ATTEMPT_COMMAND="uv run --no-sync --package open-rl-client python -m run_attempt recipe=\"${RECIPE}\" researcher=\"${RESEARCHER_ID}\" attempt_timeout_minutes=\"${ATTEMPT_TIMEOUT_MINUTES}\" name=short-slug log_root=\"${LOG_ROOT}\""
DEFAULT_CONFIG_COMMAND="uv run --no-sync --package open-rl-client python -m run_attempt recipe=\"${RECIPE}\" researcher=\"${RESEARCHER_ID}\" attempt_timeout_minutes=\"${ATTEMPT_TIMEOUT_MINUTES}\" name=default-config log_root=\"${LOG_ROOT}\""
SESSION_STATUS="running"

export RESEARCHER_ID ATTEMPT_TIMEOUT_MINUTES AGENT_TIMEOUT_MINUTES LOG_ROOT WORK_DIR REPO_DIR
export RESEARCHER_LOG_PATH RECIPE RUN_ATTEMPT_COMMAND READY_URLS READY_TIMEOUT_SECONDS
export DEFAULT_CONFIG_COMMAND
export GEMINI_CLI_TRUST_WORKSPACE=true PYTHONDONTWRITEBYTECODE=1

die() { echo "$*" >&2; exit 2; }
log() { echo "$*" | tee -a "${LAUNCHER_LOG_PATH}" "${RESEARCHER_LOG_PATH}"; }

emit_activity() {
  local log_path notes_path
  mkdir -p "$(dirname "${RESEARCHER_LOG_PATH}")" "$(dirname "${LAUNCHER_LOG_PATH}")" "${WORK_DIR}"
  touch "${RESEARCHER_LOG_PATH}" "${LAUNCHER_LOG_PATH}" "${WORK_DIR}/notes.md"
  log_path="$(readlink -f "${RESEARCHER_LOG_PATH}")"
  notes_path="$(readlink -f "${WORK_DIR}/notes.md")"
  PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" python3 -m event_log \
    --event-dir "${WORK_DIR}" \
    --researcher "${RESEARCHER_ID}" \
    --status "$1" \
    --attempt-timeout-minutes "${AGENT_TIMEOUT_MINUTES}" \
    --agent-log "${log_path}" \
    --notes "${notes_path}"
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
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "REPO_DIR=${REPO_DIR} is not a git repo."

  log "Using existing git repo at $(git rev-parse --show-toplevel)."
  git config user.email "agent@open-rl.local"
  git config user.name "Autoresearch Agent"
  exclude_artifacts_from_git
  if [ -n "$(git status --porcelain)" ]; then
    die "Existing git repo has local changes. Run from an isolated clean copy."
  fi
}

wait_for_url() {
  local url deadline last_log code
  url="$1"
  deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  last_log=0
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if curl -fsS --max-time 5 "${url}" >/dev/null; then
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
  log "Waiting for dependencies before default-config attempt: ${READY_URLS}"
  for url in ${READY_URLS//,/ }; do
    [ -n "${url}" ] && wait_for_url "${url}"
  done
}

cd "${REPO_DIR}"
REPO_DIR="$(pwd)"
export REPO_DIR
program_file="$(dirname "${RECIPE}")/program.md"
[ -f "${program_file}" ] || die "program file does not exist next to RECIPE: ${program_file}"
mkdir -p "${WORK_DIR}" "${LOG_ROOT}" "$(dirname "${RESEARCHER_LOG_PATH}")" "$(dirname "${LAUNCHER_LOG_PATH}")"
cp "${program_file}" "${WORK_DIR}/program.md"
touch "${RESEARCHER_LOG_PATH}" "${LAUNCHER_LOG_PATH}"
emit_activity running

prepare_git
wait_for_dependencies
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
