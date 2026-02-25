.PHONY: run-server run-sft run-sft-parallel run-rlvr run-rlvr-parallel

# Default VLLM model for local dev, can be overridden via `make run-server VLLM_MODEL=...`
#VLLM_MODEL ?= Qwen/Qwen2.5-0.5B
VLLM_MODEL ?= Qwen/Qwen3-4B-Instruct-2507

# Run the Uvicorn server locally, forcing the public PyPI index for uv
run-server:
	cd server && UV_INDEX_URL="https://pypi.org/simple" VLLM_MODEL="$(VLLM_MODEL)" uv run uvicorn src.main:app --host 127.0.0.1 --port 8000

# Client test targets
run-sft:
	cd client && uv run --no-sync -i https://pypi.org/simple python sft.py --base-model "$(VLLM_MODEL)" $(ARGS)

run-sft-parallel:
	cd client && uv run --no-sync -i https://pypi.org/simple python sft.py --parallel --base-model "$(VLLM_MODEL)"

run-rlvr:
	cd client && uv run --no-sync -i https://pypi.org/simple python rlvr.py --steps 15 --base-model "$(VLLM_MODEL)"

run-rlvr-parallel:
	cd client && uv run --no-sync -i https://pypi.org/simple python rlvr.py parallel --steps 15 --base-model "$(VLLM_MODEL)"

# Plot metrics from a JSONL file
# Usage: make plot-metrics [FILE=path/to/metrics.jsonl]
plot-metrics:
	cd client && uv run --no-sync -i https://pypi.org/simple python plot_metrics.py $(FILE)

# Plot parallel metrics from the RLVR log file
# Usage: make plot-logs [LOG_FILE=client/rlvr_parallel_results.log] [WATCH=1]
plot-logs:
	cd client && uv run --no-sync -i https://pypi.org/simple python plot_logs.py $(or $(LOG_FILE),rlvr_parallel_results.log) $(if $(WATCH),--watch,)

# Generate diagrams using local mmdc zsh alias
diagrams:
	zsh -ic "mmdc -i design_arch.mmd -o design_arch.svg"
	zsh -ic "mmdc -i rollout_flow.mmd -o rollout_flow.svg"

# Sync server to remote host b3
# TODO: sync only server directory
server-sync:
	rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' ./ b3:~/work/open-rl

server-tunnel:
	ssh -fN -L 8000:localhost:8000 b3

# CLI Targets
# Usage: make run-cli list OR make run-cli chat --model ...
# This hack allows passing arguments directly after the target name
ifeq (run-cli,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run-cli"
  CLI_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets so make doesn't complain
  $(eval $(CLI_ARGS):;@:)
endif

run-cli:
	@cd client && uv run --no-sync -i https://pypi.org/simple python cli.py $(CLI_ARGS)

# Shortcut: make run-cli-list
run-cli-list:
	@cd client && uv run --no-sync -i https://pypi.org/simple python cli.py list

# Shortcut: make run-cli-chat MODEL=... [PROMPT="..."]
run-cli-chat:
	@test -n "$(MODEL)" || (echo "Error: MODEL argument is required. Usage: make run-cli-chat MODEL=<model_id>" && exit 1)
	@cd client && uv run --no-sync -i https://pypi.org/simple python cli.py chat --model $(MODEL) --system-prompt "$(or $(PROMPT),You are helpful geography assistant.)"