.PHONY: run-server run-client-sft run-client-countdown run-client-simple run-client-binary run-client-guess run-client-joke

# Run the Uvicorn server locally, forcing the public PyPI index for uv
run-server:
	cd server && UV_INDEX_URL="https://pypi.org/simple" uv run uvicorn src.main:app --host 127.0.0.1 --port 8000

# Client test targets
run-client-sft:
	cd client && uv run python test_basic_workflow.py

run-client-countdown:
	cd client && uv run python test_countdown_rl.py

run-client-simple:
	cd client && uv run python test_simple_rl.py

run-client-binary:
	cd client && uv run python test_binary_rl.py

run-client-guess:
	cd client && uv run python test_guess_rl.py

run-client-joke:
	cd client && uv run python test_joke_rl.py

run-client-sft-notebook:
	cd client && uv run python test_sft_notebook.py

run-client-rlvr-notebook:
	cd client && uv run python test_rlvr_notebook.py
