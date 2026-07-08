# Harvey LAB RL

Live-rollout RL scaffold for Harvey's Legal Agent Benchmark.

The training path reuses tinker-cookbook's GRPO loop and multi-turn tool environment. This module only adapts LAB tasks, sandbox tools, rubric reward, and Gemma 4 tool-call rendering.

## Layout

- `train.py`: small LAB-specific config mapped into `tinker_cookbook.rl.train`.
- `env.py`: LAB task loading, sandbox env construction, and dataset builders.
- `tools.py`: LAB `ToolExecutor` adapter plus `submit`.
- `reward.py`: terminal rubric reward wrapper around LAB's judge.
- `renderer.py`: Gemma 4 chat renderer with XML-style tool calls.
- `score_lab_run.py`: helper executed inside the LAB checkout/venv for rubric scoring.

## Commands

Phase 0 backend smoke, from repo root on `box` with gateway at `:9003`:

```bash
export PATH=$PATH:$HOME/.local/bin
TINKER_API_KEY=tml-dummy-key TINKER_BASE_URL=http://127.0.0.1:9003 \
uv --project examples run python examples/autoresearch/recipes/math_rl/train_gemma.py \
  model_name=gemma-4-e4b renderer_name=gemma4 env=gsm8k \
  group_size=2 groups_per_batch=1 max_steps=1 max_tokens=128 \
  base_url=http://127.0.0.1:9003 save_every=0 eval_every=0 \
  behavior_if_log_dir_exists=delete log_path=artifacts/harvey-labs/phase0-math
```

Tiny LAB run:

```bash
export PATH=$PATH:$HOME/.local/bin
TINKER_API_KEY=tml-dummy-key TINKER_BASE_URL=http://127.0.0.1:9003 \
uv --project examples run python -m recipes.harvey_labs.train \
  base-url=http://127.0.0.1:9003 \
  lab-root=experiments/lab-traces/harvey-labs \
  train-limit=1 eval-limit=0 \
  max-reward-criteria=3 log-path=artifacts/harvey-labs/lab-tiny
```
