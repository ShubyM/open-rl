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
  model_name=google/gemma-4-E4B-it renderer_name=gemma4 env=gsm8k \
  group_size=2 groups_per_batch=1 max_steps=1 max_tokens=128 \
  base_url=http://127.0.0.1:9003 save_every=0 eval_every=0 \
  behavior_if_log_dir_exists=delete log_path=artifacts/harvey-labs/phase0-math
```

Tiny LAB run:

```bash
export PATH=$PATH:$HOME/.local/bin
TINKER_API_KEY=tml-dummy-key TINKER_BASE_URL=http://127.0.0.1:9003 \
uv --project examples run python examples/harvey_labs/train.py \
  base_url=http://127.0.0.1:9003 \
  lab_root=experiments/lab-traces/harvey-labs \
  train_limit=1 eval_limit=0 \
  max_reward_criteria=3 log_path=artifacts/harvey-labs/lab-tiny
```

The training defaults run 40 task groups with four rollouts per group. Each
rollout is trained as a separate microbatch and is capped at 32K trajectory
tokens, with at most 1K generated tokens per tool turn. The 32K cap completed a
full Gemma 4 E4B FFT update on one 80GB H100; a 64K trajectory did not.

LAB uses the instruction-tuned `google/gemma-4-E4B-it` checkpoint and its native
function-calling template. The environment reuses LAB's system prompt, default
`docx`/`pptx`/`xlsx` skill manuals and scripts, six sandbox tools, and rubric
judge; `submit` is the only additional tool.
