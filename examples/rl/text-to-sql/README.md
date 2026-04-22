# Text-to-SQL RL

This directory contains the Gemma 4 Text-to-SQL SFT+RL recipe. It keeps the
RL training loop, reward code, plotting utility, and recipe notes together
under the RL examples tree.

The recipe still reuses the dataset formatting and SQL evaluation helpers from
[`../../sft/text-to-sql/texttosql_sft.py`](../../sft/text-to-sql/texttosql_sft.py)
so the SFT-only and RL examples score SQL in the same way.

## Run

Start a Gemma 4 server with the vLLM sampler, then run:

```bash
cd examples/rl/text-to-sql
TINKER_BASE_URL=http://127.0.0.1:9003 \
TINKER_API_KEY=tml-dummy \
uv run python texttosql_sft_grpo.py gemma4_e2b_rl_recipe
```

See [`rl-recipe.md`](rl-recipe.md) for the full setup, known-good config, and
expected curves.

## Contents

* `texttosql_sft_grpo.py`: Gemma 4 SFT+RL training script.
* `texttosql_grpo_utils.py`: Dataset, batching, rollout, and PPO datum helpers.
* `texttosql_rewards.py`: SQL execution and partial-credit reward helpers.
* `plot_ttsql_curves.py`: Metrics plotter for recipe curves.
* `rl-recipe.md`: Full recipe guide and latest result summary.
* `texttosql_gemma4_plain_notebook.ipynb`: Gemma 4 plain-completion warmup notebook.
