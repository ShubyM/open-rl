# Gemma4 vLLM LoRA Fix

This branch still needs one local `vllm` site-packages patch for Gemma4 LoRA.

The patch file is:

- [gemma4_vllm_lora_fix.patch](/root/open-rl-gemma4-texttosql-sft-grpo/server/gemma4_vllm_lora_fix.patch)

It adds `hf_to_vllm_mapper` to `Gemma4ForCausalLM` so LoRA adapter keys saved under the
conditional wrapper naming:

- `model.language_model.*`
- `...moe.experts.*`

map onto the text-only vLLM model naming:

- `model.*`
- `...moe.*`

This branch also needs the worker-side Gemma4 override in
[vllm_worker.py](/root/open-rl-gemma4-texttosql-sft-grpo/server/src/vllm_worker.py),
otherwise vLLM tries to boot `Gemma4ForConditionalGeneration`, which does not support
LoRA in this path.

## Apply the site-packages patch

Install the versions used for the validated local run:

```bash
cd /root/open-rl-gemma4-texttosql-sft-grpo/server
uv sync --extra gpu --extra vllm
.venv/bin/python -m pip install --upgrade --force-reinstall "vllm==0.19.0" "transformers==5.5.0"
```

Apply the patch inside the installed `site-packages` tree:

```bash
cd /root/open-rl-gemma4-texttosql-sft-grpo/server/.venv/lib/python3.12/site-packages
patch -p1 < /root/open-rl-gemma4-texttosql-sft-grpo/server/gemma4_vllm_lora_fix.patch
```

Optional sanity check:

```bash
/root/open-rl-gemma4-texttosql-sft-grpo/server/.venv/bin/python - <<'PY'
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
print(Gemma4ForCausalLM.hf_to_vllm_mapper)
PY
```

## Run the stack

Start the vLLM worker:

```bash
cd /root/open-rl-gemma4-texttosql-sft-grpo/server

CUDA_VISIBLE_DEVICES=0 \
VLLM_MODEL=google/gemma-4-e2b \
VLLM_HF_OVERRIDES='{"architectures":["Gemma4ForCausalLM"]}' \
VLLM_MAX_MODEL_LEN=1024 \
VLLM_GPU_MEMORY_UTILIZATION=0.85 \
uv run --extra gpu --extra vllm python -m src.vllm_worker
```

Start the gateway in a second shell:

```bash
cd /root/open-rl-gemma4-texttosql-sft-grpo/server

OPEN_RL_USE_TEXT_TOWER_LORA=1 \
OPEN_RL_SINGLE_PROCESS=1 \
SAMPLER_BACKEND=vllm \
OPEN_RL_BASE_MODEL=google/gemma-4-e2b \
VLLM_MODEL=google/gemma-4-e2b \
VLLM_URL=http://127.0.0.1:8001 \
uv run --extra gpu uvicorn src.main:app --host 127.0.0.1 --port 9003
```

Run the SFT + GRPO client in a third shell:

```bash
cd /root/open-rl-gemma4-texttosql-sft-grpo/client

uv run --python 3.12 -i https://pypi.org/simple python -u texttosql_sft_grpo.py gemma4_e2b \
  base_url="http://127.0.0.1:9003"
```

Useful variants:

```bash
cd /root/open-rl-gemma4-texttosql-sft-grpo/client

uv run --python 3.12 -i https://pypi.org/simple python -u texttosql_sft_grpo.py gemma4_e2b \
  base_url="http://127.0.0.1:9003" \
  phase=sft_only
```

```bash
cd /root/open-rl-gemma4-texttosql-sft-grpo

make run-text-to-sql-gemma4-vllm
make run-text-to-sql-gemma4-server-gpu
make run-text-to-sql-sft-grpo
```

## Expected behavior

Without the worker-side override, this branch tries to boot
`Gemma4ForConditionalGeneration` and the vLLM worker fails with:

```text
ValueError: Gemma4ForConditionalGeneration does not support LoRA yet.
```

With the site-packages patch plus the worker override, the branch boots and trains.

One validated local 25-step run reached:

- baseline: `execution_match=0.04`
- step 25: `execution_match=0.12`

Artifact:

- [metrics.jsonl](/root/open-rl-sql-exec/client/artifacts/texttosql_recipe_grpo_branch_with_override_25step/metrics.jsonl)

The important operational requirements are:

- patch installed `vllm/model_executor/models/gemma4.py`
- force `Gemma4ForCausalLM` through `VLLM_HF_OVERRIDES`
- keep `OPEN_RL_USE_TEXT_TOWER_LORA=1`
- run the GRPO client from
  [texttosql_sft_grpo.py](/root/open-rl-gemma4-texttosql-sft-grpo/client/texttosql_sft_grpo.py)
