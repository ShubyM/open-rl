# Gemma4 vLLM LoRA Fix

This repo still needs one local `vllm` site-packages patch for Gemma4 LoRA.

The patch file is:

- [gemma4_vllm_lora_fix.patch](/root/open-rl-sql-exec/server/gemma4_vllm_lora_fix.patch)

It adds `hf_to_vllm_mapper` to `Gemma4ForCausalLM` so LoRA adapter keys saved under the
conditional wrapper naming:

- `model.language_model.*`
- `...moe.experts.*`

map onto the text-only vLLM model naming:

- `model.*`
- `...moe.*`

## Apply the site-packages patch

Install the versions used for the validated local run:

```bash
cd /root/open-rl-sql-exec/server
.venv/bin/python -m pip install --upgrade --force-reinstall "vllm==0.19.0" "transformers==5.5.0"
```

Apply the patch inside the installed `site-packages` tree:

```bash
cd /root/open-rl-sql-exec/server/.venv/lib/python3.12/site-packages
patch -p1 < /root/open-rl-sql-exec/server/gemma4_vllm_lora_fix.patch
```

Optional sanity check:

```bash
/root/open-rl-sql-exec/server/.venv/bin/python - <<'PY'
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
print(Gemma4ForCausalLM.hf_to_vllm_mapper)
PY
```

## Run the stack

Start the vLLM worker:

```bash
cd /root/open-rl-sql-exec/server

CUDA_VISIBLE_DEVICES=0 \
VLLM_MODEL=google/gemma-4-e2b \
VLLM_HF_OVERRIDES='{"architectures":["Gemma4ForCausalLM"]}' \
VLLM_MAX_MODEL_LEN=1024 \
VLLM_GPU_MEMORY_UTILIZATION=0.85 \
.venv/bin/python -m src.vllm_worker
```

Start the gateway in a second shell:

```bash
cd /root/open-rl-sql-exec/server

OPEN_RL_USE_TEXT_TOWER_LORA=1 \
OPEN_RL_SINGLE_PROCESS=1 \
SAMPLER_BACKEND=vllm \
OPEN_RL_BASE_MODEL=google/gemma-4-e2b \
VLLM_MODEL=google/gemma-4-e2b \
VLLM_URL=http://127.0.0.1:8001 \
.venv/bin/uvicorn src.main:app --host 127.0.0.1 --port 9008
```

Run the client in a third shell:

```bash
cd /root/open-rl-sql-exec

client/.venv/bin/python -u client/texttosql_sft.py gemma4_e2b \
  base_url=http://127.0.0.1:9008 \
  rank=32 \
  batch_size=1 \
  learning_rate=5e-5 \
  train_limit=100 \
  eval_limit=25 \
  eval_max_tokens=64 \
  steps=100 \
  eval_every=100
```

## Expected behavior

On the validated local setup, this configuration trains instead of staying flat.

One recent run with official `transformers==5.5.0` reached:

- baseline: `execution_match=0.04`
- step 100: `execution_match=0.48`

The stronger earlier run on the prerelease/custom Transformers path reached:

- baseline: `execution_match=0.04`
- step 100: `execution_match=0.56`

The important operational requirements are:

- patch installed `vllm/model_executor/models/gemma4.py`
- force `Gemma4ForCausalLM` through `VLLM_HF_OVERRIDES`
- keep `OPEN_RL_USE_TEXT_TOWER_LORA=1`
- use the plain SQL completion recipe from [texttosql_sft.py](/root/open-rl-sql-exec/client/texttosql_sft.py)
