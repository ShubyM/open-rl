# Weight transfer

The vLLM runtime uses 0.29.0 with CUDA 12.9. The `delta_snapshot` general plugin
registers the file receiver in each vLLM process. It applies sparse replacement
values with `load_checkpoint_weight_patches`, using native checkpoint names and
shapes so the model loader handles packing and tensor-parallel slicing.

The sampler drives each update through the engine API: pause generation and
wait for in-flight requests, start the update, load the files, finish with the
weights path as the version, reset the encoder cache, resume. A failed update
raises without resuming, so partially updated weights are never served.

Sparse files use format version 2. `metadata.json` contains `format: sparse_delta`,
`format_version: 2`, `delta_format: native`, and matching `layer_names` and
`layer_shapes` arrays. `delta.safetensors` contains `0.indices`, `0.values`,
`1.indices`, `1.values`, etc. Each parameter retains its own value dtype. Indices
must be strictly increasing and address the flattened full checkpoint tensor. An empty update has empty metadata
arrays and an empty safetensors file.

Upgrade trainer and sampler together. Old fused-coordinate files are rejected;
regenerate them with the upgraded trainer. Legacy `vllm_fused` configuration is
normalized to `native`. Legacy `full_replace` configuration selects full
checkpoint transfer (`strategy=full`); the CPU snapshot fallback has been removed.

Sampling drains each consecutive group of requests for one weights path before
updating to another. Updates pause generation and clear caches, then commit the
version and resume only after all workers succeed. A failed update prevents further
generation until the sampler is restarted. Time-slice boundary sleep/wake remains;
weight updates no longer add an extra sleep/restore cycle.

The upstream sparse helper stages full checkpoint-shaped tensors on the GPU.
Sparse transfer therefore reduces host-to-device bytes but does not eliminate
dense GPU staging. Only loaders supported by the upstream sparse-copy contract
are supported. Incremental deltas still require the matching prior weights;
arbitrary historical deltas are not standalone checkpoints.

Validation:

```bash
make test
make lint
UV_PROJECT_ENVIRONMENT=.venv-vllm make test-weight-transfer
UV_PROJECT_ENVIRONMENT=.venv-vllm OPEN_RL_GPU_TESTS=1 make test-weight-transfer
```

The last command builds a tiny local Qwen2 checkpoint and compares sparse updates
against a full reload, including packed projections, attention bias, tied weights,
and generated token logprobs. It needs one CUDA GPU and downloads no model. On
WSL2, also set `VLLM_WSL2_ENABLE_PIN_MEMORY=1` for Model Runner V2. CPU tests exercise
TP rank slicing through a small loader; they are not a multi-GPU benchmark.

The GPU smoke test runs in eager mode. If the local CUDA compiler predates 12.9,
set `VLLM_USE_FLASHINFER_SAMPLER=0` for this test to use the PyTorch sampler;
the server image supplies the CUDA 12.9 compiler needed by FlashInfer.
