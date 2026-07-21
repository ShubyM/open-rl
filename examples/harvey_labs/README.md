# Harvey LAB RL

Live-rollout RL on Harvey's Legal Agent Benchmark: the model works real LAB
tasks in a podman sandbox (documents, shell, file tools), a judge grades the
rubric, and the pass fraction is the reward. Training reuses tinker-cookbook's
GRPO loop and multi-turn tool environment; this module adapts LAB tasks,
sandbox tools, and rubric reward.

## Layout

- `train.py` — run config and entrypoint (with grading preflight and final eval).
- `tasks.py` — task discovery, loading, and the seeded random train/eval split.
- `eval_checkpoint.py` — evaluate any saved adapter checkpoint on the eval split.
- `prompts.py` — system prompt, skills, and the output-path contract.
- `env.py` — sandbox env construction and dataset builders.
- `reward.py` — rubric reward wrapper around LAB's judge.
- `tools.py` — LAB `ToolExecutor` adapter.
- `gemma4_renderer.py` — Gemma 4 tool-call renderer (Qwen uses the stock `qwen3_5` renderer).
- `plot_run.py` — plot a run's rewards and pass rate.
- `score_lab_run.py` — grading shim executed inside the LAB checkout's venv.

## Setup

One command on a bare Ubuntu GPU VM:

```bash
git clone https://github.com/ShubyM/open-rl && cd open-rl
./scripts/setup_vm.sh
```

It installs apt build deps (build-essential, python3-dev, ninja-build, tmux),
uv, finds and persists the CUDA toolchain, syncs the Python env, bootstraps
the LAB harness (sandbox image, pandoc, podman), checks disk space for
container storage, and ends with a green/red checklist. Idempotent — re-run
it any time as a health check. The `causal-conv1d` fast path lives in its own
`fastpath` extra so a compiler problem degrades to a loud 2–5x-slower warning
instead of killing the install.

The manual steps it replaces, for reference:

1. **Repo + Python env** (`nvcc` needed only for the `fastpath` extra):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync --frozen --exact --extra gpu --extra vllm --extra fastpath
   ```

   Sanity check the fast path — this must print `True` or Qwen training runs
   2–5x slower on an eager fallback:

   ```bash
   uv run --no-sync python -c "from transformers.models.qwen3_5 import modeling_qwen3_5 as m; print(m.is_fast_path_available)"
   ```

2. **LAB checkout + sandbox** — one command, from anywhere:

   ```bash
   examples/harvey_labs/setup_lab.sh
   ```

   Clones the LAB fork next to the recipe (the default `lab_root`) and runs
   LAB's own setup: harness deps, pandoc, podman, and the sandbox image from
   ghcr. The fork (`ShubyM/harvey-labs`) includes harness fixes (upstream
   PRs #85–#90) that change tool behavior and rubric scoring; rewards from
   unfixed upstream are not comparable.

3. **Judge keys** — rubric scoring calls the judge model (`judge_model`,
   default Gemini): `export GEMINI_API_KEY=...`. Also
   `export ANTHROPIC_API_KEY=...` — when the model names an output file
   differently than the rubric expects, LAB's deliverable matcher resolves it
   with an Anthropic call; without the key those episodes fail grading.

   `train.py` and `eval_checkpoint.py` preflight the grading environment at
   startup (LAB venv present, judge importable and carrying the truncation
   fixes) and refuse to start otherwise — a broken grader silently zeroes
   rewards, which is far more expensive than a loud early failure.

4. **Gateway** — an open-rl server for the policy model. Simplest vLLM shape
   is your own stock server plus the gateway pointed at it:

   ```bash
   VLLM_ALLOW_RUNTIME_LORA_UPDATING=true uv run --extra gpu --extra vllm \
     vllm serve <model> --port 8000 --enable-lora --max-lora-rank 64 \
     --max-model-len 65536 --language-model-only --disable-log-requests

   SAMPLER_BASE_URL=http://127.0.0.1:8000 BASE_MODEL=<model> \
     uv run --extra gpu --extra vllm python -m uvicorn server.gateway:app --port 9003
   ```

   All server shapes (single-process torch sampler, managed queue workers,
   FFT) are in [docs/quickstart.md](../../docs/quickstart.md). Keep the
   sampler's `--max-model-len` equal to the recipe's
   `max_trajectory_tokens` — a mismatch turns over-length rollouts into
   silent parse failures.

## Run

One command on an 8-GPU box — brings up sampler (GPUs 1-7), gateway +
trainer (GPU 0), and a typed train command in a tmux session named `work`:

```bash
MODEL=9b ./scripts/launch_work.sh    # Qwen3.5-9B at its full 262K window
MODEL=27b ./scripts/launch_work.sh   # Qwen3.5-27B at 98K (measured H200 ceiling)
```

Or by hand:

```bash
TINKER_API_KEY=tml-dummy-key \
uv --project examples run python examples/harvey_labs/train.py \
  model_name=Qwen/Qwen3.5-27B \
  base_url=http://127.0.0.1:9003 \
  learning_rate=2e-4 lora_rank=32 \
  batch_size=5 rollouts_per_example=2 max_steps=20 eval_every=5 \
  max_tokens=16384 max_trajectory_tokens=98304 max_tool_result_tokens=16384 \
  log_path=artifacts/harvey-labs/my-run
```

**Task selection** is a seeded random split of the whole runnable LAB pool
(~1,750 tasks): `train_tasks=300 eval_tasks=50 task_split_seed=0` by
default, disjoint, preflighted for instructions/criteria/documents. The
same config always reproduces the same split, so the split is the
benchmark — keep the seed fixed across runs you want to compare.
`task=<name>` trains a single task (no eval set) for smoke tests:
`task=immigration/identify-h1b-qualification-issues max_reward_criteria=3`.

The renderer is derived from the model name (`qwen3_5` / `gemma4`);
`renderer_name=` overrides. For LoRA, the cookbook recommends learning
rates near `hyperparam_utils.get_lr(model_name)` (~4.6e-4 for a 27B) — the
3e-6 default in `RunConfig` is an FFT-scale value.

In-loop evals run *before* the optimizer step of their batch (batch 0 is
the untrained baseline); `final_eval=true` (default) additionally
evaluates the last checkpoint after training. To evaluate any saved
checkpoint later:

```bash
uv --project examples run python examples/harvey_labs/eval_checkpoint.py \
  checkpoint=/tmp/open-rl/peft/<model-id>/final \
  model_name=Qwen/Qwen3.5-27B base_url=http://127.0.0.1:9003 \
  max_tokens=16384 max_trajectory_tokens=98304 max_tool_result_tokens=16384
```

Pass the same split/window knobs as the training run, or the eval measures
a different benchmark than the run's own evals.

## Watching a run

- **Console** prints one line per rollout and the standard per-step metric
  table (`log_full_rollouts=true` restores full response dumps).
- **`plot_run.py`** — rewards and held-out pass rate from a run directory:
  `uv --project examples run python examples/harvey_labs/plot_run.py artifacts/harvey-labs/<run>`
- **`metrics.jsonl`** in the run dir has every metric per step. Every episode
  reports `lab/criteria_passed` / `lab/criteria_total` (failed episodes count
  0/N), so `mean(lab/criteria_passed) x total_episodes` gives exact pooled
  criterion counts.
- **Per-rollout detail**: `iterations/iteration_*/`
  holds full transcripts (`train.html`) and machine-readable
  `*_rollout_summaries.jsonl`; keep these when archiving a run.
- **Rubric verdicts** per episode: `<lab_root>/results/<run-id>/scores.json`
  (and `reward_error.log` when the judge failed).
- Signals worth watching: `by_group/frac_all_bad` (structural failures — a
  rollout whose last action is exactly `max_tokens` long hit the generation
  cap and parse-failed), `optim/kl_sample_train_v1` (should stay ~1e-4–1e-3
  on-policy), and `lab/reward_error` (judge crashes).

## Podman notes

LAB's sandbox runs rootless podman; `setup_lab.sh` installs it and pulls
`ghcr.io/harveyai/lab-sandbox`. When it misbehaves:

- **Verify the sandbox itself**: `podman run --rm ghcr.io/harveyai/lab-sandbox:latest echo ok`.
- **Rootless under nohup/ssh**: podman needs `XDG_RUNTIME_DIR`
  (`/run/user/$(id -u)`). Detached shells sometimes lack it — export it in
  launch scripts, and `loginctl enable-linger $USER` if containers must
  outlive your ssh session.
- **Leaked containers** after killed runs eat disk and ports:
  `podman ps -a` to inspect, `podman rm -f $(podman ps -aq)` to sweep.
  Normal teardown is handled by the env's `cleanup()`.
- **Small root disks**: images and container layers live under
  `~/.local/share/containers` (~2GB+). Move with `graphroot` in
  `~/.config/containers/storage.conf` before pulling.
- **First-run uid mapping errors** (`newuidmap: ... not allowed`): the user
  needs `/etc/subuid` + `/etc/subgid` entries (`sudo usermod
  --add-subuids 100000-165535 --add-subgids 100000-165535 $USER`), then
  `podman system migrate`.

## Troubleshooting

- **Rollouts are streams of `<pad>` tokens** (`stop_reason=length`, logprobs
  exactly `-0.1`): the sampler is in mock mode — vllm failed to import in
  that process or `MOCK_VLLM=1`. Check its log for `bypassing real engine init`.
- **Empty completions / `leaves no room in max_model_len` log lines**: the
  sampler's context is smaller than `max_trajectory_tokens`; fix the mismatch
  on whichever side is wrong.
- **Sampler rejects the adapter over `lm_head`**: the tinker SDK defaults
  `train_unembed=True`; the trainer skips `lm_head` (loud log line) so
  adapters stay vLLM-loadable. `OPEN_RL_LORA_TRAIN_UNEMBED=1` opts in
  (torch sampler only).
- **Engine-core timeout while compiling Qwen kernels on first boot**: raise
  `VLLM_ENGINE_READY_TIMEOUT_S` (default 600s). Kernel builds cache in
  `~/.triton` / `~/.tilelang` / `~/.cache/flashinfer` — persist them across
  restarts; a kernel stuck 15+ minutes means stale locks (wipe `~/.triton`).
- **CUDA OOM in training**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  on the gateway; pack short rollouts with `OPEN_RL_TRAIN_TOKEN_BUDGET`
  (set on the gateway — the trainer lives in that process); add
  `OPEN_RL_ACTIVATION_CPU_OFFLOAD=1` when the budget approaches the card.
  Full knob reference: [docs/configuration.md](../../docs/configuration.md).
- **Rubric scores all zero**: judge API key, or the LAB venv is stale
  (`uv sync` inside the LAB checkout — the judge runs there via
  `score_lab_run.py`).
- **Episodes end after one turn with no tool call**: renderer/template
  mismatch — check the renderer matches the model family.

Full fine-tuning (server-side `OPEN_RL_ENABLE_FFT=true`, checkpoint
hot-reload sampling, memory ceilings and offload knobs) is covered in
[docs/quickstart.md](../../docs/quickstart.md) and
[docs/fft/single-h100-long-context.md](../../docs/fft/single-h100-long-context.md).

Run the repository unit tests with `make test unit`.
