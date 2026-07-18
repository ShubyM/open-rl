# OpenRL: self-hosted API for your RL Infrastructure

OpenRL implements [Tinker](https://tinker-docs.thinkingmachines.ai/) compatible API for fine-tuning language models that you can run on your own infrastructure (machine or a kubernetes cluster). You can use the Tinker SDK to orchestrate RL training loops by writing imperative Python code directly from your local machine.

# Why Tinker

We love Tinker. Tinker simplifies LLM post-training for developers and researchers. The Tinker API provides a smarter abstraction that decouples the underlying infrastructure from the RL training loop. This gives AI researchers complete control over their training algorithms, data loops, and loss functions and platform engineers the ability to scale the infrastructure independently.

**Bonus**: you can use [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) that has awesome tutorials/recipes and utilities!

## Quick Start

 - Follow the [Pig Latin notebook](examples/sft/pig-latin/piglatin_sft_notebook.ipynb) or [Text-to-SQL notebook](examples/sft/text-to-sql/texttosql_sft_notebook.ipynb) to see supervised fine-tuning in action.
 - Follow the [Text-to-SQL RL recipe](examples/text-to-sql/README.md) to see reinforcement learning in action.

## Inspect and Run on the Cluster

The gateway serves a dependency-free Kubernetes debugger at `/control/`. Its
purpose is cluster inspection: physical nodes, worker placement, lifecycle
phases, queues, logs, and actionable problems. Use W&B or another experiment
tracker for training metrics and charts; runs can provide a safe tracker link
through Tinker `user_metadata`, for example `{"wandb_url": run.url}`.

Agents use the same control API through a bounded diagnose → inspect → act loop:

```bash
openrl doctor --json
openrl problems --json
openrl inspect <run-id-or-node-or-pod> --json
openrl events <run-id> --limit 50 --json
openrl logs <run-id> --component sampler --json
openrl stop <run-id> --wait --timeout 2m --json
```

`problems` exits 5 when it finds warning/error issues, making it useful in scripts. `openrl --help` documents all stable exit codes. Set `OPENRL_BASE_URL` when the gateway is not at `http://127.0.0.1:9003`.

The CLI uses `chz`. Conventional kebab-case flags and native snake_case `key=value` arguments are both accepted, for example `openrl logs <run-id> --component sampler --json` or `openrl logs run_id=<run-id> component=sampler json=true`. Machine-readable control responses use snake_case; Kubernetes label and resource keys remain unchanged.

Run recipe code inside Kubernetes without a port-forward:

```bash
openrl launch examples/tiny/tiny_rl.py \
  --image registry.example/open-rl-client:revision \
  --namespace default \
  --args 'base_model=Qwen/Qwen2.5-0.5B steps=1 save_final_state=false'
```

`launch` creates an unprivileged client Job, streams the selected working-tree
source into it, and uses the gateway's Kubernetes service DNS. Rebuild the
client image only when its dependencies change; select it with `--image` or
`OPENRL_CLIENT_IMAGE` when it is not the published default. A port-forward
remains useful when running a recipe locally or opening `/control/` from your
browser, but is not part of the in-cluster job path.

Most server-only edits also avoid an image build. After the platform images and
dependencies have been deployed once, stream the current `src/` tree to the
shared PVC and roll only the gateway:

```bash
openrl deploy
# During an intentional development reset, stop all live trainer/sampler pods:
openrl deploy --reset-workers
```

The content-addressed source revision is injected into new workers. Resetting
workers ends their active runs, which must be relaunched. Use
`make push-to-cluster` when dependencies, CUDA patches, the time-slicer image,
or pod-level system files change. That slow path is only Docker plus `kubectl`;
it applies the existing `distributed-fft-timeslice` Kustomize deployment by
default. Select another checked-in overlay with `K8S_DIR=<path>`.

For a single GCP GPU VM, keep the existing source-sync workflow and run one
setup script on the VM:

```bash
make push-vm REMOTE_HOST=b7
ssh b7 'cd ~/open-rl && CUDA_VISIBLE_DEVICES=0 ./dev/infra/setup_vm.sh'
```

It installs small missing system prerequisites, selects CUDA checkpointing when
available (otherwise the no-op time-slicer), and creates persistent `redis`,
`timeslicer`, `gateway`, and `gpu` tmux windows. Configure separate pools with
`CUDA_VISIBLE_DEVICES` and `SAMPLER_CUDA_VISIBLE_DEVICES`; reconnect with
`tmux attach -t openrl`. Gateway, worker, and time-slicer output is available to
`openrl logs` and rotates at 25 MiB per file by default; adjust
`OPEN_RL_LOG_MAX_BYTES` if needed.

The executable runtime contract is intentionally small:

- client sessions are unique, expire unless heartbeated, and carry only typed metadata;
- retried SDK operations with the same sequence id enqueue once and return the same request id;
- a model revision advances only after a successful optimizer step;
- sampler sessions resolve to one immutable model revision and never silently retarget;
- repeated saves of an unchanged revision reuse its artifact, ephemeral sessions use count retention, and named sessions use TTL retention;
- an in-flight sampler session prevents its artifact from being pruned; and
- optimizer steps do not write checkpoints—only explicit save operations do.

These behaviors are covered without a GPU in
`tests/test_protocol_lifecycle.py`, `tests/test_worker_manager.py`, and
`tests/test_trainer_optimizer_correctness.py`. The real LoRA path also has a
one-process smoke test in `tests/test_in_process_training.py`: it creates a tiny
model locally and exercises the same typed commands as the gateway without
starting Redis, HTTP, a sampler, or a time-slicer.

## Development and tests

Python development uses uv and pytest directly; Make is reserved for image,
Kubernetes, and VM shell operations.

```bash
# Choose the machine environment once (CPU laptop or GPU VM).
uv sync --extra cpu
# uv sync --extra gpu

# Fast behavior suite. No dependency solving or syncing on each invocation.
uv run --no-sync pytest

# Lint and format.
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .

# Install recipe dependencies once, then test the real Tinker/HTTP boundary.
uv --project examples sync --group test
PYTHONPATH=examples/sft/pig-latin uv --project examples run --no-sync pytest \
  -m tinker_contract tests/test_piglatin_qwen.py -v

# Run one tiny recipe against infrastructure that is already deployed.
OPENRL_BASE_URL=https://gateway.example \
  uv run --no-sync pytest -m distributed tests/test_distributed_contract.py -s
```

`tinker_contract` and `distributed` are excluded from the default suite. The
first owns SDK compatibility; the second owns deployment wiring. Multi-job,
large-model, and time-slicing experiments are benchmarks launched with
`openrl launch`, not alternate branches inside a test harness.

Snippet below shows a sample Reinforcement Learning loop like GRPO, where the 4 API primitives are used to create a generate-and-reward-train loop:

```python
import asyncio
import tinker
from tinker import types

# Placeholder Environment & Reward Functions
def generate_math_problem() -> str: ...
def compute_advantages(rewards: list[float]) -> list[float]: ...
def parse_and_score_response(text: str) -> float: ...

async def rlvr_loop():
    service_client = tinker.ServiceClient(base_url="http://localhost:8000")

    # 1. Create Model
    training_client = await service_client.create_lora_training_client_async(
        base_model="Qwen/Qwen3-4B-Instruct-2507", rank=16
    )

    for epoch in range(10):
        # 2A. Extract sampling client from current weights
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"rlvr_epoch_{epoch}"
        )
        
        prompt_text = generate_math_problem()
        
        # 2B. Sample multiple rollouts (e.g. N=8) from the prompt
        response = sampling_client.sample(
            prompt=types.ModelInput.from_ints(tokens=[...]),
            num_samples=8,
            sampling_params=types.SamplingParams(max_tokens=100, temperature=0.9)
        ).result()
        
        # 3. Score the rollouts using the environment
        rewards = []
        for seq in response.sequences:
            text = decode(seq.tokens)
            rewards.append(parse_and_score_response(text))
            
        advantages = compute_advantages(rewards)
        
        # ... package sequences, text, and advantages into datums ...

        # 4. Forward-Backward Pass (Importance Sampling)
        # We pass the advantages to RL objective function
        await training_client.forward_backward_async(
            datums, 
            loss_fn="importance_sampling",
            loss_fn_config={"clip_range": 0.2} 
        )
        
        # 5. Optimizer Step
        await training_client.optim_step_async(types.AdamParams(learning_rate=1e-5))

asyncio.run(rlvr_loop())
```

## Documentation & Guides

Detailed guides and runnable examples are structured under `docs/` and `examples/`:

- **Guides:**
  - Supervised finetuning:
    - [Pig Latin SFT Notebook](examples/sft/pig-latin/piglatin_sft_notebook.ipynb) & [script guide](examples/sft/pig-latin/README.md)
    - [Text-to-SQL SFT Notebook](examples/sft/text-to-sql/texttosql_sft_notebook.ipynb)
  - Reinforcement Learning:
    - [Text-to-SQL RL Recipe](examples/text-to-sql/README.md)
- **Technical Documentation**:
  - [Architecture](docs/architecture.md)
  - [Tinker Client Compatibility](docs/tinker-client-compatibility.md)
- **Deployment**:
  - [Kubernetes Deployment Guide (GKE)](docs/setup/gke-setup.md)

## Roadmap

- [ ] Blog posts + Demo videos
- [ ] Full parameter finetuning
- [ ] Multi model support
- [ ] Model checkpoints API
- [ ] Autoresearch integration

## Contributing

This project is licensed under the [Apache 2.0 License](LICENSE).

We welcome contributions! Please see [docs/contributing.md](docs/contributing.md) for more information.

We follow [Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).

## Disclaimer

This is not an officially supported Google product.

This project is not eligible for the Google Open Source Software Vulnerability Rewards Program.
