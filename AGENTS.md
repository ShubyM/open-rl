# Open-RL Agent Instructions

Welcome, Agent! This guide outlines the project structure, environments, and execution workflows for developing and testing the Open-RL framework.

---

## 0. Development Setup Scenario
In most scenarios, developers work on local machines (such as macOS or Linux laptops) that do **not** have local NVIDIA GPUs. Instead, they use a remote GCP VM with NVIDIA GPUs (such as `b7`) as the dev-test target.
Many Makefile targets that need to interact with the remote machine accept a `REMOTE_HOST=<host_name>` parameter (e.g. `make push-vm REMOTE_HOST=b7`).

---

## 1. Project Environments

Open-RL uses `uv` for environment isolation. There are two primary projects:

- **Root Project**: Contains the gateway, workers, pytest suite, and `openrl` CLI.
- **Client/Examples Environment (`examples`)**: Contains recipes, client-side SDK compatibility checks, and E2E integration test scripts.

Run Python tools directly with uv. Make is reserved for image, Kubernetes, and
VM shell workflows. Use `uv --project examples ...` only for commands belonging
to the separate examples project.

---

## 2. Fast Syntax Validation & Running Unit Tests

Select the machine environment once, then run commands without syncing:
```bash
uv sync --extra cpu       # CPU development machine
# uv sync --extra gpu     # GPU development VM instead
uv run --no-sync pytest
```

Pytest collects the existing unittest-style cases, so tests can migrate
incrementally instead of being rewritten mechanically.

---

## 3. Training and Integration Contracts

The default suite includes a real, tiny LoRA training step in one process. It
uses the production request processor and worker, but does not start Redis,
HTTP, a sampler, or a time-slicer:

```bash
uv run --no-sync pytest tests/test_in_process_training.py -s
```

The Tinker SDK/HTTP boundary is a separate contract. Install the examples
environment once before running it:

```bash
uv --project examples sync --group test
PYTHONPATH=examples/sft/pig-latin uv --project examples run --no-sync pytest \
  -m tinker_contract tests/test_piglatin_qwen.py -v
```

The cluster contract never starts infrastructure. Point it at an existing
deployment; it runs one bounded tiny-RL recipe and cleans up its workers:

```bash
OPENRL_BASE_URL=https://gateway.example \
  uv run --no-sync pytest -m distributed tests/test_distributed_contract.py -s
```

Large models, concurrent runs, and time-slicing are benchmarks. Launch them as
normal recipes with `openrl launch` so failures remain visible through the CLI
and cluster dashboard instead of being hidden inside a mega test fixture.

---

## 4. Syncing & Testing on Remote GPU Hosts (e.g., `b7`)

### Synchronization:
To push your current workspace to a remote test machine:
```bash
make push-vm REMOTE_HOST=<host_name>
```
To pull changes back:
```bash
make pull-vm REMOTE_HOST=<host_name>
```

### Running Tests on the Remote Machine:

**Option A: Direct SSH Execution (Simple)**
Run the command directly via SSH:
```bash
ssh <host_name> "export PATH=\$PATH:\$HOME/.local/bin && cd ~/open-rl && <test_command>"
```

**Option B: Within a Tmux Session (Optional)**
If there is a persistent active tmux session (e.g., `work`) on the remote machine, you can run tests and monitor them without losing progress if you disconnect:
1. Send the test command to the tmux session:
   ```bash
   ssh <host_name> 'tmux send-keys -t work "export PATH=\$PATH:\$HOME/.local/bin && cd ~/open-rl && <test_command>" C-m'
   ```
2. Monitor the pane output:
   ```bash
   ssh <host_name> "tmux capture-pane -t work -p"
   ```

---

## 5. Required System Dependencies on VM
If you encounter errors during E2E training or evaluation on a fresh GPU VM, ensure these system packages are installed:

- **`redis-server`**: Required by the Accelerator Time-Slicer for memory/state synchronization in FFT/time-slicing scenarios (`sudo apt-get install -y redis-server`).
- **`python3-dev`**: Required for compiling custom Triton runtime kernels during vLLM engine initialization (`sudo apt-get install -y python3-dev`).

---

## 6. Repeatable Kubernetes & Deployment Workflows

When debugging or executing distributed E2E benchmarks on Kubernetes (such as `fft-gsm8k-rl-x2`), always follow these standard lifecycle workflows:

### Deploying Python and Runtime Changes
Normal Python edits use the content-addressed source deployment and do not
require a version bump or image build:
```bash
uv run --no-sync openrl deploy
```

Use `make push-to-cluster` only for dependencies, CUDA/runtime files,
Dockerfiles, the time-slicer, or Kubernetes manifest changes.

### Cleaning Up Stale Worker Pods & Background Tasks
Aborting a cluster E2E test can leave active worker pods. Clean them up by label
before relaunching runs:
```bash
kubectl delete pods -l timeslice.io/group=trainers --ignore-not-found
kubectl delete pods -l timeslice.io/group=samplers --ignore-not-found
```

### Agent inspection loop

Prefer the bounded control CLI over ad-hoc cluster scraping:

```bash
uv run --no-sync openrl doctor --json
uv run --no-sync openrl problems --json
uv run --no-sync openrl inspect <run-id-or-pod-or-node> --json
uv run --no-sync openrl logs <run-id> --component trainer --json
```

Use `openrl stop <run-id> --wait --json` only when ending that run is part of
the task. Launch benchmark recipes with `openrl launch`; do not hide cluster
lifecycle and cleanup inside a local test harness.
