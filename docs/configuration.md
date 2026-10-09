# Configuration

OpenRL is configured with environment variables. The examples below use plain
shell commands so they work even if `make` is not installed. The root
`Makefile` wraps the same commands for convenience.

## Run outside Kubernetes

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Start the API server and trainer in one process, sampling with torch:

```bash
BASE_MODEL=google/gemma-4-e2b \
uv run --extra cpu python -m uvicorn server.api_server:app --host 127.0.0.1 --port 9003
```

Because `REDIS_URL` is unset, the API server runs the trainer loop in its own
process and samples with torch. This works on a laptop without a GPU.

To train and sample on GPUs with vLLM, set `REDIS_URL`. The API server then
launches a trainer and a vLLM sampler as separate worker processes when a client
creates a model, and they share queues through Redis:

```bash
REDIS_URL=redis://127.0.0.1:6379/0 \
BASE_MODEL=google/gemma-4-e2b \
VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM \
TRAINER_CUDA_VISIBLE_DEVICES=0 \
SAMPLER_CUDA_VISIBLE_DEVICES=1 \
uv run --extra gpu python -m uvicorn server.api_server:app --host 127.0.0.1 --port 9003
```

The equivalent Makefile shortcuts are:

```bash
make server
REDIS_URL=redis://127.0.0.1:6379/0 VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM \
  TRAINER_CUDA_VISIBLE_DEVICES=0 SAMPLER_CUDA_VISIBLE_DEVICES=1 make server
```

## Core variables

| Env var | Default | What it does |
| --- | --- | --- |
| `BASE_MODEL` | unset | Hugging Face model id loaded by the trainer and, when using vLLM, by the sampler. |
| `SAMPLING_BACKEND` | `torch` in one process, `vllm` when `REDIS_URL` is set | Sampling backend selector. `torch` samples in the training process. `vllm` queues sampling requests for a vLLM sampler worker and needs `REDIS_URL`. |
| `REDIS_URL` | unset | Enables distributed mode by switching the request store to Redis. Leave unset for a single-machine run. |
| `OPEN_RL_FUTURE_TTL_S` | `300` | How long resolved request results stay readable by `retrieve_future` after a worker resolves them. |

## Server paths

| Env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_TMP_DIR` | `/tmp/open-rl` | Root directory for adapter snapshots under `peft/` and saved states under `checkpoints/`. |
| `OPEN_RL_TRAIN_TOKEN_BUDGET` | `0` | Maximum `batch_size * max_sequence_length` for padded trainer chunks inside one `forward_backward` request. `0` keeps the previous one-datum-at-a-time execution path. |
| `TRAINER_CUDA_VISIBLE_DEVICES`, `SAMPLER_CUDA_VISIBLE_DEVICES` | unset | GPUs given to the trainer and sampler worker processes the local worker manager launches. |

## Worker manager

| Env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_WORKER_MANAGER` | `local` | How the API server starts trainer and sampler workers: `local` runs them as subprocesses, `scheduler` creates `Workload`s for the OpenRL scheduler on Kubernetes, and `none` leaves workers to something else. |
| `OPEN_RL_ACCEL_TIMESLICER_SOCKET` | `/tmp/open-rl/accel-timeslicer.sock` | Unix socket path for a local accelerator time-slicer. Used when `OPEN_RL_ACCEL_TIMESLICER_HOST` is unset. |
| `OPEN_RL_ACCEL_TIMESLICER_HOST` | unset | Node-local accelerator time-slicer host for Kubernetes workers. When set, the worker uses TCP instead of the Unix socket; Kubernetes sets this from `status.hostIP`. |
| `OPEN_RL_ACCEL_TIMESLICER_PORT` | `9753` | Node-local accelerator time-slicer TCP port for Kubernetes workers. |

For local FFT subprocess mode, start `python -m accel_timeslicer.serve` before the
workers run. The local launcher tags each worker with a time-slice job id and
starts it in its own process group so the CUDA checkpoint backend can discover
the active GPU PIDs. Kubernetes deploys the equivalent process with the
`open-rl-accel-timeslicer` DaemonSet, which layers on top of the llm-d snapshot
backend by default for physical checkpoint/restore.

## vLLM variables

| Env var | Default | What it does |
| --- | --- | --- |
| `VLLM_ARCHITECTURE_OVERRIDE` | unset | Optional architecture override passed to the in-repo vLLM worker. Gemma 4 examples use `Gemma4ForCausalLM`. |
| `VLLM_ENABLE_MULTIMODAL` | `0` | By default the samplers pass `limit_mm_per_prompt={"image": 0, "video": 0}`. Text checkpoints published as `*ForConditionalGeneration` otherwise make vLLM reserve a multi-GiB encoder cache during startup that no OpenRL code path can use, which can OOM engine init. Set to `1` to restore stock vLLM behaviour. |

## llm-d sampler routing

For Kubernetes LoRA workers, set `openrl.sampler_router=llmd` in model metadata
and install the router resources with `kubectl apply -k k8s/deploy/llmd-router`.
Jobs sharing a sampler runtime must use the same router setting; use
`openrl.exclusive=true` for a separate pool.

Each routed sampler set has one Redis Stream and one CPU dispatcher pod
(the dispatcher, Envoy and the llm-d endpoint picker). The API stores each
request in the set's stream before returning its ID. The dispatcher pulls work
when it has a free slot and sends it through the router, which picks the
sampler. A request stays pending until one final result is saved, so accepted
work survives restarts of the API server, the dispatcher pod and the samplers.
A lost response can make a sampler generate a request twice; only one result is
kept. Only the holder of the set's Redis lease dispatches.

Requests must name fixed weights: a saved `sampler_weights` reference, or the
base model of a job that has not trained yet. Sampling a training job's live
adapter is refused, so a retry can never pick up newer weights.

| API server env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_SAMPLE_UNFINISHED_LIMIT` | `4096` | Unfinished requests per set. Further submissions receive HTTP 429. |
| `OPEN_RL_SAMPLE_DEADLINE_SECONDS` | `7200` | Total time a request may take, including queueing and retries. |
| `OPEN_RL_SAMPLE_MAX_SAMPLES` | `64` | Largest `num_samples` per request. |
| `OPEN_RL_SAMPLE_MAX_PAYLOAD_BYTES` | `16777216` | Largest stored request. |
| `OPEN_RL_SAMPLE_RESULT_TTL` | `1800` | How long a saved result stays readable. |
| `OPEN_RL_SAMPLE_GUARD_TTL` | `7200` | How long a finished request keeps refusing late attempts. |
| `OPEN_RL_DISPATCHER_IMAGE` | worker image | Image for the dispatcher container; the API server image is enough. |

The API server passes every `OPEN_RL_DISPATCH_*` variable to the dispatchers:

| Dispatcher env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_DISPATCH_ACTIVE` | `256` | Requests one dispatcher owns at once, including those waiting to retry. |
| `OPEN_RL_DISPATCH_MAX_ATTEMPTS` | `4` | Attempts per request. |
| `OPEN_RL_DISPATCH_ATTEMPT_TIMEOUT` | `1800` | Longest single attempt, cut short by the request deadline. |
| `OPEN_RL_DISPATCH_BACKOFF_BASE` / `_CAP` | `1` / `60` | Retry delay, doubled per attempt with jitter, up to the cap. |
| `OPEN_RL_DISPATCH_LEASE_TTL` | `30` | Lease length; renewed every `OPEN_RL_DISPATCH_RENEW_INTERVAL` (`10`). |
| `OPEN_RL_DISPATCH_RECLAIM_IDLE` | `60` | How long an unrenewed claim waits before another dispatcher takes it. |
| `OPEN_RL_DISPATCH_SHUTDOWN_GRACE` | `60` | Time active calls get to finish when the pod stops. |

Removing a set closes it to new requests and cancels its unfinished ones before
its dispatcher is deleted. Redis holds the only copy of accepted work: run it
with persistence (for example an append-only file) and keep its eviction policy
from removing these keys. An append-only file flushed every second can lose the
last second of accepted requests in a crash.

See docs/design/sampling-streams.md for the design.

## LoRA sampler snapshots

LoRA `save_weights_for_sampler` freezes the adapter into its own directory on
the shared volume before returning the reference, and samplers load that
snapshot by the reference. A reference therefore always names the same weights,
however long a request waits or however often it is retried, and vLLM can cache
prompt prefixes per adapter (prefix caching is on for LoRA samplers, off for
full fine-tuning, whose weights change in place under one name).

Save names and sequence IDs must be unique: overwriting a reference would
invalidate vLLM's adapter and prefix caches. A missing snapshot is an error.
References created before snapshots need to be saved again under a new name.
Snapshots are retained like checkpoints; clean them up when the saved clients
are no longer used.

## Client variables

| Env var | Default | What it does |
| --- | --- | --- |
| `TINKER_BASE_URL` | `http://127.0.0.1:9003` | Base URL used by example clients and scripts. |
| `TINKER_API_KEY` | `tml-dummy-key` | Passed through to the Tinker SDK. Local OpenRL does not enforce auth. |
| `HF_TOKEN` | unset | Required for gated Hugging Face models. `uv run hf auth login` is the easiest setup path. |
| `ENABLE_GCP_TRACE` | `0` | `1` exports OpenTelemetry traces to Google Cloud Trace. |
| `ENABLE_CONSOLE_TRACE` | `0` | `1` prints trace spans to stdout for debugging. |

## Kubernetes deployment

On Kubernetes the release bundles set these variables. The API server runs with
`REDIS_URL` and `OPEN_RL_WORKER_MANAGER=scheduler`; it creates a `Workload` for
each trainer and sampler, and the OpenRL scheduler starts their pods. The API
server writes each worker's environment into the `Workload`'s pod template, so
there are no worker deployments to configure by hand.
