# OpenRL on an existing DRA cluster

The LoRA overlays schedule trainer and vLLM sampler processes on **separate
GPUs**. They install the gateway, scheduler, Redis, CRDs, RBAC, and shared storage
into `openrl-system`. Workers start when a client requests a model and wait, without a deadline,
until a GPU is free. The FFT GKE overlay adds FFT support and its
time-slicing daemons. LoRA workers keep exclusive GPUs in both deployments.

## Prerequisites

- NVIDIA DRA installed, with DeviceClass `gpu.nvidia.com` and published ResourceSlices.
- At least two GPUs that fit the model, on nodes reserved for OpenRL.
- kind: one node with at least two real GPUs and the `standard` storage class.
  Fake DRA devices can test scheduling, but cannot run PyTorch or vLLM, so a
  fake-device cluster verifies placement and nothing trains.
- GKE: Filestore CSI enabled for the default `standard-rwx` shared storage class.
  The default PVC requests 1TiB. FFT samplers keep a full-weight snapshot per
  model on this volume, about 28GiB for an 8B model, and the Hugging Face cache
  lives there too, so 100GiB fills within one small sweep. The class allows
  expansion, so the request can be raised in place later.
- `kubectl` and `kustomize` for the existing `make render` command.

Enroll the GPU nodes before deployment:

```bash
kubectl --context my-cluster label nodes <gpu-node> openrl.io/enabled=true
```

On autoscaled or spot pools, set the labels on the node pool rather than on
nodes: a recreated node comes back with only the pool's labels, and the
scheduler stops seeing it.

Nodes without `openrl.io/trainer` or `openrl.io/sampler` labels support both roles.
If either role label is present, only roles explicitly labeled `true` can use
that node. Trainer and sampler must have distinct eligible GPUs.

## Deploy

Choose the release bundle for the worker types you want:

| Bundle | Workers | Checkout overlay |
| --- | --- | --- |
| `openrl-lora.yaml` | LoRA | `k8s/deploy/lora` |
| `openrl-fft.yaml` | LoRA and FFT | `k8s/deploy/fft` |

Both use the same gateway, scheduler, Redis, and GKE storage. The FFT
bundle also installs the accelerator time-slicer and llm-d snapshot agent.
These require GPU nodes labeled `nvidia.com/gpu.present=true`, access to the
GKE NVIDIA driver directory `/home/kubernetes/bin/nvidia`, and permission to
run privileged pods with host PID and network access. FFT workers may share
claims with other FFT workers; they never share a GPU with a LoRA worker.
The scheduler does not reserve a fixed number of GPUs for either kind.

Apply one bundle, for example LoRA only:

```bash
kubectl --context my-cluster apply --server-side \
  -f https://github.com/gke-labs/open-rl/releases/latest/download/openrl-lora.yaml
```

From a checkout, render the overlay with the image tag to deploy. `latest`
follows main; a release tag pins that release:

```bash
make render OVERLAY=k8s/deploy/lora VERSION=latest \
  | kubectl --context my-cluster apply --server-side -f -
```

Use `OVERLAY=k8s/deploy/lora-kind` for kind. Both overlays use the same control
plane; only storage differs. All OpenRL images, including dynamic worker images,
are pinned by the existing renderer. The release bundle publishes the GKE
overlays as the two bundles above; kind is a development target and is not
published. Use `OVERLAY=k8s/deploy/fft` to render LoRA and FFT together.

Published bundles are ordinary YAML. You can download a bundle, edit its
environment variables or storage settings, and apply the edited file with
`kubectl apply --server-side -f <file>`. Kustomize is optional for that workflow.

Server-side apply is required because the Workload CRD embeds the Kubernetes
pod schema and exceeds the client-side apply annotation limit.

For custom storage or model defaults, add a Kustomize overlay over the platform
directory and pass that directory to `make render`. GKE needs shared RWX storage
across nodes; kind uses RWO storage because every process runs on one node.
Gateway and workers must mount the same PVC. If reusing a differently named PVC,
patch the gateway's `shared-storage` volume and `OPEN_RL_SHARED_PVC` environment
variable, and omit the default PVC resource from your overlay.

Check readiness and connect to the API:

```bash
kubectl --context my-cluster -n openrl-system rollout status deployment/redis-store
kubectl --context my-cluster -n openrl-system rollout status deployment/open-rl-scheduler
kubectl --context my-cluster -n openrl-system rollout status deployment/open-rl-gateway
kubectl --context my-cluster -n openrl-system port-forward svc/open-rl-gateway-service 8000:8000
```

The default model is public `Qwen/Qwen2.5-0.5B`. Workers download model weights
on first use; gateway readiness does not imply model loading has completed.

## Verify and upgrade

With the port-forward running, the existing tiny SFT example can also save and
sample the trained adapter:

```bash
uv --project examples run python examples/tiny/tiny_sft.py \
  base_model=Qwen/Qwen2.5-0.5B base_url=http://127.0.0.1:8000 sample_after_train=true
```

This checks falling training loss and successful generation from the saved
adapter. Inspect the resulting workers and their DRA allocations:

```bash
kubectl --context my-cluster -n openrl-system get workloads,resourceclaims
```

Before upgrading, finish active runs and delete their Workloads. Their specs
and GPU assignments are immutable; applying a new deployment does not migrate
existing workers. The scheduler cleans up pods and claims when their Workloads
are deleted. Retain the PVC to preserve saved adapters.
