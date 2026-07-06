# kind GPU Dev Cluster Setup Guide

This guide brings up the full FFT time-slice stack on one Linux GPU box using
kind. It includes DRA claims, RBAC, the Kubernetes worker manager, Redis queues,
cooperative sleep/wake coordination, and actual GPU training.

This verifies the local manifests, DRA allocation path, worker-manager pod
launching, queue flow, sleep/wake coordination, and small-model FFT training. It
does not verify the llm-d checkpoint backend, multi-node scheduling, or RWX
Filestore behavior from GKE.

## Requirements

- Linux with an NVIDIA GPU and driver r550 or newer.
- Docker with the NVIDIA runtime configured as the default runtime.
- kind 0.30 or newer.
- kubectl.
- Helm v3.
- nvidia-container-toolkit with `nvidia-ctk`.

Fresh Ubuntu GPU VMs can install everything except the NVIDIA driver with `bash hack/kind/bootstrap.sh`.

## Quickstart

Create the cluster and install the NVIDIA DRA driver:

```bash
make kind-up
```

Build the local images, load them into kind, and deploy the stack:

```bash
make kind-images
make kind-deploy
make kind-status
```

Forward the gateway:

```bash
kubectl port-forward svc/open-rl-gateway-service 9003:8000
```

In another shell, run the small FFT smoke test against the cluster:

```bash
make test e2e tiny-fft BASE_URL=http://localhost:9003
```

## What Runs Locally

The kind overlay reuses the FFT time-slice deployment shape but switches images
to `open-rl-server:kind-dev` and `open-rl-gateway:kind-dev`. The worker pod
templates use smaller CPU and memory requests suitable for small models on a
single local GPU.

The time-slicer runs with `--backend noop`. That means OpenRL still exercises
the cooperative acquire/release path and sleep/wake boundaries, but it does not
physically checkpoint CUDA state. Bare-metal Linux boxes can switch the
time-slicer to `--backend cuda` when `cuda-checkpoint` is available inside the
image and the DaemonSet is allowed to use `hostPID`.

## Single-GPU Mode

The trainer and sampler pods share the trainer `ResourceClaim`
`open-rl-trainer-gpu-1`. The sampler pod still names its container resource
claim `sampler-gpu`, but that name maps to the trainer claim in the pod-level
`resourceClaims` block.

On multi-GPU boxes, restore the separate sampler claim
`08-sampler-resourceclaim.yaml` and point sampler pods back to
`open-rl-sampler-gpu-1` so trainers and samplers can run on separate devices or
roles.

