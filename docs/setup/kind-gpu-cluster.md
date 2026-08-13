# A real-GPU kind cluster

One command turns a Linux box with NVIDIA GPUs into a single-node Kubernetes
cluster where the GPUs are real, allocatable Dynamic Resource Allocation
devices — the same `gpu.nvidia.com` DeviceClass and ResourceSlices a
production GKE DRA cluster publishes, without a cloud cluster.

```
hack/kind/bootstrap.sh    # fresh Ubuntu VM only: installs docker, kind,
                          # kubectl, helm, nvidia-container-toolkit, uv
hack/kind/kind-up.sh      # idempotent: cluster + GPU passthrough + DRA driver
```

After `kind-up.sh`, context `kind-openrl-gpu` has:

- the host's GPUs visible inside the kind node (`docker exec
  openrl-gpu-control-plane nvidia-smi -L`), injected by
  nvidia-container-toolkit's volume-mount device mode;
- the NVIDIA DRA driver (`dra-driver-nvidia-gpu` chart) publishing
  ResourceSlices, with the `gpu.nvidia.com` DeviceClass registered;
- Kubernetes 1.34, where DRA is GA — no feature gates.

Anything that consumes DRA can then be developed against it: create a
ResourceClaim, reference it from a pod, and the allocation is a real GPU.

## How the passthrough works

Three host-side settings, all checked (and offered as fixes) by `kind-up.sh`:

1. docker's default runtime is `nvidia`;
2. `accept-nvidia-visible-devices-as-volume-mounts = true` in
   `/etc/nvidia-container-runtime/config.toml`;
3. the kind node mounts `/var/run/nvidia-container-devices/all`, which the
   nvidia runtime interprets as "inject every GPU into this container".

The kind node also carries the `feature.node.kubernetes.io/pci-10de.present`
label statically, because the DRA driver's kubelet plugin selects nodes by
that node-feature-discovery label and kind runs no NFD.

## Fidelity notes

- One node: everything schedules onto the control plane. Multi-node GPU
  topologies, real network fabrics, and node failure modes are out of scope.
- The GPUs are shared with the host: anything else using them (a local
  training run, another cluster on the same box) will fight over memory.
- The cluster is meant to persist between runs; `kind delete cluster --name
  openrl-gpu` removes it.
