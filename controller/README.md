# The GPU scheduler

A worker says how much accelerator memory it needs and which owner it belongs
to. The scheduler decides which bundle of accelerators it lands on and who it
takes turns with. That is the whole contract.

```yaml
apiVersion: openrl.io/v1alpha1
kind: OpenRLWorker
metadata:
  name: adapter-a
spec:
  role: trainer                # which node pools may host it
  modelId: adapter-a           # its identity everywhere
  memory: 6Gi                  # total accelerator memory, from the estimator
  ownerId: Qwen/Qwen3-0.6B     # optional: the unit of fairness it belongs to
```

Everything else — device count, per-device split, claim, node — is derived
and reported back in `status`.

## The model, in one sentence

**A claim is a bundle of accelerators; several workers may be assigned to it;
exactly one of them is resident at a time.**

- There is no co-residency in V1. Whatever the workers share, at most one
  process's state is loaded on the allocation; everyone else is suspended in
  host RAM. So estimates are never summed — each worker only has to fit the
  allocation *by itself* — and a handoff finishes suspending the outgoing
  worker before the next one is restored.
- The owner ID is an opaque string, compared and never interpreted. It is the
  unit of fairness: turns rotate between owners, so an owner never gets extra
  turns for having more processes, requests, or adapters. Naming none makes
  you an owner of one. Placement ignores it entirely.
- `role` selects nodes, never claims: a trainer and a sampler share one GPU
  by turns.
- No sharding, so device count is plain ceiling division — derived, never
  requested.

## Layout

| path | what it is |
| --- | --- |
| `api/v1alpha1` | the CRD: the request, and what was decided about it |
| `internal/placement` | the decision. Pure functions, no Kubernetes imports |
| `internal/controller` | the part that reads and writes Kubernetes objects |
| `docs/design.md` | the design |

## Try it

The behaviors live in `internal/placement/behavior_test.go`: workers arriving
and leaving, with the estimator's real tier figures on the hardware we run,
played through the same `Decide` the controller calls.

```
go test ./...
```

For the pipeline — real API server, real kube-scheduler, real DRA — there is
a kind smoke test that needs no hardware (the DRA example driver publishes
fake GPUs):

```
make smoke                                # kind + fake GPUs
USE_EXISTING_CLUSTER=1 DEVICE_CLASS=gpu.nvidia.com ./hack/kind-smoke.sh   # real GPUs
```

The controller only ever reads ResourceSlices and node labels, so fake and
real devices exercise the identical path; only the two env values differ.

## Deploy

```
kubectl apply -k ../k8s/deploy/scheduler
kubectl label node <node> openrl.io/enabled=true openrl.io/trainer=true openrl.io/max-workers-per-claim=4
```

Applying it changes nothing about a running cluster: the scheduler only acts
on OpenRLWorker objects. Node labels are policy, never hardware — the DRA
driver's ResourceSlices say what devices actually exist.

Labeling a node opts its GPUs in **exclusively**: the scheduler counts a
device as free unless one of its own claims holds it, so other GPU workloads
on an enabled node are invisible to placement and will collide with it. Give
OpenRL whole nodes.

## Everything else

Assumptions and caveats, the estimator, worker identity, claim lifecycle,
and the future optimizations all live in
[`docs/design.md`](docs/design.md); a file-by-file tour with a suggested
reading order is [`docs/layout.md`](docs/layout.md). If the code and any
document disagree, the code is right.
