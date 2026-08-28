# The GPU scheduler

A workload says how much accelerator memory it needs and brings its own pod
template. The scheduler decides which GPU it lands on and who it takes turns
with. That is the whole contract.

```yaml
apiVersion: openrl.io/v1alpha1
kind: Workload
metadata:
  name: fft-job-a-trainer
spec:
  role: trainer                # which node pools may host it
  modelId: job-a               # its identity everywhere
  ownerId: Qwen/Qwen3-0.6B     # optional: the unit of fairness it belongs to
  accelerator:
    memory: 28Gi               # peak accelerator memory, from the estimator
  template:                    # the complete worker pod, inline
    spec:
      containers:
        - name: worker
          image: ghcr.io/gke-labs/open-rl/worker:latest
```

Everything else — device, tier, claim, node — is derived and reported back in
`status`.

## The model, in one sentence

**Every workload first asks DRA for its own GPU via an ordered list of tiers;
only when the cluster says no — the pod marked Unschedulable — does it
book a seat on an existing claim's ClaimLedger and share by turns.**

- There is no free-capacity survey. The controller cuts a ResourceClaim whose
  `firstAvailable` alternatives are the device shapes that fit, tightest
  first, and kube-scheduler's allocation cycle is the mutex.
- No timers: kube-scheduler's Unschedulable verdict is the one fallback
  trigger. A pending claim with nowhere to fall back to stays standing as the
  retry vehicle — and as the autoscale signal, on fleets that have one.
- A ClaimLedger is the seat ledger for one allocated claim. Seats are booked by
  compare-and-swap, keyed by workload UID plus a per-booking assignment ID, so
  concurrent reconciles cannot double-book and a recreated workload cannot
  inherit a seat it didn't book.
- Sharing means time-slicing: several workers seated, exactly one resident in
  accelerator memory at a time, turns rotating between owners.
- `role` selects nodes, never claims: a trainer and a sampler share one GPU
  by turns.

## Layout

| path | what it is |
| --- | --- |
| `api/v1alpha1` | the CRDs: Workload (the request) and ClaimLedger (the seat ledger) |
| `internal/placement` | the decision. Pure functions, no Kubernetes imports |
| `internal/controller` | the part that reads and writes Kubernetes objects |
| `docs/design.md` | the design |

## Try it

The behaviors live in `internal/placement/behavior_test.go`: workers arriving
and leaving, with the estimator's real tier figures on the hardware we run,
played through the same decisions the controller makes.

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
kubectl apply -k scheduler/deploy/base
kubectl label node <node> openrl.io/enabled=true openrl.io/trainer=true
```

Applying it changes nothing about a running cluster: the scheduler only acts
on Workload objects. Node labels are policy, never hardware — the DRA
driver's ResourceSlices say what devices actually exist.

Labeling a node opts its GPUs in **exclusively**: the scheduler assumes its
own claims are the only GPU consumers there, so other GPU workloads on an
enabled node will collide with it. Give OpenRL whole nodes.

## Everything else

Assumptions and caveats, the estimator, worker identity, claim lifecycle,
and the future optimizations all live in
[`docs/design.md`](docs/design.md); a file-by-file tour with a suggested
reading order is [`docs/layout.md`](docs/layout.md). If the code and any
document disagree, the code is right.
