**Status:** Draft
**Author:** Shuby Mishra
**Reviewers:** —
**Last updated:** August 12, 2026

## Introduction

OpenRL is a self-hosted training platform that runs trainer and sampler
workers on accelerator hardware in Kubernetes. Today, its Kubernetes
deployment uses pre-created Dynamic Resource Allocation (DRA) ResourceClaim
objects: trainers share one fixed claim, samplers share another, and OpenRL's
time-slicer coordinates access to the devices behind them. This works for a
fixed deployment, but the accelerator capacity and its assignment to workers
must be decided before workloads arrive.

This design makes placement dynamic. OpenRL estimates how much accelerator
memory each trainer or sampler process needs. A Go controller places that
process onto a DRA claim and creates its Pod. Kubernetes and the DRA driver
choose the exact node and devices.

The placement rule is simple: use a free accelerator when possible; otherwise
assign another worker to an existing allocation. Several workers may be
assigned to one claim, but V1 allows exactly one of them to be loaded in
accelerator memory at a time. Workers take turns through suspend and restore.

## Background

The merged fft branch creates trainer and sampler Pods directly from the
gateway using role-specific templates. Those templates reference fixed
trainer and sampler claims and select a predetermined accelerator node type.
OpenRL creates and deletes Pods, but it does not estimate their memory or
create, select, and reclaim claims.

DRA already reports the hardware available to Kubernetes and allocates
devices to claims. OpenRL only needs to add workload policy: determine what
each worker needs, choose an eligible claim, and coordinate workers that
intentionally share it.

## Design

### Definitions

| Term | Meaning |
| --- | --- |
| Worker | One trainer or sampler process. This is the unit placed onto hardware. |
| Owner | The job or shared runtime that receives one turn when an allocation is contested. |
| Claim | A DRA accelerator allocation on one node. Multiple worker Pods may reference it. |
| Resident | The one worker on a claim whose state is currently loaded in accelerator memory. |

For every claim:

- Several workers may be assigned to the claim.
- At most one worker is resident and uses the allocation's compute.
- Every other assigned worker is suspended outside accelerator memory.
- A handoff completes suspension of the current resident before restoring the
  next worker.

### V1 limitation: no GPU-memory co-residency

V1 does not keep two independent worker processes loaded on the same
accelerator allocation, even when their estimated memory would fit together.
If a 45 GiB trainer and a 25 GiB sampler share an 80 GiB GPU, OpenRL still
suspends one before restoring the other.

This leaves usable accelerator memory idle and may add transfer overhead for
small workers. The tradeoff is a much smaller correctness surface: the
controller checks only whether each worker fits independently, and the
node-local scheduler tracks only one resident instead of maintaining and
recovering a packed resident set. A later version may retain multiple workers
as a node-local memory cache without changing worker identity or claim
membership.

The end-to-end flow is:

```
worker configuration ──> memory estimate ──> OpenRLWorker
                                                │
ResourceSlices + operator labels ──> Go controller
                                                │
                                                └──> ResourceClaim + Pod
                                                         │
                                                         └──> node-local scheduling
```

The gateway decides worker identity and produces the memory estimate. The Go
controller selects capacity and reconciles claims and Pods. Kubernetes and
DRA allocate the hardware. The node-local scheduler coordinates execution on
a shared claim.

### Hardware visible to OpenRL

ResourceSlice objects tell OpenRL what hardware exists. Node labels tell
OpenRL which of that hardware it is allowed to use.

The DRA driver publishes ResourceSlice objects describing devices, their
capacity and attributes, and the nodes that can access them. Platform
engineers opt nodes into OpenRL with labels:

```
openrl.io/enabled: "true"
openrl.io/trainer: "true"
openrl.io/sampler: "true"
openrl.io/max-workers-per-claim: "2"
```

The controller considers the intersection:

```
devices reported through ResourceSlices
∩ devices accessible from openrl.io/enabled nodes
= hardware visible to OpenRL
```

The role labels may restrict a node to trainers, samplers, or both. If
neither role label is present, both are allowed. `max-workers-per-claim`
limits how many workers may be assigned to one claim and defaults to 1. It
limits queueing and switch overhead; it does not permit multiple residents.

Labels express operator policy, not hardware facts. For example,
`openrl.io/trainer=true` allows trainers on a node; it does not assert that
the node contains an H100. The DRA inventory remains the source of truth for
the devices.

OpenRL can place different workers across any number of labeled nodes. A
single worker's claim must still be satisfiable on one node; V1 does not
combine GPUs from different nodes into one distributed worker.

### Worker identity

The gateway decides whether an API request reuses an existing worker or
creates a new one:

| Training path | Worker ownership | Why |
| --- | --- | --- |
| FFT | Each job gets its own owner and its own trainer and sampler workers. | The trainer changes the job's complete model weights and optimizer history. Sharing a trainer process would mix the jobs' parameters. |
| LoRA | Requests for the same base model reuse its workers until adapter capacity is full. | The base model stays fixed; each request changes only its adapter. |

An FFT job's trainer and sampler are separate workers with the same owner ID.
The trainer updates that job's weights, while the sampler serves snapshots of
those evolving weights.

Placement operates on workers because each process consumes memory.
Scheduling accounts by owner so that an owner does not receive extra turns
merely because it has more processes, requests, or adapters. The placement
controller treats the owner ID as opaque.

### Memory estimation

The estimator calculates the peak accelerator memory required by one worker.
Its inputs include the model, role, training kind, optimizer and offload
settings, context length, peak packed tokens in one forward pass, and sampler
KV-cache or adapter-slot configuration.

Peak packed tokens is not the same as model context length. If two
131k-token datums are packed into one padded forward, the activation estimate
must use the resulting 262k-token shape.

The estimator returns one placement quantity:

```yaml
memory: 50Gi
```

It does not choose a GPU model, node, device count, or sharding strategy. The
controller compares the estimate with the eligible DRA inventory and derives
an allocation. Because V1 has one resident per claim, estimates for workers
assigned to the same claim are never added together. OpenRL records the
estimator inputs, result, and version so the decision can be inspected later.

V1 may begin with a conservative table of measured configurations. A formula
or profiler can replace it without changing placement.

### Placement decision

The controller applies the same decision tree to FFT and LoRA workers:

| Cluster state | Placement | Runtime behavior |
| --- | --- | --- |
| A suitable free allocation exists. | Create a new claim. | The worker runs independently. |
| No free allocation exists, and the worker fits an existing claim independently. | Join that claim if its worker and host-memory limits allow. | Assigned workers suspend and restore as turns change. |
| The worker cannot fit on any eligible one-node allocation. | Leave it pending. | No hardware substitution or partial placement occurs. |

This policy spreads work while free accelerators exist and shares only under
contention. It does not bin-pack accelerator memory. When several existing
claims are eligible, V1 prefers the claim with the fewest assigned workers,
followed by claim name for deterministic placement.

The controller owns the claims it creates. It removes a worker's Pod and
claim membership when the worker leaves and deletes a claim after its final
member is gone. V1 does not migrate a running worker after placement.

### End-to-end scenarios

#### 1. Free GPUs on different nodes

Two labeled nodes each expose one free 80 GiB GPU. Two 50 GiB workers arrive.
The controller creates a separate claim for each worker, and DRA allocates
them across the two nodes. Both workers run independently.

This is normal multi-node cluster support. The limitation is only that one
worker cannot combine the two GPUs across the two nodes.

#### 2. Two workers would fit together

Only one 80 GiB GPU remains. An FFT trainer needs 45 GiB and its sampler
needs 25 GiB. Each worker fits independently, so the controller may assign
both Pods to the same claim when no free GPU exists.

Their combined 70 GiB would fit, but V1 deliberately does not use that fact.
Sampling and training alternate through an exclusive handoff:

```
trainer turn:  GPU [ trainer ]    host [ sampler ]
handoff:       suspend trainer -> restore sampler
sampler turn:  GPU [ sampler ]    host [ trainer ]
```

If no other worker is waiting, the current worker may remain resident.
Suspension is required when the scheduler hands the claim to a different
worker.

#### 3. Two workers do not fit together

The trainer instead needs 55 GiB and the sampler needs 35 GiB. Each fits the
80 GiB GPU, but their combined 90 GiB does not. They may still be assigned to
the same claim if its worker limit and host-memory budget allow it.

Runtime behavior is identical to scenario 2: at a safe boundary, the trainer
moves its model, gradients, and optimizer state to host memory before the
sampler wakes. When sampling finishes, vLLM sleep level 1 offloads the
sampler's weights and discards its KV cache so the trainer can restore. V1
never needs to distinguish whether two workers would fit together.

#### 4. Several jobs share one GPU

Two FFT jobs each have a 50 GiB worker on one 80 GiB claim. Their owners take
turns round-robin at safe batch or optimizer-step boundaries. If only one
owner has work, it keeps running.

A third worker first looks for free hardware elsewhere in the visible fleet.
If none exists, it may join a claim with an available worker slot. If every
eligible claim has reached `max-workers-per-claim`, it remains pending.

#### 5. Samplers use different model families

A Qwen sampler and a Llama sampler each require 50 GiB and are assigned to
one 80 GiB claim. They take turns because each runs in its own vLLM process
and sleeps its own engine before the other becomes resident.

Placement does not require a same-family or same-base-model rule. The workers
only need to fit independently and support suspension.

#### 6. LoRA and FFT workers share an allocation

An FFT worker needs 50 GiB and a LoRA worker needs 20 GiB. With no free GPUs
elsewhere, both may be assigned to one 80 GiB claim because each fits
independently.

They do not remain resident together even though their combined 70 GiB would
fit. They take turns through suspension under the same rule as any other
pair. Placement does not create a special boundary between LoRA and FFT.

#### 7. LoRA requests reuse workers before placement

Two LoRA jobs target the same Qwen base model. The gateway assigns both
adapters to the existing Qwen trainer and sampler processes, so the second
request does not create another OpenRLWorker or consume another worker slot
on a claim.

If the runtime reaches its adapter capacity, or a request uses a different
base model, the gateway creates another set of workers. Those new processes
then enter the normal placement flow.

#### 8. Node labels exclude otherwise free hardware

The DRA inventory reports a free H100 on a trainer-only node and a busy L4 on
a sampler node. A new sampler cannot use the H100 even though it is
technically capable of running there: the operator has not allowed samplers
on that node.

The sampler may join an eligible claim on the L4 if it fits independently and
a worker slot is available. Otherwise it remains pending. ResourceSlice
objects describe what exists; labels decide what OpenRL may use.

#### 9. max-workers-per-claim limits queue depth

Three 20 GiB workers arrive for one 80 GiB GPU configured with
`max-workers-per-claim: 2`. The first two may be assigned to the claim and
take exclusive turns. The third cannot join, even though it also fits the GPU
independently.

The controller looks for another free or shared allocation. If none exists,
the third worker remains pending. `max-workers-per-claim` is an operator
safety limit on wait time and switching overhead, not a memory calculation.

#### 10. GPU memory fits, but host memory does not

Two 50 GiB workers can each fit on an 80 GiB GPU. Time-slicing requires
parking one worker's state in host memory while the other is resident. If the
node lacks enough host-memory headroom, the controller does not place the
second worker on that claim.

The worker tries another eligible allocation or remains pending. This
prevents accelerator oversubscription from causing a node-level out-of-memory
failure.

#### 11. Capacity exists only across nodes

Two nodes each expose one 80 GiB GPU, but one worker requires 140 GiB. The
cluster has 160 GiB in aggregate, yet no single node can satisfy the worker,
so it remains pending.

If one node instead exposes two 80 GiB GPUs, the controller may create a
two-device claim on that node. Placement establishes that the memory exists;
the training runtime must still support that device shape.

#### 12. Capacity becomes free after placement

Two workers share one GPU because every other eligible accelerator was busy
when they arrived. Later, another GPU becomes free. V1 leaves the existing
workers on their original claim, so they continue taking turns while the
newly free GPU is available to new workers.

Moving one of the existing workers would improve throughput, but requires
rebalancing and is a future optimization.

#### 13. Workers leave a shared claim

Two workers share one claim. When the first worker exits, the controller
deletes its Pod and removes it from the claim, but the claim remains
allocated for the second worker. When the final worker exits, the controller
deletes the claim and returns the accelerator to the visible fleet.

#### 14. The memory estimate is wrong

A worker is estimated at 40 GiB and placed on an 80 GiB GPU. If it actually
peaks above 80 GiB, it may OOM during its turn. If the estimate is too high
instead, OpenRL may reject a placement that would have worked or request a
larger allocation than necessary.

V1 records the estimate and estimator version for diagnosis but does not
automatically correct future estimates. Feeding observed usage and OOMs back
into the estimator is a future optimization.

## Future optimizations

V1 chooses a predictable policy rather than a global optimum. Later versions
may add:

- node-local co-residency that keeps multiple inactive workers loaded when
  their memory fits;
- rebalancing when an accelerator becomes free;
- placement that accounts for suspend and restore cost;
- memory estimates informed by observed usage and OOMs;
- priorities, owner weights, or minimum turn durations;
- explicit device count and topology for TP or FSDP;
- distributed workers spanning multiple nodes; and
- concurrent execution through an isolation mechanism such as MIG or MPS.

These changes do not alter the V1 boundaries: DRA reports hardware, operators
choose the nodes OpenRL may use, the estimator describes a worker, the Go
controller places it, and the node-local scheduler coordinates shared access.

## Appendix A: Safety and accounting

Three checks govern whether a worker can join a claim:

1. **Independent fit:** the worker must fit the allocation by itself.
2. **Worker limit:** the claim must remain below `max-workers-per-claim`.
3. **Suspended fit:** the node must have enough host memory for every worker
   that may be parked while another is resident.

There is no resident-memory sum in V1. Each worker's memory estimate is
checked against the full allocation independently. For host-memory admission,
the conservative case parks every assigned worker except the smallest one,
which could be the current resident.

Pods sharing a claim intentionally receive the same devices, so OpenRL must
coordinate their execution. A worker may stay resident while no other worker
needs the claim, but a handoff must finish suspending it before restoration
of the next worker begins. A failed handoff does not grant the next worker
access.

The preferred suspension mechanism depends on the runtime:

| Runtime | Suspension mechanism |
| --- | --- |
| vLLM sampler | Sleep level 1 and wake |
| FFT trainer | Application-level host offload and restore |
| Other CUDA worker | CUDA process checkpoint and restore |

## Appendix B: Alternatives considered

**Pre-created claims.** This removes dynamic claim lifecycle from OpenRL but
preserves the up-front capacity partitioning the design is intended to
remove.

**Pack before spreading.** This keeps whole GPUs free for future jobs but
reduces current throughput. V1 spreads first and shares under contention.

**Co-resident GPU-memory packing.** Keeping several workers loaded can
eliminate transfers when their memory fits together. V1 rejects this
optimization because it requires resident-set accounting, eviction policy,
and recovery from partially completed evictions. It can be added later inside
the node-local scheduler without changing claim membership.

**Always suspend after a turn.** This simplifies handoff logic but pays
transfer cost even when no other worker is waiting. V1 requires suspension
before a different worker becomes resident, not merely because the current
turn ended.

**Process checkpointing for every worker.** This is a useful fallback, but
application-aware sampler sleep and trainer offload move less unnecessary
state.

**Model compatibility groups.** These are unnecessary for V1 placement
because workers never remain resident together. Each only needs to fit
independently and support suspension. LoRA process reuse is decided before
placement.

**Native DRA without OpenRLWorker.** A Pod can request a claim directly, but
OpenRL would have no durable object for worker identity, the memory estimate,
placement status, and cleanup.

## Appendix C: Inputs, outputs, and the reconcile sequence

### Inputs, read fresh on every pass

The worker itself:

```yaml
apiVersion: openrl.io/v1alpha1
kind: OpenRLWorker
metadata:
  name: trainer-job-123        # identity; pod and claim names derive from it
spec:
  role: trainer                # which node pools may host it
  modelId: job-123             # configuration for the process, never identity
  ownerId: job-123             # fairness unit; opaque, passed through raw
  memory: 60Gi                 # the estimator's figure; must be positive
  estimatorVersion: v1
```

The spec is immutable (CEL-enforced): change by deleting and recreating.

The standing inputs: node labels (`openrl.io/enabled`, `/trainer`,
`/sampler`, `/max-workers-per-claim`), ResourceSlices from the configured
driver -- only the latest complete generation of each pool counts -- giving
device count and per-device memory, and each node's allocatable host memory.

### Outputs, at most three writes per pass

The ResourceClaim, when one is cut:

```yaml
metadata:
  name: claim-trainer-job-123-6b2f01a   # worker name + UID: unique per
                                        # incarnation, stable within one
  labels:
    openrl.io/managed: "true"
    openrl.io/accelerator-count: "1"    # the claim's shape contract
    openrl.io/device-memory: 80Gi
    openrl.io/sized-against: node-a     # reserves the pool until DRA decides
spec:                                   # one request, exact count, CEL bounds:
  # floor  = the worker's per-device share
  # ceiling = the device size the claim was priced against, so DRA cannot
  #           substitute a bigger device placement never chose
```

The Pod: rendered from the operator's template, with the controller's stamps
-- the claim reference, two ORed node-affinity terms (explicit role label, or
no role labels at all: the documented default), an owner reference to the
worker, and the time-slice contract (group = claim name, owner, job id) as
env vars carrying exact values and labels carrying sanitized copies.

The status: `phase`, `deviceCount`, `memoryPerDevice`, `hostMemoryWhenParked`,
`estimatorVersion`, `claimName`, `podName`, `nodeName`, `reason`, and the
`Placed` condition.

### The sequence

1. A watch event names one worker; reconciles run one at a time, because
   placement is a fleet-wide decision.
2. **Deleting?** Tear down: delete the pod, requeue until it is verifiably
   gone, then release the finalizer. The CR -- and its memory booking --
   survives the pod's termination grace, so a seat can never free while the
   process still holds the device.
3. Ensure the finalizer; reject non-positive memory as Failed.
4. **Read the fleet:** pools from slices x labels, managed claims, and
   occupancy rebuilt from every worker's status.
5. **Adopt reality:** the pod's claim label outranks a lost status; a pod
   owned by a previous incarnation (different ownerRef UID) is replaced,
   never adopted; a vanished claim means re-place.
6. **No claim yet?** Decide: a free pool cuts a new claim (unallocated, its
   pool reserved) -> Placing; otherwise an allocated claim that passes the
   three admission checks is joined -> Placing; otherwise Pending with a
   reason that distinguishes busy from impossible, retried each interval
   until the placement timeout turns "not yet" into Failed.
7. **Pod:** create it if missing; a pod bound to a stale claim is deleted
   and rebuilt, because spec.resourceClaims is immutable.
8. **Report:** Running once the pod runs, the node read from the claim's
   allocation; failures carry the scheduler's own words.
9. In the background, the reclaim sweep deletes managed claims that no
   worker names, no live pod uses, and DRA no longer reserves, after a
   grace period.
