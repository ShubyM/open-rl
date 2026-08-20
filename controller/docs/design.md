# Dynamic Accelerator Placement and Time-Slicing for OpenRL

**Status:** Draft
**Author:** Shuby Mishra
**Reviewers:** —
**Last updated:** August 19, 2026

## Introduction

OpenRL runs trainer and sampler workers on accelerator hardware in Kubernetes. Today, the deployment uses pre-created Dynamic Resource Allocation (DRA) `ResourceClaim` objects. Trainers share one fixed claim, samplers share another, and OpenRL coordinates access to the devices behind them.

This requires accelerator capacity and worker assignment to be decided before workloads arrive.

This design makes placement dynamic:

```text
API request
    |
    v
OpenRL API server
    |
    | create OpenRLWorkload
    v
placement controller
    |
    | create OpenRLClaimGroup, ResourceClaim, and Pod
    v
Kubernetes, DRA, and cluster autoscaler
    |
    | allocate or provision a GPU
    v
worker runs alone or joins an allocated ClaimGroup
    |
    v
node-local time-slicing
```

The API server only creates desired state. It does not inspect Kubernetes placement state.

A new worker first requests its own GPU through DRA. If that allocation remains pending, the controller may move the worker to an existing allocated `OpenRLClaimGroup`. Multiple workers in one group share the same `ResourceClaim` and take turns on its GPU.

## Overview

OpenRL uses `openrl-system` for:

* the API server;
* the placement controller;
* `OpenRLWorkload` objects;
* `OpenRLClaimGroup` objects;
* managed `ResourceClaim` objects;
* worker Pods.

The main objects are:

| Object             | Meaning                                             |
| ------------------ | --------------------------------------------------- |
| `OpenRLWorkload`   | One desired trainer or sampler runtime process.     |
| `OpenRLClaimGroup` | The workers assigned to one accelerator allocation. |
| `ResourceClaim`    | The DRA request and resulting device allocation.    |
| Worker Pod         | The process consuming that allocation.              |

There are two distinct forms of sharing:

```text
API jobs --------> runtime processes --------> OpenRLClaimGroups
                    runtime reuse               accelerator sharing
```

LoRA jobs may reuse runtime processes before placement. FFT jobs do not. Distinct runtime processes may later share one ClaimGroup under accelerator contention.

## 1. API server handling

The API server decides which runtime processes must exist.

### LoRA requests

Compatible LoRA jobs may reuse one trainer and one sampler runtime:

```text
LoRA job A ---\
               +--> Qwen trainer runtime
LoRA job B ---/

LoRA job A ---\
               +--> Qwen sampler runtime
LoRA job B ---/
```

The API server derives a runtime key from properties that determine whether reuse is safe, including:

* base-model and tokenizer revision;
* runtime configuration;
* dtype or quantization;
* context limit;
* adapter format and capacity;
* trainer or sampler role.

Placement is not part of the runtime key.

The API server also selects a runtime instance:

```text
lora-qwen-0-trainer
lora-qwen-0-sampler
```

When that runtime reaches adapter capacity, the API server selects another instance:

```text
lora-qwen-1-trainer
lora-qwen-1-sampler
```

Adapter assignment is API-server application state. The placement controller does not manage adapter slots.

### FFT requests

FFT jobs never reuse trainer or sampler processes:

```text
FFT job A --> trainer A
          \-> sampler A

FFT job B --> trainer B
          \-> sampler B
```

Each FFT job has its own mutable model weights and optimizer state. Its sampler serves snapshots of those job-specific weights.

The workloads are named from the job:

```text
fft-<job-id>-trainer
fft-<job-id>-sampler
```

The trainer and sampler remain separate processes but use the same owner ID for time-slicing fairness.

### API server boundary

The API server does not:

* list Nodes;
* inspect `ResourceSlice` objects;
* calculate free accelerator capacity;
* list or select claims or ClaimGroups;
* inspect worker Pods;
* choose a node or device;
* wait for placement before accepting another request.

Its placement-facing operation is to create or delete an `OpenRLWorkload`.

It may read workload status to report readiness or failure. Status does not participate in runtime identity or hardware selection.

## 2. `OpenRLWorkload`

One `OpenRLWorkload` represents one desired trainer or sampler runtime process.

The API server creates workloads unconditionally using deterministic names.

### Name collisions

For two concurrent compatible LoRA requests:

```text
request A creates lora-qwen-0-trainer -> Created
request B creates lora-qwen-0-trainer -> AlreadyExists
```

`AlreadyExists` means the selected runtime has already been requested and should be reused. It does not mean placement or startup has completed.

The second LoRA request does not create another:

* workload;
* worker Pod;
* ClaimGroup member;
* placement attempt.

For FFT, `AlreadyExists` means the same job-specific workload was submitted more than once. It provides idempotency, not reuse between jobs.

### Workload API

```yaml
apiVersion: openrl.io/v1alpha1
kind: OpenRLWorkload
metadata:
  name: fft-job-123-trainer
  namespace: openrl-system

spec:
  role: trainer
  trainingKind: fft
  modelId: job-123
  ownerId: job-123

  accelerator:
    memory: 60Gi

  estimatorVersion: v1-tier-table
  workerContainerName: worker

  template:
    metadata:
      labels:
        openrl.io/role: trainer
    spec:
      restartPolicy: OnFailure
      containers:
      - name: worker
        image: ghcr.io/gke-labs/open-rl/trainer:latest
        command:
        - uv
        - run
        - python
        - -m
        - server.training_requests_processor
        args:
        - --model-id
        - job-123
        env:
        - name: OPEN_RL_TIME_SLICE_OWNER
          value: job-123
        - name: OPEN_RL_WORKLOAD_ID
          value: fft-job-123-trainer
        resources:
          requests:
            cpu: "12"
            memory: "100Gi"
          limits:
            memory: "110Gi"
```

### Inline Pod template

The workload contains the complete Pod template. The placement controller has no role-specific ConfigMap template and no per-model overlay.

The API server supplies everything known before placement:

* image, command, and arguments;
* environment variables;
* CPU and host-memory requests;
* volumes;
* service account;
* security context;
* tolerations;
* sidecars.

`workerContainerName` identifies the container that consumes the accelerator.

The controller adds only placement-derived fields:

* the Pod-level `ResourceClaim` reference;
* the worker container’s claim-consumption entry;
* affinity for eligible OpenRL nodes;
* the ClaimGroup and assignment IDs;
* the Pod owner reference.

Admission rejects placement-owned fields in the supplied template, including:

* `nodeName`;
* node selectors;
* required node affinity;
* existing DRA claim references;
* accelerator extended-resource requests;
* OpenRL-owned assignment and time-slice fields.

The workload spec is immutable. Changing the runtime configuration, Pod template, resources, or accelerator requirement requires deleting and recreating the workload.

## 3. Single-GPU execution model

Each `OpenRLWorkload` currently consumes exactly one GPU.

The estimator reports the peak accelerator memory required on that device:

```yaml
accelerator:
  memory: 60Gi
```

The controller does not divide this requirement across multiple GPUs. If no single eligible GPU can hold the workload, the workload cannot be placed by the current runtime.

The trainer does not currently use `torchrun`, data parallelism, FSDP, or tensor parallelism. The sampler also runs on one GPU.

Multi-GPU support will be introduced in stages:

1. **Data parallelism:** multiple one-GPU replicas placed and started as one process group.
2. **Tensor parallelism:** multiple GPUs used by one model replica, with explicit model-compatible device counts.
3. **Combined DP and TP:** a two-dimensional process mesh.

A future placement plan may look like:

```yaml
placementPlans:
- replicas: 4
  gpusPerReplica: 1     # data parallel

- replicas: 2
  gpusPerReplica: 4     # data + tensor parallel
```

The controller will choose from explicit plans. It will not infer distributed layouts by dividing aggregate memory by device size.

## 4. `OpenRLClaimGroup`

An `OpenRLClaimGroup` represents one accelerator allocation and the workloads assigned to it.

Each ClaimGroup owns one `ResourceClaim`. In the current single-GPU model, that claim allocates one GPU.

```text
OpenRLClaimGroup
    |
    +--> ResourceClaim
    |
    +--> workload A
    +--> workload B
    \--> workload C
```

A group begins with one member. More workers may join after dedicated placement fails.

### ClaimGroup API

```yaml
apiVersion: openrl.io/v1alpha1
kind: OpenRLClaimGroup
metadata:
  name: group-fft-job-123-trainer-a81f4c
  namespace: openrl-system

spec:
  resourceClaimName: claim-fft-job-123-trainer-a81f4c

  members:
  - workload:
      name: fft-job-123-trainer
      uid: 4f164c5e-7e3f-4dc3-a6ec-6997db852f52
    ownerId: job-123
    assignmentId: 6f04fe8d-7971-480d-aee5-5ee228c4da3f

status:
  phase: Allocated
  nodeName: accelerator-node-1
  deviceMemory: 80Gi
```

`spec.members` is the authoritative assignment record.

Membership uses workload UIDs rather than only names. Deleting and recreating a workload under the same name creates a different workload identity.

The ClaimGroup owns its `ResourceClaim`. Worker Pods remain owned by their `OpenRLWorkload`.

A ClaimGroup is not owned by its first workload because it may continue serving other members after that workload exits.

### Assignment identity

Every membership entry has an `assignmentId`. The same value is recorded in workload status and stamped into the Pod:

```yaml
status:
  assignment:
    groupName: group-fft-job-123-trainer-a81f4c
    assignmentId: 6f04fe8d-7971-480d-aee5-5ee228c4da3f
```

```text
OPEN_RL_CLAIM_GROUP=group-fft-job-123-trainer-a81f4c
OPEN_RL_ASSIGNMENT_ID=6f04fe8d-7971-480d-aee5-5ee228c4da3f
```

A Pod may use the accelerator only when its workload UID and assignment ID match a current ClaimGroup member.

This prevents an obsolete Pod or stale controller action from becoming an active assignment.

## 5. Controller placement

The controller first attempts dedicated placement. If that allocation remains pending, it may move the workload to an allocated ClaimGroup.

### Dedicated placement

For a workload without an assignment, the controller:

1. creates a deterministic ClaimGroup containing that workload;
2. creates the group’s one-GPU `ResourceClaim`;
3. records the assignment in workload status;
4. creates the worker Pod referencing the claim.

```text
OpenRLWorkload
    |
    v
dedicated OpenRLClaimGroup
    |
    v
one-GPU ResourceClaim
    |
    v
worker Pod
```

The claim describes an acceptable GPU shape but does not name a currently free device or node.

Kubernetes and DRA decide whether the claim can be allocated.

### Device-size ordering

The controller maintains a cached catalog of device memory sizes. The catalog describes available device types, not current free-device counts.

A worker requiring 20 GiB may request ordered one-GPU alternatives such as:

```text
24 GiB
48 GiB
80 GiB
```

A worker requiring 60 GiB may request:

```text
80 GiB
```

The exact request is expressed through ordered, bounded DRA alternatives. This preserves tight-fit ordering without OpenRL surveying which individual devices are free.

A single-GPU 60 GiB FFT worker therefore cannot be placed on several L4s. It needs one device that satisfies the full 60 GiB requirement.

### Concurrent reconciliation

Workloads may reconcile concurrently.

Correctness does not depend on one global reconcile loop or one elected writer. Shared state is updated through Kubernetes optimistic concurrency.

To join a group, a reconciler:

1. reads the ClaimGroup;
2. validates the allocated GPU against the workload;
3. adds a member with a new assignment ID;
4. updates the ClaimGroup using its current `resourceVersion`.

If another reconciler updates the group first, the stale update receives a conflict and retries:

```text
reconciler A reads version 10
reconciler B reads version 10

A updates version 10 -> success
B updates version 10 -> conflict
B rereads and retries
```

A stale candidate read may cause a retry or a less balanced first choice. It cannot overwrite another membership update.

The controller uses indexed caches to find:

* allocated ClaimGroups;
* group node and device memory;
* current members;
* workload assignments.

It does not re-read the entire accelerator fleet for every workload.

### Committing an assignment

ClaimGroup membership and workload status are separate Kubernetes objects, so assignment is committed through an idempotent sequence:

1. add the member to the target ClaimGroup;
2. update workload status with the same assignment ID;
3. create the Pod only after the two records agree;
4. remove speculative membership if the workload assignment is superseded.

The runtime gate validates the assignment again before granting accelerator residency.

## 6. Falling back to sharing

A dedicated claim may remain unallocated because no matching GPU is currently available.

After the configured scale-out grace period, the controller may assign the workload to an existing allocated ClaimGroup.

### Eligible groups

A ClaimGroup is eligible when:

* its `ResourceClaim` is allocated;
* it owns one GPU;
* that GPU has enough memory for the workload;
* its node permits the workload role;
* any operator-owned membership guardrail permits another member.

Host-memory capacity is represented through the worker Pod’s memory request. Because the shared claim constrains the Pod to the claim’s node, kube-scheduler checks the Pod against that node’s remaining CPU and host memory.

OpenRL does not maintain a second node-memory allocator.

Among eligible groups, the controller prefers:

1. the smallest allocated GPU that fits;
2. fewer group members;
3. group name.

This ordering is a placement preference. Conditional membership updates provide correctness.

### Moving to a shared group

Changing claims requires recreating the worker Pod because Pod claim references are immutable.

The controller:

1. books the target ClaimGroup;
2. records a new assignment ID in workload status;
3. revokes the previous assignment;
4. deletes the old Pod and waits for it to terminate;
5. removes the workload from its dedicated group;
6. creates a Pod referencing the shared group’s claim;
7. deletes the empty dedicated group and claim.

```text
dedicated pending group
        |
        v
book allocated shared group
        |
        v
commit new assignment
        |
        v
delete old Pod
        |
        v
create shared Pod
```

If the old dedicated claim allocates during this transition, its obsolete assignment is not granted runtime access.

Claim sharing is automatic under contention. There is no workload-level sharing flag.

## 7. Kubernetes, DRA, and autoscaling

The placement controller defines acceptable device requests. Kubernetes and DRA choose the actual allocation.

```text
ClaimGroup creates ResourceClaim and Pod
        |
        v
kube-scheduler evaluates Pod and claim
        |
        +--> existing device matches
        |       |
        |       v
        |   claim allocates
        |
        \--> no current device matches
                |
                v
            Pod remains pending
```

Multiple Pods may reference the same manually managed `ResourceClaim`. DRA records the authorized Pod consumers, while OpenRL controls which process is resident.

### Scale-up behavior

A pending Pod with an unallocated DRA claim may cause the cluster autoscaler to add a compatible node, when the autoscaler and DRA driver support DRA-aware scale-up.

Dedicated-first placement therefore creates a choice:

```text
no current GPU
    |
    +--> wait for a new node
    |
    \--> share an existing allocated GPU
```

The controller uses an operator-owned `scaleOutGracePeriod` to choose between them.

During the grace period:

* the dedicated ClaimGroup, claim, and Pod remain pending;
* DRA continues trying to allocate the claim;
* the cluster autoscaler may add a compatible node.

After the grace period:

* if the dedicated claim allocated, the worker remains independent;
* if the claim is still unallocated and an eligible shared group exists, the worker moves to that group;
* if no shared group exists, the dedicated attempt remains pending and continues providing a scale-up signal.

```text
short grace period
    -> favor sharing and lower infrastructure growth

long grace period
    -> favor dedicated execution and scale-out
```

This is platform policy, not a workload field.

### Autoscaler race

A new node may already be provisioning when OpenRL moves the workload to a shared group.

Deleting the dedicated pending Pod removes its future scale-up signal, but it may not cancel a node already requested by the autoscaler. That node may still arrive and becomes available for later workloads.

The grace period controls preference; it does not make scale-out and sharing mutually exclusive.

### Scale-down behavior

An allocated ClaimGroup continues holding its GPU while any worker remains assigned, including suspended workers.

The node is not idle from DRA’s perspective until the final member leaves and the `ResourceClaim` is deleted. This prevents scale-down from removing a node that still backs OpenRL workers.

## 8. Runtime time-slicing

A ClaimGroup with one member runs without handoffs.

A ClaimGroup with multiple members becomes a time-slice group:

```text
group-a / claim-a / GPU 0

worker A: resident
worker B: suspended
worker C: suspended
```

Every worker starts behind the time-slicer’s execution gate. It does not initialize or use the accelerator until granted residency.

The time-slicer grants execution only when:

* the workload UID is a current ClaimGroup member;
* the Pod’s assignment ID matches that member;
* no other member is resident.

### Handoff

```text
resident reaches a safe boundary
    |
    v
suspend resident
    |
    v
confirm suspension
    |
    v
restore next worker
    |
    v
grant execution
```

The next worker cannot restore before the current resident finishes suspension.

If no other owner is runnable, the current worker may remain resident.

Suspension depends on the runtime:

| Runtime           | Mechanism                                  |
| ----------------- | ------------------------------------------ |
| vLLM sampler      | Sleep and wake                             |
| FFT trainer       | Application-level host offload and restore |
| Other CUDA worker | CUDA process checkpoint and restore        |

### Fairness

Fairness is accounted by owner rather than worker count.

An FFT trainer and sampler from the same job share one owner ID and do not automatically receive two independent shares.

Several LoRA jobs may reuse one runtime process and therefore remain one ClaimGroup member and one group-level fairness participant.

Shared accelerator service is best effort. The design does not guarantee a minimum accelerator share or maximum delay between turns.

## 9. Lifecycle and recovery

### Object creation

The dedicated ClaimGroup name derives from the workload name and UID:

```text
group-<workload-name>-<uid-prefix>
```

The claim name derives from the group name.

Concurrent creates use the same deterministic names. One create succeeds and the other reconciler adopts the existing object.

If reconciliation stops after creating only part of the object graph:

* ClaimGroup reconciliation creates a missing claim;
* workload reconciliation creates a missing Pod;
* assignment reconciliation repairs mismatched membership and status.

No age-based grace period is required to distinguish a partially created claim from an abandoned claim.

### Workload deletion

A workload finalizer keeps the workload present until its Pod is gone.

Deletion proceeds as follows:

1. revoke the current assignment ID;
2. delete the worker Pod;
3. wait for Pod termination;
4. remove the member from its ClaimGroup with a conditional update;
5. remove the workload finalizer.

Membership remains booked during Pod termination because the process may still hold the device.

### ClaimGroup deletion

When a group has no members:

1. verify that no live Pod references its claim;
2. delete the `ResourceClaim`;
3. wait for DRA to release the allocation;
4. remove the group finalizer;
5. delete the ClaimGroup.

A shared group may outlive the workload that originally created it.

## 10. Core scenarios

### Two FFT jobs

The API server creates separate trainer and sampler workloads for each job.

Each workload receives a dedicated ClaimGroup and claim. DRA allocates separate GPUs while capacity is available. Remaining claims stay pending and may later trigger scale-out or move into allocated groups.

No FFT processes are reused between jobs.

### Two compatible LoRA jobs

Both requests select the same deterministic trainer and sampler workload names.

For each name, one create succeeds and the other returns `AlreadyExists`.

Only one trainer runtime and one sampler runtime enter placement. ClaimGroup sharing is not involved in LoRA process reuse.

### Two incompatible LoRA jobs

Different runtime keys or a full adapter runtime produce another trainer and sampler instance.

Those new processes receive their own dedicated placement attempts and may later join allocated ClaimGroups.

### One GPU and two workers

Both workers create dedicated groups.

DRA allocates one claim. The other remains pending.

After the scale-out grace period, the pending worker joins the allocated group:

```text
group-a / GPU 0:
    worker A
    worker B
```

The workers then time-slice.

### FFT and LoRA under contention

The API server creates job-specific FFT workers and reused or newly created LoRA workers.

Placement treats each resulting process uniformly. An FFT and LoRA worker may share one allocated ClaimGroup when each independently fits its GPU and kube-scheduler can place the Pod on the group’s node.

### Concurrent joins

Two workloads select the same ClaimGroup concurrently.

One membership update succeeds. The other receives a `resourceVersion` conflict, rereads the group, and retries.

No assignment is overwritten.

### Autoscaling versus sharing

A worker’s dedicated claim remains pending while an allocated shared group exists.

* If a new node arrives during the grace period, the worker runs independently.
* If the grace period expires first, the worker joins the existing group.
* A node already being provisioned may still arrive after the worker begins sharing.

## Initial scope

The initial implementation assumes:

* one GPU per workload;
* one GPU per ClaimGroup;
* one node per workload;
* one resident process per ClaimGroup;
* trusted OpenRL workers that obey the runtime gate;
* inline Pod templates;
* automatic sharing after the scale-out grace period;
* best-effort time-slice fairness.

It does not include:

* data parallelism;
* FSDP;
* tensor parallelism;
* multi-node process groups;
* time-slicing of multi-GPU workers;
* GPU-memory co-residency for separate processes;
* per-workload hardware capability requirements;
* migration from a shared group to newly available capacity;
* throughput or turn-delay guarantees;
* MIG or MPS execution.

Future work will add data parallelism first, followed by tensor parallelism and combined process meshes. Other extensions include migration, bounded-service scheduling, observed-memory feedback, measured handoff costs, capability-aware device selectors, and multi-resident GPU caching.

## Alternatives considered

**API server creates Pods directly.** This mixes runtime identity with Kubernetes placement and requires the API server to inspect cluster state.

**OpenRL calculates free devices.** DRA already owns free-device allocation. Reconstructing availability in OpenRL introduces races and repeated fleet scans.

**No ClaimGroup resource.** A `ResourceClaim` records device allocation and authorized Pods, but not OpenRL owner identity, assignment generations, or time-slicing membership.

**Serial full-fleet placement.** This avoids some races but makes throughput depend on serialized fleet scans and a single active writer.

**Immediate sharing.** This minimizes pending time but removes the autoscaler’s opportunity to provide dedicated capacity.

**Always wait for scale-out.** This preserves independent execution but leaves workers pending when safe shared capacity already exists.

**Scale-out grace period.** This gives operators one policy control over the tradeoff between infrastructure growth and accelerator sharing.

**Multi-GPU placement by ceiling division.** Distributed execution strategies do not support every device count and do not scale memory linearly. Future DP and TP support will use explicit placement plans.
