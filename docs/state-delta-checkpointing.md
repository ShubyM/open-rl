# State Delta Checkpointing Plan

This plan makes state synchronization transport-independent. Correctness comes
from a semantic delta and apply verification, not from whether bytes moved
through shared memory, CUDA IPC, NCCL, HTTP, local disk, or object storage.

## Goal

Open-RL should have one model for:

```text
hot sync:
  trainer -> inference engine

durable checkpoint:
  trainer -> filesystem/object store

restore:
  checkpoint store -> trainer or inference engine

future runtime materialization:
  clean runtime image for a known state version
```

The invariant is:

```text
base state B + delta D(version=N) -> semantic state S_N
```

This must hold regardless of the transport used to move the bytes.

## Design Rule

Open-RL state synchronization is defined by semantic deltas, not transports.

A `StateDeltaManifest` describes the version transition and tensors required to
reach a target state. Transports move bytes for that manifest. Materializers
verify and apply the manifest to a runtime. Durable checkpoints store the same
manifest plus tensor payloads or durable references to those payloads, so hot
synchronization and restore share one state contract.

This abstraction covers LoRA now and full fine-tuning later. For LoRA, the delta
is a small set of adapter tensors. For full fine-tuning, the same manifest can
point to full-weight tensors, chunks, or shards.

## Pieces

### StateDeltaManifest

The manifest is the source of truth for correctness.

```python
@dataclass(frozen=True)
class TensorEntry:
    name: str
    normalized_name: str
    role: str
    dtype: str
    shape: tuple[int, ...]
    checksum: str | None
    storage_key: str


@dataclass(frozen=True)
class StateDeltaManifest:
    schema_version: int
    run_id: str
    version: int
    base_ref: str
    delta_id: str
    training_mode: str
    apply_target: str
    adapter_config_hash: str | None
    tensors: tuple[TensorEntry, ...]
    created_at: float
```

For LoRA, `name` is the trainer-side PEFT key and `normalized_name` is the
runtime apply key. For vLLM, that means converting:

```text
...lora_A.<adapter>.weight -> ...lora_A.weight
...lora_B.<adapter>.weight -> ...lora_B.weight
```

Validation checks:

```text
schema version
run_id
version
base_ref
training_mode
apply_target
unique storage keys
unique normalized tensor names
valid dtype and shape
adapter config hash
optional tensor checksums
```

### Transport

Transport knows how bytes move right now. It does not know LoRA semantics.

```python
class Transport(Protocol):
    def publish_bytes(
        self,
        manifest: StateDeltaManifest,
        tensors: Iterable[tuple[TensorEntry, torch.Tensor]],
    ) -> TransportReceipt:
        ...


@dataclass(frozen=True)
class TransportReceipt:
    transport: str
    delta_id: str
    version: int
    locations: dict[str, str]
    expires_at: float | None
```

Examples:

```text
shared memory:
  storage_key -> shm://name

CUDA IPC:
  storage_key -> cuda-ipc://handle

NCCL:
  storage_key -> nccl://session/sequence

file:
  storage_key -> file:///path/to/blob

object store:
  storage_key -> s3://bucket/key
```

### Materializer

Materializer owns semantic apply for one runtime.

```python
class Materializer(Protocol):
    def can_apply(self, manifest: StateDeltaManifest) -> bool:
        ...

    def apply(
        self,
        manifest: StateDeltaManifest,
        receipt: TransportReceipt,
    ) -> ApplyResult:
        ...
```

Apply checks:

```text
manifest version is newer than receiver version
base_ref matches receiver state
receipt delta_id/version match manifest
tensor count and names match
shapes match
dtypes match or an explicit runtime cast policy allows mismatch
checksums match when required
adapter_config_hash matches
target runtime supports apply_target
```

Materializers we care about:

```text
VLLMLoraMaterializer
SGLangLoraMaterializer
TrainerLoraMaterializer
FullWeightMaterializer later
```

### CheckpointStore

`CheckpointStore` is the durable Open-RL envelope. It does not define how bytes
move on the hot path. It records enough metadata to use the same checkpoint for
trainer resume or inference materialization.

```python
@dataclass(frozen=True)
class CheckpointMetadata:
    base_model: str | None
    model_id: str
    kind: str
    targets: tuple["trainer" | "inference", ...]
    adapter_ref: str | None
    optimizer_ref: str | None
    state_delta_ref: str | None
```

For LoRA today, the backing can stay simple:

```text
checkpoint/
  adapter_config.json
  adapter_model.safetensors
  optimizer.pt optional
  delta/... optional
  metadata.json committed last
```

The checkpoint is the durable thing a user or scheduler names. The
`StateDeltaManifest` is the semantic sidecar that lets a runtime verify and
apply the exact adapter tensors without caring whether the payload came from
shared memory, HTTP, local disk, or object storage.

### DeltaStore

`DeltaStore` is the common abstraction for hot and durable backing.

```python
class DeltaStore(Protocol):
    def write_delta(
        self,
        manifest: StateDeltaManifest,
        tensors: Iterable[tuple[TensorEntry, torch.Tensor]],
    ) -> DeltaRef:
        ...

    def read_delta(self, ref: DeltaRef) -> tuple[StateDeltaManifest, TensorReader]:
        ...
```

Hot stores can be short-lived:

```text
shm://...
cuda-ipc://...
nccl://...
```

Durable stores must write the payload first and commit the manifest last:

```text
file://...
object://...
tinker://...
```

## Flows

### Hot Sync

```text
trainer optim_step
  adapter tensors mutate

build manifest
  select exact tensors
  normalize names for apply target
  assign version
  compute config hash

write hot delta
  store/transport moves bytes
  returns delta ref or transport receipt

apply
  sampler reads bytes
  verifies manifest
  materializes runtime state
  records active version

sample
  inference engine serves version N
```

### Durable Checkpoint

```text
save checkpoint
  build same manifest
  write tensor payloads to checkpoint store
  write manifest last
  return durable delta ref
```

The durable ref should be enough to restore into either trainer or inference.

### Restore

```text
restore delta ref
  read manifest
  read tensor payloads
  verify payloads
  materialize into trainer or inference engine
  record restored version
```

### Future Runtime Materialization

Runtime image materialization should be a cache of an already verified semantic
state, not the source of truth.

```text
semantic state S_N
  -> optional clean runtime image / GCR snapshot
  -> restore fast path
  -> fallback to manifest + payload restore if image is invalid
```

## Cleanup Model

Cleanup is state collection, not file deletion.

Live roots include:

```text
active trainer adapter versions
active sampler adapter versions
public checkpoint refs
pending operations
resolved operations not past retention
runtime materializations with valid leases
```

Deletion requires:

```text
no live root reaches the delta
lineage is not required by a live descendant
lease token still matches before mutating or deleting a slot
```

This follows the TLA lesson: stale completions and public refs can keep states
alive, so cleanup must reason over roots, lineage, and leases.

## Migration Plan

### Step 1: Keep Manifest Pure

Keep `state_delta.py` small and independent:

```text
manifest dataclasses
LoRA normalization
config hash
optional tensor checksum
validation
```

No vLLM, SGLang, HTTP, shared memory, Redis, or checkpoint store imports.

### Step 2: Move Hot Byte Handling Out Of weight_sync.py

Before this migration, `weight_sync.py` owned too much:

```text
manifest construction
safetensors serialization
shared-memory creation
HTTP control request
fallback behavior
```

Current shape:

```text
weight_sync.py
  trainer-facing publisher
  manifest construction
  vLLM control request

delta_store.py
  shared memory store
  HTTP debug store
  file store
  transport receipt construction
  tensor payload serialization

vllm_sampler.py
  materializer endpoint
```

This removes shared-memory lifecycle, base64 encoding, and safetensors payload
serialization from `weight_sync.py`. `FileDeltaStore` uses the same
`DeltaStore` interface and writes `tensors.safetensors` before committing
`manifest.json`.

### Step 3: Replace /sync_lora_tensors Shape

Current shape is still transport flavored:

```text
POST /sync_lora_tensors
  manifest
  transport_receipt
  tensors_safetensors_shm
```

Target shape:

```text
POST /apply_delta
  delta_ref
```

or, for short-lived hot transports:

```text
POST /apply_delta
  manifest
  receipt
```

The endpoint name should reflect semantic apply, not tensor sync.

### Step 4: Make CheckpointStore Use The Same Delta

Checkpointing should not invent another format.

```text
save_weights
  write StateDeltaManifest + payloads
  keep existing PEFT files for compatibility

save_weights_for_sampler
  write hot StateDeltaManifest + payloads

restore
  read StateDeltaManifest + payloads
```

Hot sync and durable restore should differ only by backing store.

### Step 5: Add More Materializers

After vLLM LoRA works:

```text
SGLang LoRA materializer
trainer LoRA restore materializer
full-weight materializer
```

Full-weight sync can use vLLM native weight transfer more directly because it
expects base model parameter names.

## What Can Go Away

If this design lands, these should shrink or disappear:

```text
VLLMLoraTensorHttpTransferEngine
VLLMLoraTensorSharedMemoryTransferEngine
manual transport_receipt construction inside weight_sync.py
adapter config loading inside weight_sync.py
shared-memory lifecycle inside weight_sync.py
endpoint-specific naming like /sync_lora_tensors
file snapshot and hot sync as separate state concepts
```

`weight_sync.py` should become a thin facade:

```python
class WeightPublisher:
    def publish_for_inference(...):
        manifest, tensors = build_delta(...)
        delta_ref = hot_store.write_delta(manifest, tensors)
        materializer.apply(delta_ref)
        return PublishedState(...)
```

## Naming Rules

Avoid underscore methods in new code. Prefer:

```text
module-level functions
small public methods
plain dataclasses
composition over subclass hooks
```

Underscore names are acceptable only for Python protocol hooks or genuinely
private compatibility shims we do not expect humans to call.

## Open Questions

```text
1. Should hot sync always create a DeltaRef, or allow manifest+receipt inline?
2. Should checksums be required for all hot paths or only durable paths?
3. Should dtype casts create a new manifest or be recorded as an apply policy?
4. What is the first durable ref scheme: file://, delta://, or tinker://?
5. Should sampler active version state live in process memory, Redis, or both?
6. How long do short-lived hot delta refs live?
7. What lease token model do we want for runtime materializations?
```

## Proposed Next Code Shape

```text
src/server/state_delta.py
  pure manifest and validation

src/server/delta_store.py
  DeltaRef
  TransportReceipt
  SharedMemoryDeltaStore
  FileDeltaStore

src/server/weight_sync.py
  WeightPublisher
  LoraTensorSelector

src/server/vllm_materializer.py
  verify + materialize vLLM LoRA delta

src/server/vllm_sampler.py
  endpoint glue only
```

The near-term goal is not to add more machinery. It is to move the machinery to
the layer where it belongs so the trainer path gets smaller as durability gets
stronger.
