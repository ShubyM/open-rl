# Where everything is

The short version: two CRDs define the API, one pure package makes the
placement decision, one reconciler drives Kubernetes, and everything else is
glue you can read once and trust.

```
scheduler/controller/             the Go module
├── api/v1alpha1/
│   ├── workload_types.go         Workload: Spec (what the API server asks) and
│   │                             Status (claim + assignment). The +kubebuilder
│   │                             comments generate the CRD and its validation;
│   │                             doc comments here become `kubectl explain` text.
│   ├── claimledger_types.go       ClaimLedger: the seat ledger for one claim.
│   │                             Seats carry workload name+UID, assignment ID,
│   │                             owner, and host request; host memory, not a
│   │                             seat count, bounds how many may park.
│   ├── groupversion_info.go      scheme registration boilerplate
│   └── zz_generated.deepcopy.go  generated; never edit (make generate)
│
├── internal/placement/           THE DECISION. Pure functions, no Kubernetes,
│   │                             and no free-capacity survey anywhere.
│   ├── tiers.go                  Catalog (device shapes per role) and Tiers
│   │                             (ordered firstAvailable alternatives: fewest
│   │                             devices, then least waste). What a dedicated
│   │                             claim asks DRA for.
│   ├── placement.go              SelectClaim (which allocated claim a worker
│   │                             joins on the fallback) and the admission
│   │                             checks; Claim/Node/Fleet are plain structs.
│   ├── behavior_test.go          the end-to-end behaviors as arrivals and
│   │                             departures, with a simulated DRA playing the
│   │                             allocator. Start reading tests here.
│   ├── placement_test.go         unit tests for arithmetic and tie-breaks
│   └── tiers_test.go             property fuzz over the tier compiler
│
├── internal/controller/          THE KUBERNETES GLUE.
│   ├── workload_controller.go    Reconcile -> place(): cut a tiered claim,
│   │                             fall back to sharing on kube-scheduler's
│   │                             verdict (fallBackToSharing), abandon wedged
│   │                             allocated claims, render the pod. Watch
│   │                             wiring and status writing live here too.
│   ├── ledger.go                  the seat CAS: ensureSeat (create-or-book,
│   │                             idempotent per incarnation) and releaseSeat,
│   │                             which retires the emptied ledger inline.
│   ├── fleet.go                  folds ResourceSlices x node labels into
│   │                             pools and ledger seats into occupancy --
│   │                             informer-cache reads only.
│   ├── pod.go                    renders the worker pod from the inline
│   │                             template plus the controller's stamps, and
│   │                             builds the firstAvailable ResourceClaim.
│   ├── workload_controller_test.go  fake-client tests for the glue
│   ├── ledger_test.go            seat-ledger tests: booking, teardown order,
│   │                             the fallback move, the abandon
│   └── stress_test.go            the placement storm: random fleets, churn,
│   │                             concurrent reconciles, invariants per round
│   │                             (build tag stress; `make stress`)
│
└── cmd/manager/main.go           flags/env -> Manager -> run. Boilerplate.

scheduler/hack/
├── kind-smoke.sh                 the pipeline on kind: fake GPUs by default,
│                                 real DRA via env (see header). `make smoke`.
├── stress.sh                     churn many workers against a live cluster
└── retest-on-box.sh              dev helper: sync + test on the GPU box

scheduler/docs/
├── design.md                     the spec. If code and spec disagree, one of
│                                 them is a bug.
└── layout.md                     this file

scheduler/deploy/
├── base/                         namespace + two CRDs + RBAC + Deployment,
│                                 all in openrl-system. Inert until someone
│                                 creates Workload objects.
└── overlays/smoke/               kustomize overlay the smoke test applies:
                                  local image, fake-driver env
```

Reading order for a first pass: `api/v1alpha1/workload_types.go` and
`claimledger_types.go` (the contract), then
`internal/placement/behavior_test.go` (what it promises), then `tiers.go` and
`placement.go` (the decision), then `workload_controller.go`'s `place()` (how
it touches Kubernetes). Everything else is in service of those.

Not in this module: the node-local time-slicer (who is *resident* right now)
is Python, in `src/accel_timeslicer/`, and ships with the FFT line along with
the gateway's `src/server/scheduler_worker_manager.py`, which turns API
requests into Workload objects using the estimator's footprint.
