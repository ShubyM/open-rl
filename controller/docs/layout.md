# Where everything is

The short version: two structs define the API, three functions make the
decision, one function drives Kubernetes, and everything else is glue you can
read once and trust.

```
controller/
├── api/v1alpha1/
│   ├── openrlworker_types.go     the contract: Spec (what the caller asks) and
│   │                             Status (what was decided). The +kubebuilder
│   │                             comments generate the CRD and its validation;
│   │                             doc comments here become `kubectl explain` text.
│   ├── groupversion_info.go      scheme registration boilerplate
│   └── zz_generated.deepcopy.go  generated; never edit (make generate)
│
├── internal/placement/           THE DECISION. Pure functions, no Kubernetes.
│   ├── placement.go              Decide = ChoosePool (spread onto free devices)
│   │                             then SelectClaim (share under contention).
│   │                             Claim/Node/Fleet are plain structs; the three
│   │                             admission checks and the parked-memory formula
│   │                             live here and nowhere else.
│   ├── behavior_test.go          the end-to-end behaviors, written as arrivals
│   │                             and departures with the estimator's real tier
│   │                             figures. Start reading tests here.
│   └── placement_test.go         unit tests for the arithmetic, tie-breaks,
│                                 and the pending-claim reservation rules
│
├── internal/controller/          THE KUBERNETES GLUE.
│   ├── openrlworker_controller.go  Reconcile -> place(): one decision tree per
│   │                             pass (has claim? ensure pod : decide, create,
│   │                             or mark Pending). Also the watch wiring
│   │                             (SetupWithManager) and status writing.
│   ├── fleet.go                  reads the world into placement's Fleet:
│   │                             ResourceSlices x node labels = pools, managed
│   │                             claims + worker statuses = occupancy
│   ├── pod.go                    renders the worker pod: operator template +
│   │                             the controller's stamps (claim, affinity,
│   │                             time-slice env) and builds the ResourceClaim
│   ├── reclaim.go                the sweep that deletes claims nobody uses
│   └── openrlworker_controller_test.go  fake-client tests for the glue
│
├── cmd/manager/main.go           flags/env -> Manager -> run. Boilerplate.
│
├── hack/
│   ├── kind-smoke.sh             the pipeline on kind: fake GPUs by default,
│   │                             real DRA via env (see header). `make smoke`.
│   ├── stress.sh                 churn many workers against a live cluster and
│   │                             assert the seat arithmetic every round
│   └── retest-on-box.sh          dev helper: sync + test on the GPU box
│
└── docs/
    ├── design.md                 the spec. If code and spec disagree, one of
    │                             them is a bug.
    └── layout.md                 this file

k8s/deploy/
├── scheduler/                    CRD + RBAC + Deployment. Inert until someone
│                                 creates OpenRLWorker objects.
└── scheduler-smoke/              kustomize overlay the smoke test applies:
                                  local image, fake-driver env, sleep templates
```

Reading order for a first pass: `api/v1alpha1/openrlworker_types.go` (the
contract), then `internal/placement/behavior_test.go` (what it promises), then
`placement.go`'s `Decide`/`SelectClaim`/`ChoosePool` (how), then
`openrlworker_controller.go`'s `place()` (how it touches Kubernetes). That is
~500 lines and everything else is in service of it.

Not in this module: the node-local time-slicer (who is *resident* right now)
is Python, in `src/accel_timeslicer/`, and ships with the FFT line along with
the gateway's `src/server/scheduler_worker_manager.py`, which turns API
requests into OpenRLWorker objects using the estimator's memory figure.
