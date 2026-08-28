package controller

import (
	"context"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	openrlv1alpha1 "github.com/gke-labs/open-rl/scheduler/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

func getLedger(t *testing.T, r *WorkloadReconciler, name string) *openrlv1alpha1.ClaimLedger {
	t.Helper()
	var ledger openrlv1alpha1.ClaimLedger
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: name}, &ledger); err != nil {
		t.Fatalf("get ledger %s: %v", name, err)
	}
	return &ledger
}

// expectGone asserts the object was deleted: retirement is proven by
// absence, not by an empty spec.
func expectGone(t *testing.T, r *WorkloadReconciler, obj client.Object, name string) {
	t.Helper()
	err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: name}, obj)
	if !apierrors.IsNotFound(err) {
		t.Fatalf("%T %q still present: %v", obj, name, err)
	}
}

// Placing a worker writes the same booking in three places: the ledger's
// seat, the worker's status, and the pod's env. All three must agree, or
// the runtime gate has nothing to check.
func TestPlacingRecordsTheSeatEverywhere(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")

	w := getWorker(t, r, "w-a")
	if w.Status.AssignmentID == "" {
		t.Fatal("status carries no assignment")
	}

	ledger := getLedger(t, r, ledgerNameFor(w.Status.ClaimName))
	if ledger.Spec.ClaimName != w.Status.ClaimName {
		t.Errorf("ledger pairs with claim %q, worker holds %q", ledger.Spec.ClaimName, w.Status.ClaimName)
	}
	seat := findSeat(ledger, "w-a")
	if seat == nil {
		t.Fatalf("ledger %s records no seat for w-a: %+v", ledger.Name, ledger.Spec.Seats)
	}
	if seat.AssignmentID != w.Status.AssignmentID {
		t.Errorf("seat assignment %q, status says %q", seat.AssignmentID, w.Status.AssignmentID)
	}

	pod := getPod(t, r, "orw-w-a")
	if pod == nil {
		t.Fatal("no pod for w-a")
	}
	if got := envOf(pod.Spec.Containers[0], claimLedgerEnv); got != ledger.Name {
		t.Errorf("pod %s=%q, want %q", claimLedgerEnv, got, ledger.Name)
	}
	if got := envOf(pod.Spec.Containers[0], assignmentIDEnv); got != w.Status.AssignmentID {
		t.Errorf("pod %s=%q, want %q", assignmentIDEnv, got, w.Status.AssignmentID)
	}
}

// A joiner books a seat beside the founder's on the founder's ledger; it does
// not get a ledger of its own.
func TestJoinerBooksASeatNextToTheFounder(t *testing.T) {
	r := newReconciler(t, append(enabledNode(),
		trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"), trainerWorker("w-c", "model-c"))...)

	runReconcile(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	runReconcile(t, r, "w-b")
	allocateClaim(t, r, claimOf(t, r, "w-b"))
	runReconcile(t, r, "w-c")
	abandoned := ledgerNameFor(claimOf(t, r, "w-c"))
	fallBackToSharing(t, r, "w-c")

	shared := claimOf(t, r, "w-c")
	ledger := getLedger(t, r, ledgerNameFor(shared))
	if len(ledger.Spec.Seats) != 2 {
		t.Fatalf("shared ledger holds %d seats, want 2: %+v", len(ledger.Spec.Seats), ledger.Spec.Seats)
	}
	if findSeat(ledger, "w-c") == nil {
		t.Errorf("no seat for the joiner w-c: %+v", ledger.Spec.Seats)
	}
	// Releasing the founder-less ledger's last seat retires it inline; only
	// its absence proves the seat came back.
	expectGone(t, r, &openrlv1alpha1.ClaimLedger{}, abandoned)
}

// Booking is idempotent per incarnation: ensuring the same worker's seat
// again adopts the recorded assignment instead of minting a second one.
func TestRebookingAdoptsTheRecordedSeat(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	w := getWorker(t, r, "w-a")
	first := w.Status.AssignmentID

	_, seat, err := r.ensureSeat(context.Background(), w.Status.ClaimName, newSeat(w, requestFrom(w)), true)
	if err != nil {
		t.Fatal(err)
	}
	if seat.AssignmentID != first {
		t.Errorf("rebooking replaced assignment %q with %q", first, seat.AssignmentID)
	}
	ledger := getLedger(t, r, ledgerNameFor(w.Status.ClaimName))
	if len(ledger.Spec.Seats) != 1 {
		t.Fatalf("rebooking grew the chart to %d seats: %+v", len(ledger.Spec.Seats), ledger.Spec.Seats)
	}
}

// Deleting a worker frees its seat only once the pod is verifiably gone, and
// the same reconcile then retires the empty ledger and its claim.
func TestTeardownReclaimsTheClaimAndGroupInline(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	claimName := getWorker(t, r, "w-a").Status.ClaimName
	ledgerName := ledgerNameFor(claimName)

	if err := r.Delete(context.Background(), getWorker(t, r, "w-a")); err != nil {
		t.Fatal(err)
	}
	// First pass deletes the pod; second observes it gone, frees the last
	// seat, and reclaims the ledger and claim in the same reconcile -- an
	// allocated claim pins a device, so nothing may linger.
	runReconcile(t, r, "w-a")
	runReconcile(t, r, "w-a")

	expectGone(t, r, &openrlv1alpha1.Workload{}, "w-a")
	expectGone(t, r, &openrlv1alpha1.ClaimLedger{}, ledgerName)
	expectGone(t, r, &resourcev1.ResourceClaim{}, claimName)
}

// A dedicated claim that DRA has not satisfied is abandoned the moment
// kube-scheduler declines its pod -- the verdict, not a timer, is the
// trigger: the worker books a seat on an allocated claim, frees its old
// seat, and its pod is rebuilt against the shared claim.
func TestPendingWorkerFallsBackToSharingOnTheVerdict(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"))...)

	// w-a's claim allocates; w-b's never does -- the device DRA priced it
	// for went to someone else.
	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	settle(t, r, "w-b")
	dedicated := claimOf(t, r, "w-b")

	// Without the verdict the worker waits on its dedicated claim: a pod
	// kube-scheduler has not judged yet is not stuck.
	runReconcile(t, r, "w-b")
	if got := claimOf(t, r, "w-b"); got != dedicated {
		t.Fatalf("w-b moved to %q without kube-scheduler's verdict", got)
	}

	markUnschedulable(t, r, "w-b", time.Now())

	runReconcile(t, r, "w-b")
	shared := claimOf(t, r, "w-a")
	if got := claimOf(t, r, "w-b"); got != shared {
		t.Fatalf("w-b holds %q after the verdict, want the allocated claim %q", got, shared)
	}
	if seat := findSeat(getLedger(t, r, ledgerNameFor(shared)), "w-b"); seat == nil {
		t.Fatal("no seat for w-b on the shared ledger")
	}
	// The abandoned dedicated claim and its ledger are reclaimed in the same
	// reconcile, retracting the scale-out signal the moment we stop wanting it.
	expectGone(t, r, &openrlv1alpha1.ClaimLedger{}, ledgerNameFor(dedicated))
	expectGone(t, r, &resourcev1.ResourceClaim{}, dedicated)

	// The old pod was bound to the dedicated claim; the swap rebuilds it.
	runReconcile(t, r, "w-b")
	pod := getPod(t, r, "orw-w-b")
	if pod == nil {
		t.Fatal("no pod for w-b after the move")
	}
	if got := pod.Labels[LabelClaim]; got != shared {
		t.Errorf("pod bound to %q, want the shared claim %q", got, shared)
	}
}

// An allocated claim whose node refuses the pod past the wedge grace is
// abandoned: pod, seat, and claim all go, and the worker starts its
// lifecycle over.
func TestAbandonsAWedgedAllocatedClaim(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	wedged := claimOf(t, r, "w-a")
	allocateClaim(t, r, wedged)
	// Host memory vanished between pod incarnations (or the Spot node did):
	// kube-scheduler has refused the pod since well past the wedge grace.
	markUnschedulable(t, r, "w-a", time.Now().Add(-2*wedgeGracePeriod))

	runReconcile(t, r, "w-a")

	if got := getWorker(t, r, "w-a").Status.ClaimName; got != "" {
		t.Fatalf("worker still holds %q, want the wedged claim abandoned", got)
	}
	if pod := getPod(t, r, "orw-w-a"); pod != nil {
		t.Fatal("the wedged pod survived")
	}
	expectGone(t, r, &resourcev1.ResourceClaim{}, wedged)
	expectGone(t, r, &openrlv1alpha1.ClaimLedger{}, ledgerNameFor(wedged))

	// The lifecycle restarts cleanly: fresh claim, fresh seat.
	settle(t, r, "w-a")
	fresh := claimOf(t, r, "w-a")
	if findSeat(getLedger(t, r, ledgerNameFor(fresh)), "w-a") == nil {
		t.Fatal("no seat after the restart")
	}
}

// A claim deleted out from under its worker leaves the ledger holding a seat
// nothing tracks. Re-placing must release that booking, not adopt it: claim
// names are deterministic, so the proof is a fresh assignment ID on a fresh
// ledger, never the stale seat carried over.
func TestVanishedClaimReleasesItsSeat(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	claimName := claimOf(t, r, "w-a")
	first := getWorker(t, r, "w-a").Status.AssignmentID
	var claim resourcev1.ResourceClaim
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: claimName}, &claim); err != nil {
		t.Fatal(err)
	}
	if err := r.Delete(context.Background(), &claim); err != nil {
		t.Fatal(err)
	}

	settle(t, r, "w-a")
	if got := getWorker(t, r, "w-a").Status.AssignmentID; got == first {
		t.Fatal("the stale seat was adopted; want it released and re-booked")
	}
	ledger := getLedger(t, r, ledgerNameFor(claimOf(t, r, "w-a")))
	if len(ledger.Spec.Seats) != 1 {
		t.Fatalf("ledger holds %d seats after the re-place, want 1: %+v", len(ledger.Spec.Seats), ledger.Spec.Seats)
	}
}

// One seat per worker: a booking stranded on another ledger -- a status write
// lost after a seat move -- is released by the holder's next reconcile.
func TestStraySeatIsReleasedOnReconcile(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"))...)

	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	settle(t, r, "w-b")
	allocateClaim(t, r, claimOf(t, r, "w-b"))

	// Strand a seat: book w-a onto w-b's ledger behind its status's back.
	wa := getWorker(t, r, "w-a")
	if _, _, err := r.ensureSeat(context.Background(), claimOf(t, r, "w-b"), newSeat(wa, requestFrom(wa)), false); err != nil {
		t.Fatal(err)
	}

	settle(t, r, "w-a")
	ledger := getLedger(t, r, ledgerNameFor(claimOf(t, r, "w-b")))
	if findSeat(ledger, "w-a") != nil {
		t.Errorf("stray seat for w-a survived its reconcile: %+v", ledger.Spec.Seats)
	}
	if findSeat(ledger, "w-b") == nil {
		t.Error("the release took the rightful tenant's seat with it")
	}
	if findSeat(getLedger(t, r, ledgerNameFor(claimOf(t, r, "w-a"))), "w-a") == nil {
		t.Error("w-a lost its own seat")
	}
}

// A retiring claim -- ledger already deleted, claim not yet -- must never be
// re-seated: joins append to existing ledgers only, so the ledger's absence is
// the tombstone, and no retiring state is needed between empty and gone.
func TestJoinNeverResurrectsARetiringGroup(t *testing.T) {
	orphan := &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: "claim-orphan", Namespace: testNamespace,
			Labels: map[string]string{LabelManaged: "true"},
		},
		Status: resourcev1.ResourceClaimStatus{
			Allocation: &resourcev1.AllocationResult{
				Devices: resourcev1.DeviceAllocationResult{
					Results: []resourcev1.DeviceRequestAllocationResult{{
						Request: podClaimName, Driver: testDriver, Pool: testNode, Device: "gpu-0",
					}},
				},
				NodeSelector: &corev1.NodeSelector{NodeSelectorTerms: []corev1.NodeSelectorTerm{{
					MatchFields: []corev1.NodeSelectorRequirement{{
						Key: "metadata.name", Operator: corev1.NodeSelectorOpIn, Values: []string{testNode},
					}},
				}}},
			},
		},
	}
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"), orphan)...)
	r.PlacementStrategy = placement.StrategyBinPack

	// Binpack would love the orphan: allocated, empty, fewest owners. The
	// missing ledger must turn the join away and force a dedicated claim.
	settle(t, r, "w-a")
	expectGone(t, r, &openrlv1alpha1.ClaimLedger{}, ledgerNameFor("claim-orphan"))
	if got := claimOf(t, r, "w-a"); got == "claim-orphan" {
		t.Fatal("worker seated on a retiring claim")
	}
}
