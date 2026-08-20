package controller

import (
	"context"
	"testing"
	"time"

	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
)

func getGroup(t *testing.T, r *OpenRLWorkloadReconciler, name string) *openrlv1alpha1.OpenRLClaimGroup {
	t.Helper()
	var group openrlv1alpha1.OpenRLClaimGroup
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: name}, &group); err != nil {
		t.Fatalf("get group %s: %v", name, err)
	}
	return &group
}

// Placing a worker writes the same booking in three places: the group's
// seat, the worker's status, and the pod's env. All three must agree, or
// the runtime gate has nothing to check.
func TestPlacingRecordsTheSeatEverywhere(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")

	w := getWorker(t, r, "w-a")
	if w.Status.AssignmentID == "" {
		t.Fatal("status carries no assignment")
	}

	group := getGroup(t, r, groupNameFor(w.Status.ClaimName))
	if group.Spec.ClaimName != w.Status.ClaimName {
		t.Errorf("group pairs with claim %q, worker holds %q", group.Spec.ClaimName, w.Status.ClaimName)
	}
	seat := findSeat(group, "w-a")
	if seat == nil {
		t.Fatalf("group %s records no seat for w-a: %+v", group.Name, group.Spec.Seats)
	}
	if seat.AssignmentID != w.Status.AssignmentID {
		t.Errorf("seat assignment %q, status says %q", seat.AssignmentID, w.Status.AssignmentID)
	}

	pod := getPod(t, r, "orw-w-a")
	if pod == nil {
		t.Fatal("no pod for w-a")
	}
	if got := envOf(pod.Spec.Containers[0], claimGroupEnv); got != group.Name {
		t.Errorf("pod %s=%q, want %q", claimGroupEnv, got, group.Name)
	}
	if got := envOf(pod.Spec.Containers[0], assignmentIDEnv); got != w.Status.AssignmentID {
		t.Errorf("pod %s=%q, want %q", assignmentIDEnv, got, w.Status.AssignmentID)
	}
}

// A joiner books a seat beside the founder's on the founder's group; it does
// not get a group of its own.
func TestJoinerBooksASeatNextToTheFounder(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4),
		trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"), trainerWorker("w-c", "model-c"))...)

	runReconcile(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	runReconcile(t, r, "w-b")
	allocateClaim(t, r, claimOf(t, r, "w-b"))
	runReconcile(t, r, "w-c")
	abandoned := groupNameFor(claimOf(t, r, "w-c"))
	shareAfterGrace(t, r, "w-c")

	shared := claimOf(t, r, "w-c")
	group := getGroup(t, r, groupNameFor(shared))
	if len(group.Spec.Seats) != 2 {
		t.Fatalf("shared group holds %d seats, want 2: %+v", len(group.Spec.Seats), group.Spec.Seats)
	}
	if findSeat(group, "w-c") == nil {
		t.Errorf("no seat for the joiner w-c: %+v", group.Spec.Seats)
	}
	if seats := getGroup(t, r, abandoned).Spec.Seats; len(seats) != 0 {
		t.Errorf("the abandoned dedicated group still seats %+v", seats)
	}
}

// The group's MaxSeats is enforced at the write, not just in the placement
// snapshot: a booking that arrives at a full group is turned away.
func TestAFullGroupTurnsAJoinerAway(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4),
		&openrlv1alpha1.OpenRLClaimGroup{
			ObjectMeta: metav1.ObjectMeta{Name: "group-full", Namespace: testNamespace},
			Spec: openrlv1alpha1.OpenRLClaimGroupSpec{
				ClaimName: "claim-full",
				MaxSeats:  1,
				Seats:     []openrlv1alpha1.Seat{{Workload: "w-first", AssignmentID: "seat-1"}},
			},
		})...)

	_, _, err := r.ensureSeat(context.Background(), "claim-full", 1, openrlv1alpha1.Seat{Workload: "w-late", AssignmentID: "seat-2"})
	if err != errSeatUnavailable {
		t.Fatalf("booking on a full group returned %v, want errSeatUnavailable", err)
	}

	group := getGroup(t, r, "group-full")
	if len(group.Spec.Seats) != 1 || group.Spec.Seats[0].Workload != "w-first" {
		t.Errorf("the full group changed: %+v", group.Spec.Seats)
	}
}

// Booking is idempotent per incarnation: ensuring the same worker's seat
// again adopts the recorded assignment instead of minting a second one.
func TestRebookingAdoptsTheRecordedSeat(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	w := getWorker(t, r, "w-a")
	first := w.Status.AssignmentID

	_, seat, err := r.ensureSeat(context.Background(), w.Status.ClaimName, 4, newSeat(w, requestFrom(w)))
	if err != nil {
		t.Fatal(err)
	}
	if seat.AssignmentID != first {
		t.Errorf("rebooking replaced assignment %q with %q", first, seat.AssignmentID)
	}
	group := getGroup(t, r, groupNameFor(w.Status.ClaimName))
	if len(group.Spec.Seats) != 1 {
		t.Fatalf("rebooking grew the chart to %d seats: %+v", len(group.Spec.Seats), group.Spec.Seats)
	}
}

// Deleting a worker frees its seat only once the pod is verifiably gone, and
// the sweep then retires the empty chart after its claim.
func TestTeardownFreesTheSeatThenTheSweepRetiresTheGroup(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	claimName := getWorker(t, r, "w-a").Status.ClaimName
	groupName := groupNameFor(claimName)

	if err := r.Delete(context.Background(), getWorker(t, r, "w-a")); err != nil {
		t.Fatal(err)
	}
	// First pass deletes the pod; second observes it gone and frees the seat.
	runReconcile(t, r, "w-a")
	runReconcile(t, r, "w-a")

	var gone openrlv1alpha1.OpenRLWorkload
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: "w-a"}, &gone); !apierrors.IsNotFound(err) {
		t.Fatalf("worker still present: %v", err)
	}
	group := getGroup(t, r, groupName)
	if len(group.Spec.Seats) != 0 {
		t.Fatalf("seats survive their worker: %+v", group.Spec.Seats)
	}

	// The sweep takes the claim first (nothing books it), then the group on a
	// later pass -- but not before the claim is genuinely gone.
	backdate(t, r, group)
	if err := r.reclaimIdleClaims(context.Background()); err != nil {
		t.Fatal(err)
	}
	var claim resourcev1.ResourceClaim
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: claimName}, &claim); !apierrors.IsNotFound(err) {
		t.Fatalf("idle claim survived the sweep: %v", err)
	}
	if err := r.reclaimIdleClaims(context.Background()); err != nil {
		t.Fatal(err)
	}
	var g openrlv1alpha1.OpenRLClaimGroup
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: groupName}, &g); !apierrors.IsNotFound(err) {
		t.Fatalf("empty group survived the sweep after its claim: %v", err)
	}
}

// A seat is a booking even when the worker's status write never landed: the
// sweep must not reclaim a claim whose group still seats someone.
func TestSeatsKeepClaimsAliveThroughTheSweep(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	w := getWorker(t, r, "w-a")

	// Erase the status booking, leaving only the seat.
	w.Status.ClaimName, w.Status.AssignmentID = "", ""
	if err := r.Status().Update(context.Background(), w); err != nil {
		t.Fatal(err)
	}
	// Delete the pod too: the seat alone must carry the booking.
	if pod := getPod(t, r, "orw-w-a"); pod != nil {
		if err := r.Delete(context.Background(), pod); err != nil {
			t.Fatal(err)
		}
	}

	var claims resourcev1.ResourceClaimList
	if err := r.List(context.Background(), &claims); err != nil {
		t.Fatal(err)
	}
	for i := range claims.Items {
		backdate(t, r, &claims.Items[i])
	}

	if err := r.reclaimIdleClaims(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := r.List(context.Background(), &claims); err != nil {
		t.Fatal(err)
	}
	if len(claims.Items) != 1 {
		t.Fatalf("%d claims survive, want 1: the seat alone should hold it", len(claims.Items))
	}
}

// A dedicated claim that DRA never satisfies is abandoned after the
// scale-out grace period: the worker books a seat on an allocated claim,
// frees its old seat, and its pod is rebuilt against the shared claim.
func TestPendingWorkerFallsBackToSharingAfterTheGrace(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4), trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"))...)

	// w-a's claim allocates; w-b's never does -- the device DRA priced it
	// for went to someone else, or the autoscaler node never came.
	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	settle(t, r, "w-b")
	dedicated := claimOf(t, r, "w-b")

	// Within the grace period the worker waits on its dedicated claim.
	runReconcile(t, r, "w-b")
	if got := claimOf(t, r, "w-b"); got != dedicated {
		t.Fatalf("w-b moved to %q before the grace period expired", got)
	}

	var claim resourcev1.ResourceClaim
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: dedicated}, &claim); err != nil {
		t.Fatal(err)
	}
	claim.CreationTimestamp = metav1.NewTime(time.Now().Add(-2 * r.ScaleOutGracePeriod))
	if err := r.Update(context.Background(), &claim); err != nil {
		t.Fatal(err)
	}
	markUnschedulable(t, r, "w-b", time.Now())

	runReconcile(t, r, "w-b")
	shared := claimOf(t, r, "w-a")
	if got := claimOf(t, r, "w-b"); got != shared {
		t.Fatalf("w-b holds %q after the grace, want the allocated claim %q", got, shared)
	}
	if seat := findSeat(getGroup(t, r, groupNameFor(shared)), "w-b"); seat == nil {
		t.Fatal("no seat for w-b on the shared group")
	}
	if seats := getGroup(t, r, groupNameFor(dedicated)).Spec.Seats; len(seats) != 0 {
		t.Fatalf("the abandoned group still seats %+v", seats)
	}

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

// An allocated claim whose node refuses the pod past the grace is abandoned:
// pod, seat, and claim all go, and the worker starts its lifecycle over.
func TestAbandonsAWedgedAllocatedClaim(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	wedged := claimOf(t, r, "w-a")
	allocateClaim(t, r, wedged)
	// Host memory vanished between pod incarnations (or the Spot node did):
	// kube-scheduler has refused the pod since well past the grace.
	markUnschedulable(t, r, "w-a", time.Now().Add(-2*r.ScaleOutGracePeriod))

	runReconcile(t, r, "w-a")

	if got := getWorker(t, r, "w-a").Status.ClaimName; got != "" {
		t.Fatalf("worker still holds %q, want the wedged claim abandoned", got)
	}
	if pod := getPod(t, r, "orw-w-a"); pod != nil {
		t.Fatal("the wedged pod survived")
	}
	var claim resourcev1.ResourceClaim
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: wedged}, &claim); !apierrors.IsNotFound(err) {
		t.Fatalf("wedged claim survived: %v", err)
	}
	if seats := getGroup(t, r, groupNameFor(wedged)).Spec.Seats; len(seats) != 0 {
		t.Fatalf("seat survived the abandon: %+v", seats)
	}

	// The lifecycle restarts cleanly: fresh claim, fresh seat.
	settle(t, r, "w-a")
	fresh := claimOf(t, r, "w-a")
	if findSeat(getGroup(t, r, groupNameFor(fresh)), "w-a") == nil {
		t.Fatal("no seat after the restart")
	}
}

// backdate ages an object past the reclaim grace period. The fake client
// preserves creationTimestamp on update.
func backdate(t *testing.T, r *OpenRLWorkloadReconciler, obj client.Object) {
	t.Helper()
	obj.SetCreationTimestamp(metav1.NewTime(time.Now().Add(-2 * claimGracePeriod)))
	if err := r.Update(context.Background(), obj); err != nil {
		t.Fatal(err)
	}
}
