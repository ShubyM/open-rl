package controller

import (
	"context"
	"fmt"
	"strings"

	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"sigs.k8s.io/controller-runtime/pkg/client"

	openrlv1alpha1 "github.com/gke-labs/open-rl/scheduler/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

// bookAttempts bounds the CAS retry loop; past it the reconcile requeues.
const bookAttempts = 5

// errBookingContended: the booking lost every CAS retry to concurrent
// writers. Callers treat it as "no placement this pass", not a failure.
var errBookingContended = fmt.Errorf("seat booking lost every CAS retry on the chosen ledger")

// ledgerNameFor derives the ClaimLedger name from the claim name; it is never
// stored anywhere.
func ledgerNameFor(claimName string) string {
	return "ledger-" + strings.TrimPrefix(claimName, "claim-")
}

// newSeat mints a fresh assignment ID; owner and host figures come from the
// Request so they cannot drift from placement's.
func newSeat(worker *openrlv1alpha1.Workload, request placement.Request) openrlv1alpha1.Seat {
	return openrlv1alpha1.Seat{
		Workload:     worker.Name,
		WorkloadUID:  string(worker.UID),
		AssignmentID: string(uuid.NewUUID()),
		Owner:        request.OwnerKey(),
		HostRequest:  *resource.NewQuantity(request.HostRequestBytes, resource.BinarySI),
	}
}

// ensureSeat books the seat on the claim's ClaimLedger via one CAS loop. A seat
// held by the same incarnation is adopted and a predecessor's is replaced.
// There is no seat ceiling: host memory is the limit on parked workers,
// checked advisorily at selection and enforced for real by kube-scheduler
// against the pods' memory requests.
//
// createMissing is true only for a worker's own claim: the founder books its
// ClaimLedger before the claim exists, and a re-book heals the ledger under a
// running pod. A join must never create -- retirement deletes the ClaimLedger
// before the claim, so for a joiner the ClaimLedger's absence is the tombstone of
// a dying claim, and booking would resurrect it.
func (r *WorkloadReconciler) ensureSeat(ctx context.Context, claimName string, seat openrlv1alpha1.Seat, createMissing bool) (*openrlv1alpha1.ClaimLedger, *openrlv1alpha1.Seat, error) {
	ledgerName := ledgerNameFor(claimName)
	for attempt := 0; attempt < bookAttempts; attempt++ {
		var claimLedger openrlv1alpha1.ClaimLedger
		err := r.fleetReader().Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: ledgerName}, &claimLedger)
		if apierrors.IsNotFound(err) {
			if !createMissing {
				return nil, nil, errBookingContended
			}
			fresh := &openrlv1alpha1.ClaimLedger{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ledgerName,
					Namespace: r.Namespace,
					Labels:    map[string]string{LabelManaged: "true", LabelClaim: claimName},
				},
				Spec: openrlv1alpha1.ClaimLedgerSpec{
					ClaimName: claimName,
					Seats:     []openrlv1alpha1.Seat{seat},
				},
			}
			if err := r.Create(ctx, fresh); err != nil {
				if apierrors.IsAlreadyExists(err) {
					continue // Lost the create; book on the winner's copy.
				}
				return nil, nil, fmt.Errorf("create ledger %s: %w", ledgerName, err)
			}
			return fresh, &fresh.Spec.Seats[0], nil
		}
		if err != nil {
			return nil, nil, fmt.Errorf("read ledger %s: %w", ledgerName, err)
		}

		if existing := findSeat(&claimLedger, seat.Workload); existing != nil {
			if existing.WorkloadUID == seat.WorkloadUID {
				return &claimLedger, existing, nil // already booked; adopt it
			}
			*existing = seat // a predecessor's seat under our name: replace it
		} else {
			claimLedger.Spec.Seats = append(claimLedger.Spec.Seats, seat)
		}

		err = r.Update(ctx, &claimLedger)
		if err == nil {
			return &claimLedger, findSeat(&claimLedger, seat.Workload), nil
		}
		if !apierrors.IsConflict(err) {
			return nil, nil, fmt.Errorf("book seat on ledger %s: %w", ledgerName, err)
		}
		// Conflict: someone else booked first. Re-read and re-check.
	}
	return nil, nil, errBookingContended
}

// releaseSeatAndReclaim removes one workload incarnation's seat via one CAS
// loop. A missing ClaimLedger or missing seat is success: the seat is gone.
// Releasing the last seat retires the ClaimLedger itself -- deleted against the
// resourceVersion that showed it empty, so a joiner booking concurrently
// bumps the version and keeps the ClaimLedger -- and deletes the claim in the same
// pass: an allocated claim pins its device with or without pods, so a
// seatless one must not idle a GPU waiting for the sweep. The sweep remains
// the backstop for a controller that dies between the two deletions.
func (r *WorkloadReconciler) releaseSeatAndReclaim(ctx context.Context, claimName, workloadName, workloadUID string) error {
	ledgerName := ledgerNameFor(claimName)
	for attempt := 0; attempt < bookAttempts; attempt++ {
		var claimLedger openrlv1alpha1.ClaimLedger
		if err := r.fleetReader().Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: ledgerName}, &claimLedger); err != nil {
			if apierrors.IsNotFound(err) {
				return r.deleteClaim(ctx, claimName)
			}
			return fmt.Errorf("read ledger %s: %w", ledgerName, err)
		}

		kept := claimLedger.Spec.Seats[:0]
		for _, seat := range claimLedger.Spec.Seats {
			if seat.Workload == workloadName && seat.WorkloadUID == workloadUID {
				continue
			}
			kept = append(kept, seat)
		}
		if len(kept) == 0 {
			err := r.Delete(ctx, &claimLedger, client.Preconditions{ResourceVersion: &claimLedger.ResourceVersion})
			if apierrors.IsConflict(err) {
				continue // someone booked between the read and the delete
			}
			if err != nil && !apierrors.IsNotFound(err) {
				return fmt.Errorf("retire ledger %s: %w", ledgerName, err)
			}
			return r.deleteClaim(ctx, claimName)
		}
		if len(kept) == len(claimLedger.Spec.Seats) {
			return nil
		}
		claimLedger.Spec.Seats = kept

		err := r.Update(ctx, &claimLedger)
		if err == nil {
			return nil
		}
		if !apierrors.IsConflict(err) {
			return fmt.Errorf("release seat on ledger %s: %w", ledgerName, err)
		}
	}
	return fmt.Errorf("release seat on ledger %s: conflict retries exhausted", ledgerName)
}

// deleteClaim removes a seatless claim; already gone is success.
func (r *WorkloadReconciler) deleteClaim(ctx context.Context, claimName string) error {
	claim := &resourcev1.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: claimName, Namespace: r.Namespace}}
	if err := r.Delete(ctx, claim); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("reclaim claim %s: %w", claimName, err)
	}
	return nil
}

// findSeat returns the seat held under a workload name, or nil.
func findSeat(claimLedger *openrlv1alpha1.ClaimLedger, workload string) *openrlv1alpha1.Seat {
	for i := range claimLedger.Spec.Seats {
		if claimLedger.Spec.Seats[i].Workload == workload {
			return &claimLedger.Spec.Seats[i]
		}
	}
	return nil
}
