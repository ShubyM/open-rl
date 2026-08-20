package controller

import (
	"context"
	"fmt"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/controller/internal/placement"
)

// bookAttempts bounds the CAS retry loop; past it the reconcile requeues.
const bookAttempts = 5

// errSeatUnavailable: the group filled between the placement decision and
// the write. Callers treat it as "no placement this pass", not a failure.
var errSeatUnavailable = fmt.Errorf("no seat available on the chosen group")

// groupNameFor derives the group name from the claim name; it is never
// stored anywhere.
func groupNameFor(claimName string) string {
	return "group-" + strings.TrimPrefix(claimName, "claim-")
}

// newSeat mints a fresh assignment ID; owner and host figures come from the
// Request so they cannot drift from placement's.
func newSeat(worker *openrlv1alpha1.OpenRLWorkload, request placement.Request) openrlv1alpha1.Seat {
	return openrlv1alpha1.Seat{
		Workload:     worker.Name,
		WorkloadUID:  string(worker.UID),
		AssignmentID: string(uuid.NewUUID()),
		Owner:        request.OwnerKey(),
		HostRequest:  *resource.NewQuantity(request.HostRequestBytes, resource.BinarySI),
	}
}

// ensureSeat books the seat on the claim's group via one CAS loop, creating
// the group when absent. A seat held by the same incarnation is adopted, a
// predecessor's is replaced, and a full group returns errSeatUnavailable --
// resourceVersion arbitrates, so two bookings cannot both win the last seat.
func (r *OpenRLWorkloadReconciler) ensureSeat(ctx context.Context, claimName string, maxSeats int, seat openrlv1alpha1.Seat) (*openrlv1alpha1.OpenRLClaimGroup, *openrlv1alpha1.Seat, error) {
	groupName := groupNameFor(claimName)
	for attempt := 0; attempt < bookAttempts; attempt++ {
		var group openrlv1alpha1.OpenRLClaimGroup
		err := r.fleetReader().Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: groupName}, &group)
		if apierrors.IsNotFound(err) {
			fresh := &openrlv1alpha1.OpenRLClaimGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      groupName,
					Namespace: r.Namespace,
					Labels:    map[string]string{LabelManaged: "true", LabelClaim: claimName},
				},
				Spec: openrlv1alpha1.OpenRLClaimGroupSpec{
					ClaimName: claimName,
					MaxSeats:  int32(maxSeats),
					Seats:     []openrlv1alpha1.Seat{seat},
				},
			}
			if err := r.Create(ctx, fresh); err != nil {
				if apierrors.IsAlreadyExists(err) {
					continue // Lost the create; book on the winner's copy.
				}
				return nil, nil, fmt.Errorf("create group %s: %w", groupName, err)
			}
			return fresh, &fresh.Spec.Seats[0], nil
		}
		if err != nil {
			return nil, nil, fmt.Errorf("read group %s: %w", groupName, err)
		}

		// Stamp the ceiling once the first booking knows the node's policy.
		if group.Spec.MaxSeats == 0 && maxSeats > 0 {
			group.Spec.MaxSeats = int32(maxSeats)
		}

		if existing := findSeat(&group, seat.Workload); existing != nil {
			if existing.WorkloadUID == seat.WorkloadUID {
				return &group, existing, nil // already booked; adopt it
			}
			*existing = seat // a predecessor's seat under our name: replace it
		} else {
			if group.Spec.MaxSeats > 0 && len(group.Spec.Seats) >= int(group.Spec.MaxSeats) {
				return nil, nil, errSeatUnavailable
			}
			group.Spec.Seats = append(group.Spec.Seats, seat)
		}

		err = r.Update(ctx, &group)
		if err == nil {
			return &group, findSeat(&group, seat.Workload), nil
		}
		if !apierrors.IsConflict(err) {
			return nil, nil, fmt.Errorf("book seat on group %s: %w", groupName, err)
		}
		// Conflict: someone else booked first. Re-read and re-check.
	}
	return nil, nil, errSeatUnavailable
}

// releaseSeat removes one workload incarnation's seat. A missing group or
// missing seat is success: the seat is gone.
func (r *OpenRLWorkloadReconciler) releaseSeat(ctx context.Context, groupName, workloadName, workloadUID string) error {
	for attempt := 0; attempt < bookAttempts; attempt++ {
		var group openrlv1alpha1.OpenRLClaimGroup
		if err := r.fleetReader().Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: groupName}, &group); err != nil {
			if apierrors.IsNotFound(err) {
				return nil
			}
			return fmt.Errorf("read group %s: %w", groupName, err)
		}

		kept := group.Spec.Seats[:0]
		for _, s := range group.Spec.Seats {
			if s.Workload == workloadName && s.WorkloadUID == workloadUID {
				continue
			}
			kept = append(kept, s)
		}
		if len(kept) == len(group.Spec.Seats) {
			return nil
		}
		group.Spec.Seats = kept

		err := r.Update(ctx, &group)
		if err == nil {
			return nil
		}
		if !apierrors.IsConflict(err) {
			return fmt.Errorf("release seat on group %s: %w", groupName, err)
		}
	}
	return fmt.Errorf("release seat on group %s: conflict retries exhausted", groupName)
}

// findSeat returns the seat held under a workload name, or nil.
func findSeat(group *openrlv1alpha1.OpenRLClaimGroup, workload string) *openrlv1alpha1.Seat {
	for i := range group.Spec.Seats {
		if group.Spec.Seats[i].Workload == workload {
			return &group.Spec.Seats[i]
		}
	}
	return nil
}

// maxSeatsFor is the operator's per-claim ceiling for the node a claim was
// allocated to, or 0 while the node is unknown -- the ceiling is stamped
// onto the group by the first booking that knows it.
func maxSeatsFor(fleet *placement.Fleet, claimName string) int {
	claim, ok := fleet.Claims[claimName]
	if !ok || claim.Node == "" {
		return 0
	}
	if node := fleet.Nodes[claim.Node]; node != nil {
		return node.MaxWorkersPerClaim
	}
	return 0
}
