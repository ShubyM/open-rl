package controller

import (
	"context"
	"time"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
)

// claimGracePeriod is how long a newly cut claim is safe from the reclaim
// sweep. A claim is created before the worker status that names it, so
// without a grace period the sweep could delete a claim out from under a
// worker that is still being placed.
const claimGracePeriod = 2 * time.Minute

// managedClaims restricts the ResourceClaim watch to the ones this controller
// created, so unrelated DRA traffic does not wake every worker.
func managedClaims() predicate.Predicate {
	return predicate.NewPredicateFuncs(func(obj client.Object) bool {
		return obj.GetLabels()[LabelManaged] == "true"
	})
}

// runReclaim sweeps idle claims until the manager stops.
func (r *OpenRLWorkloadReconciler) runReclaim(ctx context.Context) error {
	ticker := time.NewTicker(r.ReclaimInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			if err := r.reclaimIdleClaims(ctx); err != nil {
				log.FromContext(ctx).Error(err, "reclaim sweep failed")
			}
		}
	}
}

// reclaimIdleClaims deletes managed claims nothing backs. Shared claims
// carry no owner reference, so this sweep must exist. Stays of execution: a
// seated group, a live pod, a DRA reservation, or youth.
func (r *OpenRLWorkloadReconciler) reclaimIdleClaims(ctx context.Context) error {
	logger := log.FromContext(ctx)

	var claims resourcev1.ResourceClaimList
	if err := r.List(ctx, &claims, client.InNamespace(r.Namespace), client.MatchingLabels{LabelManaged: "true"}); err != nil {
		return err
	}

	spokenFor := map[string]bool{}

	// A group with seats keeps its claim: seats are the booking record.
	var groups openrlv1alpha1.OpenRLClaimGroupList
	if err := r.fleetReader().List(ctx, &groups, client.InNamespace(r.Namespace), client.MatchingLabels{LabelManaged: "true"}); err != nil {
		return err
	}
	for i := range groups.Items {
		if len(groups.Items[i].Spec.Seats) > 0 {
			spokenFor[groups.Items[i].Spec.ClaimName] = true
		}
	}

	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(r.Namespace), client.HasLabels{LabelClaim}); err != nil {
		return err
	}
	for i := range pods.Items {
		pod := &pods.Items[i]
		if pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
			continue
		}
		spokenFor[pod.Labels[LabelClaim]] = true
	}

	liveClaims := map[string]bool{}
	for i := range claims.Items {
		claim := &claims.Items[i]
		liveClaims[claim.Name] = true
		if spokenFor[claim.Name] || len(claim.Status.ReservedFor) > 0 ||
			time.Since(claim.CreationTimestamp.Time) < claimGracePeriod {
			continue
		}
		logger.Info("reclaiming idle claim", "claim", claim.Name)
		if err := r.Delete(ctx, claim); err != nil && !apierrors.IsNotFound(err) {
			logger.Error(err, "failed to delete idle claim", "claim", claim.Name)
		}
	}

	// An empty group follows its claim out, one sweep later.
	for i := range groups.Items {
		group := &groups.Items[i]
		if len(group.Spec.Seats) > 0 || liveClaims[group.Spec.ClaimName] ||
			time.Since(group.CreationTimestamp.Time) < claimGracePeriod {
			continue
		}
		logger.Info("reclaiming empty claim group", "group", group.Name)
		if err := r.Delete(ctx, group); err != nil && !apierrors.IsNotFound(err) {
			logger.Error(err, "failed to delete empty claim group", "group", group.Name)
		}
	}
	return nil
}
