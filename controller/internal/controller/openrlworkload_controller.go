package controller

import (
	"context"
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/controller/internal/placement"
)

// OpenRLWorkloadReconciler turns OpenRLWorkload requests into ResourceClaims
// and pods; the decision lives in internal/placement. Seat booking is
// CAS-arbitrated on the OpenRLClaimGroup, so reconciles run concurrently.
type OpenRLWorkloadReconciler struct {
	client.Client
	Recorder record.EventRecorder

	// Namespace is where workers, claims and pods live.
	Namespace string
	// DeviceClass is the DRA DeviceClass generated claims request.
	DeviceClass string
	// DeviceDriver is the driver publishing the ResourceSlices, and the CEL
	// domain its capacities live under. Distinct from DeviceClass in principle,
	// identical for NVIDIA's driver.
	DeviceDriver string
	// RetryInterval is how often a worker that could not be placed is retried.
	RetryInterval time.Duration
	// PlacementTimeout is how long a worker may go unplaced before the request
	// is declared unsatisfiable. Without it an impossible request waits
	// forever, indistinguishable from one that is merely queued.
	PlacementTimeout time.Duration
	// ReclaimInterval is how often idle claims are swept.
	ReclaimInterval time.Duration
	// ScaleOutGracePeriod is how long a dedicated claim may wait for DRA or
	// the autoscaler before its worker falls back to sharing. 0 never shares.
	ScaleOutGracePeriod time.Duration
	// MaxConcurrentReconciles is how many workers place at once.
	MaxConcurrentReconciles int

	// reader bypasses the informer cache for read-modify-write paths only:
	// the seat CAS, claim adoption, and the reclaim sweep. Nil (in tests)
	// falls back to the regular client.
	reader client.Reader
}

// fleetReader is the consistent reader for fleet state.
func (r *OpenRLWorkloadReconciler) fleetReader() client.Reader {
	if r.reader != nil {
		return r.reader
	}
	return r.Client
}

// +kubebuilder:rbac:groups=openrl.io,resources=openrlworkloads,verbs=get;list;watch;update
// +kubebuilder:rbac:groups=openrl.io,resources=openrlworkloads/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=openrl.io,resources=openrlclaimgroups,verbs=get;list;watch;create;update;delete
// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceclaims,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceslices,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups=core,resources=nodes,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=events,verbs=create;patch
// +kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch;create;update;patch;delete

// Reconcile places one worker, deciding against a fresh read of the fleet.
func (r *OpenRLWorkloadReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	// Read through the consistent reader: deciding against a stale copy of
	// your own status is how seats get handed out twice.
	var worker openrlv1alpha1.OpenRLWorkload
	if err := r.fleetReader().Get(ctx, req.NamespacedName, &worker); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}
	if !worker.DeletionTimestamp.IsZero() {
		return r.teardown(ctx, &worker)
	}
	// The finalizer is the seat guarantee: teardown releases the group seat
	// only after the pod is verifiably gone, so the CR must outlive its pod
	// or the seat frees while the process still holds the device.
	if controllerutil.AddFinalizer(&worker, workerFinalizer) {
		if err := r.Update(ctx, &worker); err != nil {
			return ctrl.Result{}, err
		}
	}

	fleet, err := r.readFleet(ctx)
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("cannot read the fleet, placing nothing this pass: %w", err)
	}

	return r.place(ctx, &worker, requestFrom(&worker), fleet)
}

func (r *OpenRLWorkloadReconciler) place(ctx context.Context, worker *openrlv1alpha1.OpenRLWorkload, request placement.Request, fleet *placement.Fleet) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	if request.Memory <= 0 {
		return ctrl.Result{}, r.fail(ctx, worker, "InvalidSpec", "spec.accelerator.memory must be a positive quantity")
	}
	if err := validateTemplate(worker); err != nil {
		// Terminal, not retried: the spec is immutable, so an invalid
		// template can only be fixed by deleting and recreating.
		return ctrl.Result{}, r.fail(ctx, worker, "InvalidTemplate", err.Error())
	}

	podName := workerPodName(worker)
	pod, err := r.findPod(ctx, podName)
	if err != nil {
		return ctrl.Result{}, err
	}
	if pod != nil && pod.Labels[LabelWorker] != "" && pod.Labels[LabelWorker] != sanitizeLabel(worker.Name) {
		// A pod wearing this name but another worker's label is a collision,
		// not something to adopt or delete.
		return ctrl.Result{}, r.fail(ctx, worker, "PodConflict",
			fmt.Sprintf("pod %s belongs to worker %s", podName, pod.Labels[LabelWorker]))
	}
	if pod != nil {
		if owner := metav1.GetControllerOf(pod); owner != nil && owner.UID != worker.UID {
			// A predecessor's pod: adopting it would inherit a seat this CR
			// never booked and report its dying phase as ours. Replace it.
			logger.Info("pod belongs to a previous incarnation; replacing it", "pod", podName, "worker", worker.Name)
			if err := r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID}); err != nil && !apierrors.IsNotFound(err) {
				return ctrl.Result{}, err
			}
			// No requeue: the deletion event re-enqueues through Owns.
			return ctrl.Result{}, r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkloadStatus) {
				s.Phase = openrlv1alpha1.PhasePlacing
				s.Reason = "ReplacingPredecessorPod"
			})
		}
	}

	claimName := worker.Status.ClaimName
	if pod != nil && pod.Labels[LabelClaim] != "" && pod.Labels[LabelClaim] != claimName {
		// The running pod is the truth; adopting it recovers from a restart
		// that lost an unpatched status.
		claimName = pod.Labels[LabelClaim]
	}
	if claimName != "" {
		if _, live := fleet.Claims[claimName]; !live {
			logger.Info("assigned claim no longer exists; re-placing", "claim", claimName, "worker", worker.Name)
			claimName = ""
		}
	}

	if abandoned, err := r.abandonWedgedClaim(ctx, worker, fleet, claimName, pod); err != nil {
		return ctrl.Result{}, err
	} else if abandoned {
		return ctrl.Result{RequeueAfter: r.RetryInterval}, nil
	}

	claimName, err = r.shareAfterGrace(ctx, worker, request, fleet, claimName, pod)
	if err != nil {
		return ctrl.Result{}, err
	}

	if claimName == "" {
		claim, seat, verb, err := r.assign(ctx, worker, request, fleet)
		if err != nil {
			return ctrl.Result{}, err
		}
		if claim == nil {
			reason := placement.Explain(request, fleet, "")
			if r.expired(worker) {
				return ctrl.Result{}, r.fail(ctx, worker, "Unsatisfiable", reason)
			}
			return ctrl.Result{RequeueAfter: r.RetryInterval}, r.markPending(ctx, worker, reason)
		}
		claimName = claim.Name

		if err := r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkloadStatus) {
			s.Phase = openrlv1alpha1.PhasePlacing
			s.ClaimName = claimName
			s.AssignmentID = seat.AssignmentID
			s.Reason = verb
		}); err != nil {
			return ctrl.Result{}, err
		}
	} else if worker.Status.AssignmentID == "" || worker.Status.ClaimName != claimName {
		// A retained or pod-adopted claim has no fresh booking: re-book
		// idempotently so the seat list stays authoritative.
		_, seat, err := r.ensureSeat(ctx, claimName, maxSeatsFor(fleet, claimName), newSeat(worker, request))
		if err == errSeatUnavailable {
			reason := "SeatLost: the claim's group is full; waiting for a seat or a re-placement"
			return ctrl.Result{RequeueAfter: r.RetryInterval}, r.markPending(ctx, worker, reason)
		}
		if err != nil {
			return ctrl.Result{}, err
		}
		if err := r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkloadStatus) {
			s.ClaimName = claimName
			s.AssignmentID = seat.AssignmentID
		}); err != nil {
			return ctrl.Result{}, err
		}
	}

	if pod != nil && pod.Labels[LabelClaim] != "" && pod.Labels[LabelClaim] != claimName {
		// A pod's spec.resourceClaims is immutable, so a re-placed worker's
		// old pod can never reach the new claim: delete it and rebuild next
		// pass. No requeue -- the deletion event re-enqueues via Owns.
		logger.Info("pod is bound to a stale claim; recreating it", "pod", podName, "was", pod.Labels[LabelClaim], "now", claimName, "worker", worker.Name)
		if err := r.Delete(ctx, pod); err != nil && !apierrors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkloadStatus) {
			s.Phase = openrlv1alpha1.PhasePlacing
			s.ClaimName = claimName
			s.Reason = "RecreatingPodOnNewClaim"
		})
	}

	if pod == nil {
		if err := r.createPod(ctx, worker, podName, claimName); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkloadStatus) {
			s.Phase = openrlv1alpha1.PhasePlacing
			s.ClaimName = claimName
			s.PodName = podName
			s.Reason = "PodCreated"
		})
	}

	if detail := unschedulableMessage(pod); detail != "" {
		reason := placement.Explain(request, fleet, detail)
		if r.expired(worker) {
			return ctrl.Result{}, r.fail(ctx, worker, "Unschedulable", reason)
		}
		return ctrl.Result{RequeueAfter: r.RetryInterval}, r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkloadStatus) {
			s.Phase = openrlv1alpha1.PhasePending
			s.ClaimName, s.PodName, s.Reason = claimName, podName, reason
			setCondition(s, metav1.ConditionFalse, "Unschedulable", reason)
		})
	}

	return ctrl.Result{}, r.reportPod(ctx, worker, fleet, pod, claimName, podName)
}

// abandonWedgedClaim frees a worker whose *allocated* claim can no longer
// host its pod. The claim pins one node; if kube-scheduler has refused the
// pod there past the grace (host memory taken between pod incarnations, the
// node gone -- Spot preemption), nothing else can unpin it. The pod never
// started, so no process holds the device: delete the pod and claim, free
// the seat, clear the assignment, and the next pass starts over with fresh
// tiers.
func (r *OpenRLWorkloadReconciler) abandonWedgedClaim(ctx context.Context, worker *openrlv1alpha1.OpenRLWorkload, fleet *placement.Fleet, claimName string, pod *corev1.Pod) (bool, error) {
	if claimName == "" || r.ScaleOutGracePeriod <= 0 || pod == nil {
		return false, nil
	}
	if claim := fleet.Claims[claimName]; claim == nil || !claim.Allocated() {
		return false, nil // pending claims are shareAfterGrace's to handle
	}
	stuck := unschedulableSince(pod)
	if stuck.IsZero() || time.Since(stuck) < r.ScaleOutGracePeriod {
		return false, nil
	}

	log.FromContext(ctx).Info("abandoning a wedged allocated claim",
		"worker", worker.Name, "claim", claimName, "unschedulable", time.Since(stuck).Round(time.Second), "detail", unschedulableMessage(pod))
	if err := r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID}); err != nil && !apierrors.IsNotFound(err) {
		return false, err
	}
	if err := r.releaseSeat(ctx, groupNameFor(claimName), worker.Name, string(worker.UID)); err != nil {
		return false, err
	}
	claim := &resourcev1.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: claimName, Namespace: r.Namespace}}
	if err := r.Delete(ctx, claim); err != nil && !apierrors.IsNotFound(err) {
		return false, err
	}
	return true, r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkloadStatus) {
		s.Phase = openrlv1alpha1.PhasePending
		s.ClaimName, s.PodName, s.AssignmentID, s.NodeName = "", "", "", ""
		s.Reason = "ReplacingWedgedClaim: the allocated node refused the pod past the grace"
	})
}

// shareAfterGrace is the fall-back-to-sharing move: a dedicated claim DRA
// has not satisfied within the grace, whose pod kube-scheduler marked
// unschedulable, moves its worker onto an allocated claim -- new seat booked
// first, then the old one released; the stale-pod branch swaps the pod and
// the sweep retires the abandoned claim. Returns the (possibly new) claim.
func (r *OpenRLWorkloadReconciler) shareAfterGrace(ctx context.Context, worker *openrlv1alpha1.OpenRLWorkload, request placement.Request, fleet *placement.Fleet, claimName string, pod *corev1.Pod) (string, error) {
	if claimName == "" || r.ScaleOutGracePeriod <= 0 {
		return claimName, nil
	}
	claim := fleet.Claims[claimName]
	// A zero Created means the timestamp is not yet observed; the grace
	// clock starts once it is.
	if claim.Allocated() || claim.Created.IsZero() || time.Since(claim.Created) < r.ScaleOutGracePeriod {
		return claimName, nil
	}
	// Only an unschedulable pod falls back: a finished pod's deallocated
	// claim is terminal, not pending, and a pod without kube-scheduler's
	// verdict yet is not stuck.
	if pod == nil || unschedulableMessage(pod) == "" {
		return claimName, nil
	}
	target := placement.SelectClaim(request, fleet)
	if target == nil || target.Name == claimName {
		return claimName, nil
	}
	_, seat, err := r.ensureSeat(ctx, target.Name, maxSeatsFor(fleet, target.Name), newSeat(worker, request))
	if err == errSeatUnavailable {
		return claimName, nil // lost the seat race; stay dedicated and retry
	}
	if err != nil {
		return "", err
	}
	// Recheck the dedicated claim past the cache: if it allocated while the
	// seat was booked, prefer it and give the speculative seat back.
	var fresh resourcev1.ResourceClaim
	if err := r.fleetReader().Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: claimName}, &fresh); err == nil && fresh.Status.Allocation != nil {
		if err := r.releaseSeat(ctx, groupNameFor(target.Name), worker.Name, string(worker.UID)); err != nil {
			return "", err
		}
		return claimName, nil
	}
	log.FromContext(ctx).Info("moving pending worker to a shared claim",
		"worker", worker.Name, "from", claimName, "to", target.Name, "waited", time.Since(claim.Created).Round(time.Second))
	if err := r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkloadStatus) {
		s.Phase = openrlv1alpha1.PhasePlacing
		s.ClaimName = target.Name
		s.AssignmentID = seat.AssignmentID
		s.Reason = "SharedAfterScaleOutGrace"
	}); err != nil {
		return "", err
	}
	if err := r.releaseSeat(ctx, groupNameFor(claimName), worker.Name, string(worker.UID)); err != nil {
		return "", err
	}
	return target.Name, nil
}

// workerFinalizer is the deleted-worker seat guarantee: the CR -- and with
// it, the memory booking -- survives until its pod is verifiably gone.
const workerFinalizer = "openrl.io/placement"

// teardown drives a deleting worker: delete its pod, wait for the process to
// actually exit, then let the CR go. Claims are not touched here -- they are
// shared, and the reclaim sweep owns their end of life.
func (r *OpenRLWorkloadReconciler) teardown(ctx context.Context, worker *openrlv1alpha1.OpenRLWorkload) (ctrl.Result, error) {
	if !controllerutil.ContainsFinalizer(worker, workerFinalizer) {
		return ctrl.Result{}, nil
	}
	pod, err := r.findPod(ctx, workerPodName(worker))
	if err != nil {
		return ctrl.Result{}, err
	}
	if pod != nil && pod.Labels[LabelWorker] == sanitizeLabel(worker.Name) {
		if pod.DeletionTimestamp.IsZero() {
			if err := r.Delete(ctx, pod); err != nil && !apierrors.IsNotFound(err) {
				return ctrl.Result{}, err
			}
		}
		// Still terminating: the seat stays booked until it is gone.
		return ctrl.Result{RequeueAfter: r.RetryInterval}, nil
	}
	// The pod is verifiably gone: give the seat back before the CR goes.
	// Keyed by UID, so a successor incarnation's fresh seat survives this.
	if worker.Status.ClaimName != "" {
		if err := r.releaseSeat(ctx, groupNameFor(worker.Status.ClaimName), worker.Name, string(worker.UID)); err != nil {
			return ctrl.Result{}, err
		}
	}
	controllerutil.RemoveFinalizer(worker, workerFinalizer)
	return ctrl.Result{}, r.Update(ctx, worker)
}

// assign cuts a dedicated claim: ordered tiers for DRA to satisfy, never a
// survey of what is free. The seat is booked before the claim exists, so a
// joiner can never find a chartless claim; sharing happens only in
// shareAfterGrace.
func (r *OpenRLWorkloadReconciler) assign(ctx context.Context, worker *openrlv1alpha1.OpenRLWorkload, request placement.Request, fleet *placement.Fleet) (*placement.Claim, *openrlv1alpha1.Seat, string, error) {
	tiers := placement.Tiers(request, placement.Catalog(fleet, request.Role))
	if len(tiers) == 0 {
		return nil, nil, "", nil
	}

	claim := &placement.Claim{Name: claimNameFor(worker)}
	claim.Book(request.WorkerID, request.OwnerKey(), request.HostRequestBytes)

	// MaxSeats 0: the ceiling is a node property, unknowable until DRA
	// decides; the first booking that knows it stamps it.
	_, seat, err := r.ensureSeat(ctx, claim.Name, 0, newSeat(worker, request))
	if err != nil {
		return nil, nil, "", err
	}

	summary := tierSummary(tiers)
	verb := "CreatedClaim: " + summary
	log.FromContext(ctx).Info("cutting a claim", "claim", claim.Name, "worker", worker.Name, "tiers", summary)

	if err := r.Create(ctx, r.buildClaim(claim.Name, tiers)); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return nil, nil, "", fmt.Errorf("create claim %s: %w", claim.Name, err)
		}
		// Claim names are UID-derived, so an existing claim is this same
		// incarnation's earlier create. Adopt the cluster's copy -- it may
		// already be allocated.
		var existing resourcev1.ResourceClaim
		if err := r.fleetReader().Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: claim.Name}, &existing); err != nil {
			return nil, nil, "", fmt.Errorf("read existing claim %s: %w", claim.Name, err)
		}
		if existing.DeletionTimestamp != nil {
			// Our abandoned predecessor, still held by DRA's finalizer.
			// Wait it out rather than adopting a dying claim.
			return nil, nil, "", nil
		}
		adopted := claimFrom(&existing)
		adopted.Book(request.WorkerID, request.OwnerKey(), request.HostRequestBytes)
		fleet.Claims[adopted.Name] = adopted
		return adopted, seat, verb, nil
	}
	fleet.Claims[claim.Name] = claim
	return claim, seat, verb, nil
}

// tierSummary renders the ordered alternatives for logs and status.
func tierSummary(tiers []placement.Tier) string {
	names := make([]string, len(tiers))
	for i, tier := range tiers {
		names[i] = tier.Name
	}
	return strings.Join(names, "|")
}

func (r *OpenRLWorkloadReconciler) createPod(ctx context.Context, worker *openrlv1alpha1.OpenRLWorkload, podName, claimName string) error {
	pod, err := r.renderPod(worker, podName, claimName)
	if err != nil {
		// Record the failure, then still return the error: a nil return here
		// would read as "pod created" and nothing would retry the render.
		if patchErr := r.fail(ctx, worker, "TemplateError", err.Error()); patchErr != nil {
			return patchErr
		}
		return err
	}
	if err := r.Create(ctx, pod); err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("create pod %s: %w", podName, err)
	}
	return nil
}

// findPod returns the worker's pod, or nil if it has none.
func (r *OpenRLWorkloadReconciler) findPod(ctx context.Context, podName string) (*corev1.Pod, error) {
	var pod corev1.Pod
	err := r.Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: podName}, &pod)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read pod %s: %w", podName, err)
	}
	return &pod, nil
}

func (r *OpenRLWorkloadReconciler) reportPod(ctx context.Context, worker *openrlv1alpha1.OpenRLWorkload, fleet *placement.Fleet, pod *corev1.Pod, claimName, podName string) error {
	phase := openrlv1alpha1.PhasePlacing
	reason := ""
	switch pod.Status.Phase {
	case corev1.PodRunning, corev1.PodSucceeded:
		phase = openrlv1alpha1.PhaseRunning
	case corev1.PodFailed:
		phase, reason = openrlv1alpha1.PhaseFailed, "PodFailed"
	}

	node := pod.Spec.NodeName
	if node == "" {
		if claim, ok := fleet.Claims[claimName]; ok {
			node = claim.Node
		}
	}

	return r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkloadStatus) {
		s.Phase, s.ClaimName, s.PodName, s.NodeName, s.Reason = phase, claimName, podName, node, reason
		// The footprint is DRA's answer, recordable only once the
		// allocation is observed: which tier held, and what each device
		// carries under it.
		if claim, ok := fleet.Claims[claimName]; ok && claim.DeviceCount > 0 {
			s.DeviceCount = int32(claim.DeviceCount)
			s.MemoryPerDevice = gibQuantity(requestFrom(worker).PerDeviceBytes(claim.DeviceCount))
		}
		if phase == openrlv1alpha1.PhaseRunning {
			setCondition(s, metav1.ConditionTrue, "Placed", "worker is running on "+claimName)
		}
	})
}

// -- status -------------------------------------------------------------------

func (r *OpenRLWorkloadReconciler) patchStatus(ctx context.Context, worker *openrlv1alpha1.OpenRLWorkload, mutate func(*openrlv1alpha1.OpenRLWorkloadStatus)) error {
	before := worker.Status.DeepCopy()
	mutate(&worker.Status)
	worker.Status.ObservedGeneration = worker.Generation
	// Skip no-op writes: they would re-trigger the watch forever.
	if apiequality.Semantic.DeepEqual(before, &worker.Status) {
		return nil
	}
	if err := r.Status().Update(ctx, worker); err != nil {
		if apierrors.IsConflict(err) {
			// Someone else wrote first; the next reconcile recomputes from scratch.
			return nil
		}
		return fmt.Errorf("patch status of %s: %w", worker.Name, err)
	}
	return nil
}

func (r *OpenRLWorkloadReconciler) markPending(ctx context.Context, worker *openrlv1alpha1.OpenRLWorkload, reason string) error {
	return r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkloadStatus) {
		s.Phase, s.Reason = openrlv1alpha1.PhasePending, reason
		setCondition(s, metav1.ConditionFalse, "WaitingForCapacity", reason)
	})
}

func (r *OpenRLWorkloadReconciler) fail(ctx context.Context, worker *openrlv1alpha1.OpenRLWorkload, reason, message string) error {
	if r.Recorder != nil {
		r.Recorder.Event(worker, corev1.EventTypeWarning, reason, message)
	}
	return r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkloadStatus) {
		s.Phase, s.Reason = openrlv1alpha1.PhaseFailed, message
		setCondition(s, metav1.ConditionFalse, reason, message)
	})
}

// expired reports whether this worker has been waiting past the point where
// "not yet" should be called "no".
func (r *OpenRLWorkloadReconciler) expired(worker *openrlv1alpha1.OpenRLWorkload) bool {
	if r.PlacementTimeout <= 0 {
		return false
	}
	since := worker.CreationTimestamp.Time
	if condition := apimeta.FindStatusCondition(worker.Status.Conditions, openrlv1alpha1.ConditionPlaced); condition != nil {
		since = condition.LastTransitionTime.Time
	}
	return time.Since(since) > r.PlacementTimeout
}

func setCondition(status *openrlv1alpha1.OpenRLWorkloadStatus, state metav1.ConditionStatus, reason, message string) {
	// Kubernetes rejects condition messages over 32KiB; SetStatusCondition
	// does not truncate.
	if len(message) > 32768 {
		message = message[:32768]
	}
	apimeta.SetStatusCondition(&status.Conditions, metav1.Condition{
		Type:    openrlv1alpha1.ConditionPlaced,
		Status:  state,
		Reason:  reason,
		Message: message,
	})
}

// -- wiring --------------------------------------------------------------------

// SetupWithManager registers the reconciler and the claim-reclaim sweep.
func (r *OpenRLWorkloadReconciler) SetupWithManager(mgr ctrl.Manager) error {
	r.reader = mgr.GetAPIReader()

	if err := mgr.Add(manager.RunnableFunc(r.runReclaim)); err != nil {
		return err
	}

	// Capacity changes are fleet-wide: a freed claim might unblock any
	// pending worker, so these events wake every worker rather than one.
	wakeAll := handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, _ client.Object) []reconcile.Request {
		var workers openrlv1alpha1.OpenRLWorkloadList
		if err := mgr.GetClient().List(ctx, &workers, client.InNamespace(r.Namespace)); err != nil {
			return nil
		}
		requests := make([]reconcile.Request, 0, len(workers.Items))
		for i := range workers.Items {
			requests = append(requests, reconcile.Request{
				NamespacedName: types.NamespacedName{Namespace: workers.Items[i].Namespace, Name: workers.Items[i].Name},
			})
		}
		return requests
	})

	// Kubelet heartbeats rewrite node status every few seconds; only label
	// and allocatable-memory changes are capacity events worth waking for.
	nodeCapacityChanged := predicate.Or(
		predicate.LabelChangedPredicate{},
		predicate.Funcs{
			UpdateFunc: func(e event.UpdateEvent) bool {
				before, okBefore := e.ObjectOld.(*corev1.Node)
				after, okAfter := e.ObjectNew.(*corev1.Node)
				return okBefore && okAfter && !before.Status.Allocatable.Memory().Equal(*after.Status.Allocatable.Memory())
			},
		},
	)

	return ctrl.NewControllerManagedBy(mgr).
		For(&openrlv1alpha1.OpenRLWorkload{}).
		Owns(&corev1.Pod{}).
		Watches(&resourcev1.ResourceClaim{}, wakeAll, builder.WithPredicates(managedClaims())).
		Watches(&corev1.Node{}, wakeAll, builder.WithPredicates(nodeCapacityChanged)).
		WithOptions(controller.Options{MaxConcurrentReconciles: max(1, r.MaxConcurrentReconciles)}).
		Complete(r)
}
