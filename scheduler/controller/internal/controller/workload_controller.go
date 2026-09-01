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
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	openrlv1alpha1 "github.com/gke-labs/open-rl/scheduler/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

// WorkloadReconciler turns Workload requests into ResourceClaims
// and pods; the decision lives in internal/placement. Seat booking is
// CAS-arbitrated on the ClaimLedger, so reconciles run concurrently.
type WorkloadReconciler struct {
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
	// PlacementStrategy orders the two placement moves; the zero value reads
	// as spread.
	PlacementStrategy placement.Strategy
	// MaxConcurrentReconciles is how many workers place at once.
	MaxConcurrentReconciles int

	// reader bypasses the informer cache for read-modify-write paths only:
	// the seat CAS and claim adoption. Nil (in tests) falls back to the
	// regular client.
	reader client.Reader
}

// fleetReader is the consistent reader for fleet state.
func (r *WorkloadReconciler) fleetReader() client.Reader {
	if r.reader != nil {
		return r.reader
	}
	return r.Client
}

// +kubebuilder:rbac:groups=openrl.io,resources=workloads,verbs=get;list;watch;update
// +kubebuilder:rbac:groups=openrl.io,resources=workloads/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=openrl.io,resources=claimledgers,verbs=get;list;watch;create;update;delete
// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceclaims,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceslices,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups=core,resources=nodes,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=events,verbs=create;patch
// +kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch;create;update;patch;delete

// Reconcile places one worker, deciding against a fresh read of the fleet.
func (r *WorkloadReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	// Read through the consistent reader: deciding against a stale copy of
	// your own status is how seats get handed out twice.
	var worker openrlv1alpha1.Workload
	if err := r.fleetReader().Get(ctx, req.NamespacedName, &worker); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}
	if !worker.DeletionTimestamp.IsZero() {
		return r.teardown(ctx, &worker)
	}
	// The finalizer is the seat guarantee: teardown releases the ledger seat
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

type placementScope struct {
	worker  *openrlv1alpha1.Workload
	request placement.Request
	fleet   *placement.Fleet

	pod       *corev1.Pod
	podName   string
	claimName string
}

type placementPhaseResult struct {
	done   bool
	result ctrl.Result
}

type placementPhase func(context.Context, *placementScope) (placementPhaseResult, error)

// place drives one worker through an ordered lifecycle of independently testable phases.
func (r *WorkloadReconciler) place(ctx context.Context, worker *openrlv1alpha1.Workload, request placement.Request, fleet *placement.Fleet) (ctrl.Result, error) {
	scope := &placementScope{
		worker:    worker,
		request:   request,
		fleet:     fleet,
		podName:   workerPodName(worker),
		claimName: worker.Status.ClaimName,
	}

	// Reconcile through ordered phases, mirroring the structure used by mature Kubernetes controllers.
	phases := []placementPhase{
		r.validatePlacement,
		r.resolvePlacementPod,
		r.repairPlacementAssignment,
		r.applyPlacementFallbacks,
		r.ensurePlacementClaim,
		r.reconcilePlacementPod,
		r.reportPlacement,
	}
	for _, phase := range phases {
		outcome, err := phase(ctx, scope)
		if err != nil || outcome.done {
			return outcome.result, err
		}
	}
	return ctrl.Result{}, nil
}

func (r *WorkloadReconciler) validatePlacement(ctx context.Context, scope *placementScope) (placementPhaseResult, error) {
	// The spec is immutable, so these failures are terminal, not retried.
	if scope.request.Memory <= 0 {
		return placementPhaseResult{done: true}, r.fail(ctx, scope.worker, "InvalidSpec", "spec.accelerator.memory must be a positive quantity")
	}
	if err := validateTemplate(scope.worker); err != nil {
		return placementPhaseResult{done: true}, r.fail(ctx, scope.worker, "InvalidTemplate", err.Error())
	}
	return placementPhaseResult{}, nil
}

func (r *WorkloadReconciler) resolvePlacementPod(ctx context.Context, scope *placementScope) (placementPhaseResult, error) {
	pod, done, err := r.resolveWorkerPod(ctx, scope.worker, scope.podName)
	if done != nil {
		return placementPhaseResult{done: true, result: *done}, err
	}
	scope.pod = pod
	return placementPhaseResult{}, err
}

func (r *WorkloadReconciler) repairPlacementAssignment(ctx context.Context, scope *placementScope) (placementPhaseResult, error) {
	worker := scope.worker
	logger := log.FromContext(ctx)

	if scope.pod != nil && scope.pod.Labels[LabelClaim] != "" && scope.pod.Labels[LabelClaim] != scope.claimName {
		// The running pod is the truth; adopting it recovers from a restart
		// that lost an unpatched status.
		scope.claimName = scope.pod.Labels[LabelClaim]
	}
	if scope.claimName != "" {
		if _, live := scope.fleet.Claims[scope.claimName]; !live {
			// The ledger can outlive its claim, and a booking held there would
			// count against the node's host memory forever.
			logger.Info("assigned claim no longer exists; releasing the seat and re-placing", "claim", scope.claimName, "worker", worker.Name)
			if err := r.releaseSeatAndReclaim(ctx, scope.claimName, worker.Name, worker.UID); err != nil {
				return placementPhaseResult{}, err
			}
			scope.claimName = ""
		}
	}
	// One seat per worker, on its own claim only. A status write lost after
	// a seat move strands a booking on a ledger this worker no longer tracks,
	// and no other path revisits it: teardown releases only the status claim.
	for _, stray := range scope.fleet.Claims {
		if stray.Name == scope.claimName || !stray.Booked(worker.Name) {
			continue
		}
		logger.Info("releasing a stray seat", "claim", stray.Name, "worker", worker.Name)
		if err := r.releaseSeatAndReclaim(ctx, stray.Name, worker.Name, worker.UID); err != nil {
			return placementPhaseResult{}, err
		}
	}
	return placementPhaseResult{}, nil
}

func (r *WorkloadReconciler) applyPlacementFallbacks(ctx context.Context, scope *placementScope) (placementPhaseResult, error) {
	abandoned, err := r.abandonWedgedClaim(ctx, scope.worker, scope.fleet, scope.claimName, scope.pod)
	if err != nil {
		return placementPhaseResult{}, err
	}
	if abandoned {
		return placementPhaseResult{done: true, result: ctrl.Result{RequeueAfter: r.RetryInterval}}, nil
	}

	scope.claimName, err = r.fallBackToSharing(ctx, scope.worker, scope.request, scope.fleet, scope.claimName, scope.pod)
	return placementPhaseResult{}, err
}

func (r *WorkloadReconciler) ensurePlacementClaim(ctx context.Context, scope *placementScope) (placementPhaseResult, error) {
	worker := scope.worker
	if scope.claimName == "" {
		claim, seat, verb, err := r.assign(ctx, worker, scope.request, scope.fleet)
		if err != nil {
			return placementPhaseResult{}, err
		}
		if claim == nil {
			reason := placement.Explain(scope.request, scope.fleet, "")
			if r.expired(worker) {
				return placementPhaseResult{done: true}, r.fail(ctx, worker, "Unsatisfiable", reason)
			}
			return placementPhaseResult{done: true, result: ctrl.Result{RequeueAfter: r.RetryInterval}}, r.markPending(ctx, worker, reason)
		}
		scope.claimName = claim.Name

		return placementPhaseResult{}, r.patchStatus(ctx, worker, func(status *openrlv1alpha1.WorkloadStatus) {
			status.Phase = openrlv1alpha1.PhasePlacing
			status.ClaimName = scope.claimName
			status.AssignmentID = seat.AssignmentID
			status.Reason = verb
		})
	}
	if worker.Status.AssignmentID != "" && worker.Status.ClaimName == scope.claimName {
		return placementPhaseResult{}, nil
	}

	// A retained or pod-adopted claim has no fresh booking: re-book
	// idempotently so the seat list stays authoritative.
	_, seat, err := r.ensureSeat(ctx, scope.claimName, newSeat(worker, scope.request), true)
	if err == errBookingContended {
		reason := "SeatLost: the ledger is contended; the booking will be retried"
		return placementPhaseResult{done: true, result: ctrl.Result{RequeueAfter: r.RetryInterval}}, r.markPending(ctx, worker, reason)
	}
	if err != nil {
		return placementPhaseResult{}, err
	}
	return placementPhaseResult{}, r.patchStatus(ctx, worker, func(status *openrlv1alpha1.WorkloadStatus) {
		status.ClaimName = scope.claimName
		status.AssignmentID = seat.AssignmentID
	})
}

func (r *WorkloadReconciler) reconcilePlacementPod(ctx context.Context, scope *placementScope) (placementPhaseResult, error) {
	worker := scope.worker
	if scope.pod != nil && scope.pod.Labels[LabelClaim] != "" && scope.pod.Labels[LabelClaim] != scope.claimName {
		// A pod's spec.resourceClaims is immutable, so a re-placed worker's
		// old pod can never reach the new claim: delete it and rebuild next pass.
		log.FromContext(ctx).Info("pod is bound to a stale claim; recreating it", "pod", scope.podName, "was", scope.pod.Labels[LabelClaim], "now", scope.claimName, "worker", worker.Name)
		if err := r.Delete(ctx, scope.pod); err != nil && !apierrors.IsNotFound(err) {
			return placementPhaseResult{}, err
		}
		return placementPhaseResult{done: true}, r.patchStatus(ctx, worker, func(status *openrlv1alpha1.WorkloadStatus) {
			status.Phase = openrlv1alpha1.PhasePlacing
			status.ClaimName = scope.claimName
			status.Reason = "RecreatingPodOnNewClaim"
		})
	}

	if scope.pod == nil {
		if err := r.createPod(ctx, worker, scope.podName, scope.claimName, placement.PreferTightFit(scope.fleet, scope.request)); err != nil {
			return placementPhaseResult{}, err
		}
		return placementPhaseResult{done: true}, r.patchStatus(ctx, worker, func(status *openrlv1alpha1.WorkloadStatus) {
			status.Phase = openrlv1alpha1.PhasePlacing
			status.ClaimName = scope.claimName
			status.PodName = scope.podName
			status.Reason = "PodCreated"
		})
	}
	return placementPhaseResult{}, nil
}

func (r *WorkloadReconciler) reportPlacement(ctx context.Context, scope *placementScope) (placementPhaseResult, error) {
	if detail := unschedulableMessage(scope.pod); detail != "" {
		reason := placement.Explain(scope.request, scope.fleet, detail)
		if r.expired(scope.worker) {
			return placementPhaseResult{done: true}, r.fail(ctx, scope.worker, "Unschedulable", reason)
		}
		return placementPhaseResult{done: true, result: ctrl.Result{RequeueAfter: r.RetryInterval}}, r.patchStatus(ctx, scope.worker, func(status *openrlv1alpha1.WorkloadStatus) {
			status.Phase = openrlv1alpha1.PhasePending
			status.ClaimName, status.PodName, status.Reason = scope.claimName, scope.podName, reason
			setCondition(status, metav1.ConditionFalse, "Unschedulable", reason)
		})
	}

	err := r.reportPod(ctx, scope.worker, scope.fleet, scope.pod, scope.claimName, scope.podName)
	return placementPhaseResult{done: true}, err
}

// resolveWorkerPod fetches the worker's pod and checks it belongs to this
// worker. A pod with this name but another worker's label is a naming
// collision: the workload is marked Failed and the pod is left alone. A pod
// owned by an earlier incarnation of this worker is deleted, because its
// seat and assignment belonged to the old CR; the next pass creates a fresh
// one. If done is non-nil, place() returns it immediately.
func (r *WorkloadReconciler) resolveWorkerPod(ctx context.Context, worker *openrlv1alpha1.Workload, podName string) (*corev1.Pod, *ctrl.Result, error) {
	pod, err := r.findPod(ctx, podName)
	if err != nil {
		return nil, &ctrl.Result{}, err
	}
	if pod == nil {
		return nil, nil, nil
	}
	if pod.Labels[LabelWorker] != "" && pod.Labels[LabelWorker] != sanitizeLabel(worker.Name) {
		return nil, &ctrl.Result{}, r.fail(ctx, worker, "PodConflict",
			fmt.Sprintf("pod %s belongs to worker %s", podName, pod.Labels[LabelWorker]))
	}
	if owner := metav1.GetControllerOf(pod); owner != nil && owner.UID != worker.UID {
		log.FromContext(ctx).Info("pod belongs to a previous incarnation; replacing it", "pod", podName, "worker", worker.Name)
		if err := r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID}); err != nil && !apierrors.IsNotFound(err) {
			return nil, &ctrl.Result{}, err
		}
		// No requeue: the deletion event re-enqueues through Owns.
		return nil, &ctrl.Result{}, r.patchStatus(ctx, worker, func(status *openrlv1alpha1.WorkloadStatus) {
			status.Phase = openrlv1alpha1.PhasePlacing
			status.Reason = "ReplacingPredecessorPod"
		})
	}
	return pod, nil, nil
}

// wedgeGracePeriod is how long an *allocated* claim's pod may sit
// unschedulable before the claim is declared wedged and abandoned. Not an
// autoscaler wait: it only distinguishes a stuck pod from pod-turnover
// transients -- a terminating predecessor holds its memory request until it
// is gone, and abandoning an allocated claim over that would churn a GPU.
// The pending-claim fallback takes the same verdict with no debounce: a
// misread there costs a detour onto a shared seat, not a freed device.
const wedgeGracePeriod = 2 * time.Minute

// abandonWedgedClaim frees a worker whose *allocated* claim can no longer
// host its pod. The claim pins one node; if kube-scheduler has refused the
// pod there past the wedge grace (host memory taken between pod
// incarnations, the node gone -- Spot preemption), nothing else can unpin
// it. The pod never started, so no process holds the device: delete the pod
// and claim, free the seat, clear the assignment, and the next pass starts
// over with fresh tiers.
func (r *WorkloadReconciler) abandonWedgedClaim(ctx context.Context, worker *openrlv1alpha1.Workload, fleet *placement.Fleet, claimName string, pod *corev1.Pod) (bool, error) {
	if claimName == "" || pod == nil {
		return false, nil
	}
	if claim := fleet.Claims[claimName]; claim == nil || !claim.Allocated() {
		return false, nil // pending claims are fallBackToSharing's to handle
	}
	stuck := unschedulableSince(pod)
	if stuck.IsZero() || time.Since(stuck) < wedgeGracePeriod {
		return false, nil
	}

	log.FromContext(ctx).Info("abandoning a wedged allocated claim",
		"worker", worker.Name, "claim", claimName, "unschedulable", time.Since(stuck).Round(time.Second), "detail", unschedulableMessage(pod))
	if err := r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID}); err != nil && !apierrors.IsNotFound(err) {
		return false, err
	}
	// releaseSeatAndReclaim deletes the claim only when this was its last
	// seat: a shared claim still backs its co-tenants, who lose nothing but us.
	if err := r.releaseSeatAndReclaim(ctx, claimName, worker.Name, worker.UID); err != nil {
		return false, err
	}
	return true, r.patchStatus(ctx, worker, func(status *openrlv1alpha1.WorkloadStatus) {
		status.Phase = openrlv1alpha1.PhasePending
		status.ClaimName, status.PodName, status.AssignmentID, status.NodeName = "", "", "", ""
		status.Reason = "ReplacingWedgedClaim: the allocated node refused the pod past the grace"
	})
}

// fallBackToSharing is the fall-back-to-sharing move, triggered by
// kube-scheduler's verdict and nothing else: a dedicated claim DRA has not
// satisfied, whose pod is marked unschedulable, moves its worker onto an
// allocated claim -- new seat booked first, then the old one released; the
// stale-pod branch swaps the pod. No timer: on a fixed fleet waiting is dead
// time, and the verdict is the event that says the cluster has no free
// device now. Returns the (possibly new) claim.
func (r *WorkloadReconciler) fallBackToSharing(ctx context.Context, worker *openrlv1alpha1.Workload, request placement.Request, fleet *placement.Fleet, claimName string, pod *corev1.Pod) (string, error) {
	if claimName == "" {
		return claimName, nil
	}
	if fleet.Claims[claimName].Allocated() {
		return claimName, nil // a satisfied claim is home, not a fallback case
	}
	// Only an unschedulable pod falls back: a finished pod's deallocated
	// claim is terminal, not pending, and a pod without kube-scheduler's
	// verdict yet is not stuck.
	if pod == nil || unschedulableMessage(pod) == "" {
		return claimName, nil
	}
	target, seat, err := r.joinExistingClaim(ctx, worker, request, fleet, claimName)
	if err != nil {
		return "", err
	}
	if target == nil {
		return claimName, nil // no eligible ledger, or lost the booking race
	}
	// Recheck the dedicated claim past the cache: if it allocated while the
	// seat was booked, prefer it and give the speculative seat back.
	var fresh resourcev1.ResourceClaim
	if err := r.fleetReader().Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: claimName}, &fresh); err == nil && fresh.Status.Allocation != nil {
		if err := r.releaseSeatAndReclaim(ctx, target.Name, worker.Name, worker.UID); err != nil {
			return "", err
		}
		return claimName, nil
	}
	log.FromContext(ctx).Info("moving pending worker to a shared claim",
		"worker", worker.Name, "from", claimName, "to", target.Name)
	if err := r.patchStatus(ctx, worker, func(status *openrlv1alpha1.WorkloadStatus) {
		status.Phase = openrlv1alpha1.PhasePlacing
		status.ClaimName = target.Name
		status.AssignmentID = seat.AssignmentID
		status.Reason = "SharedOnUnschedulable"
	}); err != nil {
		return "", err
	}
	// Reclaiming the abandoned dedicated claim inline also retracts the
	// autoscale signal the moment we stop wanting the node.
	if err := r.releaseSeatAndReclaim(ctx, claimName, worker.Name, worker.UID); err != nil {
		return "", err
	}
	return target.Name, nil
}

// workerFinalizer is the deleted-worker seat guarantee: the CR -- and with
// it, the memory booking -- survives until its pod is verifiably gone.
const workerFinalizer = "openrl.io/placement"

// teardown drives a deleting worker: delete its pod, wait for the process to
// actually exit, then let the CR go. Releasing the last seat reclaims the
// claim inline; the sweep only backstops a crash between the two.
func (r *WorkloadReconciler) teardown(ctx context.Context, worker *openrlv1alpha1.Workload) (ctrl.Result, error) {
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
		if err := r.releaseSeatAndReclaim(ctx, worker.Status.ClaimName, worker.Name, worker.UID); err != nil {
			return ctrl.Result{}, err
		}
	}
	controllerutil.RemoveFinalizer(worker, workerFinalizer)
	return ctrl.Result{}, r.Update(ctx, worker)
}

// joinExistingClaim books a seat on the claim SelectClaim proposes and keeps
// the fleet snapshot's ledger current. A nil claim means no join this pass:
// no eligible ledger beyond current, or the booking lost every CAS retry.
func (r *WorkloadReconciler) joinExistingClaim(ctx context.Context, worker *openrlv1alpha1.Workload, request placement.Request, fleet *placement.Fleet, current string) (*placement.Claim, *openrlv1alpha1.Seat, error) {
	target := placement.SelectClaim(request, fleet)
	if target == nil || target.Name == current {
		return nil, nil, nil
	}
	_, seat, err := r.ensureSeat(ctx, target.Name, newSeat(worker, request), false)
	if err == errBookingContended {
		return nil, nil, nil
	}
	if err != nil {
		return nil, nil, err
	}
	target.Book(request.WorkerID, request.OwnerKey(), request.HostRequestBytes)
	return target, seat, nil
}

// assign orders joining and claim creation according to the configured strategy.
func (r *WorkloadReconciler) assign(ctx context.Context, worker *openrlv1alpha1.Workload, request placement.Request, fleet *placement.Fleet) (*placement.Claim, *openrlv1alpha1.Seat, string, error) {
	switch r.PlacementStrategy {
	case "", placement.StrategySpread:
		return r.cutDedicatedClaim(ctx, worker, request, fleet)
	case placement.StrategyBinPack:
		target, seat, err := r.joinExistingClaim(ctx, worker, request, fleet, "")
		if err != nil {
			return nil, nil, "", err
		}
		if target != nil {
			log.FromContext(ctx).Info("packing onto an existing claim", "claim", target.Name, "worker", worker.Name)
			return target, seat, "JoinedClaim: " + target.Name, nil
		}
		return r.cutDedicatedClaim(ctx, worker, request, fleet)
	default:
		return nil, nil, "", fmt.Errorf("unsupported placement strategy %q", r.PlacementStrategy)
	}
}

// cutDedicatedClaim books the first seat, then creates an ordered set of DRA alternatives.
func (r *WorkloadReconciler) cutDedicatedClaim(ctx context.Context, worker *openrlv1alpha1.Workload, request placement.Request, fleet *placement.Fleet) (*placement.Claim, *openrlv1alpha1.Seat, string, error) {
	tiers := placement.Tiers(request, placement.Catalog(fleet, request.Role))
	if len(tiers) == 0 {
		return nil, nil, "", nil
	}

	claim := &placement.Claim{Name: claimNameFor(worker)}
	claim.Book(request.WorkerID, request.OwnerKey(), request.HostRequestBytes)

	claimLedger, seat, err := r.ensureSeat(ctx, claim.Name, newSeat(worker, request), true)
	if err != nil {
		return nil, nil, "", err
	}

	summary := tierSummary(tiers)
	verb := "CreatedClaim: " + summary
	log.FromContext(ctx).Info("cutting a claim", "claim", claim.Name, "worker", worker.Name, "tiers", summary)

	body := r.buildClaim(claim.Name, tiers)
	// The ledger owns the claim. Retirement still deletes the ledger first --
	// its absence stays the tombstone joiners honor -- and if the controller
	// crashes before the follow-up claim delete, garbage collection finishes
	// the job instead of leaving an allocated device orphaned.
	body.OwnerReferences = []metav1.OwnerReference{{
		APIVersion: openrlv1alpha1.GroupVersion.String(),
		Kind:       "ClaimLedger",
		Name:       claimLedger.Name,
		UID:        claimLedger.UID,
	}}
	if err := r.Create(ctx, body); err != nil {
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

func (r *WorkloadReconciler) createPod(ctx context.Context, worker *openrlv1alpha1.Workload, podName, claimName string, prefs []placement.NodePreference) error {
	pod, err := r.renderPod(worker, podName, claimName, prefs)
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
func (r *WorkloadReconciler) findPod(ctx context.Context, podName string) (*corev1.Pod, error) {
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

func (r *WorkloadReconciler) reportPod(ctx context.Context, worker *openrlv1alpha1.Workload, fleet *placement.Fleet, pod *corev1.Pod, claimName, podName string) error {
	phase := openrlv1alpha1.PhasePlacing
	reason := ""
	switch pod.Status.Phase {
	case corev1.PodRunning:
		phase = openrlv1alpha1.PhaseRunning
	case corev1.PodSucceeded:
		phase = openrlv1alpha1.PhaseSucceeded
	case corev1.PodFailed:
		phase, reason = openrlv1alpha1.PhaseFailed, "PodFailed"
	}

	node := pod.Spec.NodeName
	if node == "" {
		if claim, ok := fleet.Claims[claimName]; ok {
			node = claim.Node
		}
	}

	return r.patchStatus(ctx, worker, func(status *openrlv1alpha1.WorkloadStatus) {
		status.Phase, status.ClaimName, status.PodName, status.NodeName, status.Reason = phase, claimName, podName, node, reason
		// The footprint is DRA's answer, recordable only once the
		// allocation is observed: which tier held, and what each device
		// carries under it.
		if claim, ok := fleet.Claims[claimName]; ok && claim.DeviceCount > 0 {
			status.DeviceCount = int32(claim.DeviceCount)
			status.MemoryPerDevice = gibQuantity(requestFrom(worker).PerDeviceBytes(claim.DeviceCount))
		}
		if phase == openrlv1alpha1.PhaseRunning {
			setCondition(status, metav1.ConditionTrue, "Placed", "worker is running on "+claimName)
		}
	})
}

// -- status -------------------------------------------------------------------

func (r *WorkloadReconciler) patchStatus(ctx context.Context, worker *openrlv1alpha1.Workload, mutate func(*openrlv1alpha1.WorkloadStatus)) error {
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

func (r *WorkloadReconciler) markPending(ctx context.Context, worker *openrlv1alpha1.Workload, reason string) error {
	return r.patchStatus(ctx, worker, func(status *openrlv1alpha1.WorkloadStatus) {
		status.Phase, status.Reason = openrlv1alpha1.PhasePending, reason
		setCondition(status, metav1.ConditionFalse, "WaitingForCapacity", reason)
	})
}

func (r *WorkloadReconciler) fail(ctx context.Context, worker *openrlv1alpha1.Workload, reason, message string) error {
	if r.Recorder != nil {
		r.Recorder.Event(worker, corev1.EventTypeWarning, reason, message)
	}
	return r.patchStatus(ctx, worker, func(status *openrlv1alpha1.WorkloadStatus) {
		status.Phase, status.Reason = openrlv1alpha1.PhaseFailed, message
		setCondition(status, metav1.ConditionFalse, reason, message)
	})
}

// expired reports whether this worker has been waiting past the point where
// "not yet" should be called "no". A placed worker never expires: a True
// condition's age is time spent running, not waiting, so the clock reads
// only time spent continuously unplaced.
func (r *WorkloadReconciler) expired(worker *openrlv1alpha1.Workload) bool {
	if r.PlacementTimeout <= 0 {
		return false
	}
	since := worker.CreationTimestamp.Time
	if condition := apimeta.FindStatusCondition(worker.Status.Conditions, openrlv1alpha1.ConditionPlaced); condition != nil {
		if condition.Status == metav1.ConditionTrue {
			return false
		}
		since = condition.LastTransitionTime.Time
	}
	return time.Since(since) > r.PlacementTimeout
}

func setCondition(status *openrlv1alpha1.WorkloadStatus, state metav1.ConditionStatus, reason, message string) {
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

// managedClaims restricts the ResourceClaim watch to the ones this controller
// created, so unrelated DRA traffic does not wake every worker.
func managedClaims() predicate.Predicate {
	return predicate.NewPredicateFuncs(func(obj client.Object) bool {
		return obj.GetLabels()[LabelManaged] == "true"
	})
}

// SetupWithManager registers the reconciler.
func (r *WorkloadReconciler) SetupWithManager(mgr ctrl.Manager) error {
	r.reader = mgr.GetAPIReader()

	// Capacity changes are fleet-wide: a freed claim might unblock any
	// pending worker, so these events wake every worker rather than one.
	wakeAll := handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, _ client.Object) []reconcile.Request {
		var workers openrlv1alpha1.WorkloadList
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
		For(&openrlv1alpha1.Workload{}).
		Owns(&corev1.Pod{}).
		Watches(&resourcev1.ResourceClaim{}, wakeAll, builder.WithPredicates(managedClaims())).
		Watches(&corev1.Node{}, wakeAll, builder.WithPredicates(nodeCapacityChanged)).
		WithOptions(controller.Options{MaxConcurrentReconciles: max(1, r.MaxConcurrentReconciles)}).
		Complete(r)
}
