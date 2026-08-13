package controller

import (
	"context"
	"fmt"
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

// OpenRLWorkerReconciler turns OpenRLWorker requests into ResourceClaims and
// pods. The decision lives in internal/placement; this is the part that reads
// and writes Kubernetes objects. Concurrency is one: two reconciles at once
// would each decide against a fleet missing the other's booking.
type OpenRLWorkerReconciler struct {
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
	// DefaultPodTemplates names the ConfigMap per role used when a worker does
	// not name one itself.
	DefaultPodTemplates map[openrlv1alpha1.WorkerRole]string
	// RetryInterval is how often a worker that could not be placed is retried.
	RetryInterval time.Duration
	// PlacementTimeout is how long a worker may go unplaced before the request
	// is declared unsatisfiable. Without it an impossible request waits
	// forever, indistinguishable from one that is merely queued.
	PlacementTimeout time.Duration
	// ReclaimInterval is how often idle claims are swept.
	ReclaimInterval time.Duration

	// reader reads straight from the API server, past the informer cache: a
	// placement is three writes the cache reflects only eventually, and a
	// burst of workers reconciled back to back must each see the previous
	// one's booking. Nil (in tests) falls back to the regular client.
	reader client.Reader
}

// fleetReader is the consistent reader for fleet state.
func (r *OpenRLWorkerReconciler) fleetReader() client.Reader {
	if r.reader != nil {
		return r.reader
	}
	return r.Client
}

// +kubebuilder:rbac:groups=openrl.io,resources=openrlworkers,verbs=get;list;watch;update
// +kubebuilder:rbac:groups=openrl.io,resources=openrlworkers/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceclaims,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceslices,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups=core,resources=nodes,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=events,verbs=create;patch
// +kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch;create;update;patch;delete

// Reconcile places one worker, deciding against a fresh read of the fleet.
func (r *OpenRLWorkerReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	// Read through the consistent reader: deciding against a stale copy of
	// your own status is how seats get handed out twice.
	var worker openrlv1alpha1.OpenRLWorker
	if err := r.fleetReader().Get(ctx, req.NamespacedName, &worker); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}
	if !worker.DeletionTimestamp.IsZero() {
		return r.teardown(ctx, &worker)
	}
	// The finalizer is the seat guarantee: occupancy is rebuilt from worker
	// statuses, so the CR must outlive its pod or the seat frees while the
	// process still holds the device.
	if controllerutil.AddFinalizer(&worker, workerFinalizer) {
		if err := r.Update(ctx, &worker); err != nil {
			return ctrl.Result{}, err
		}
	}

	request := requestFrom(&worker)

	var workers openrlv1alpha1.OpenRLWorkerList
	if err := r.fleetReader().List(ctx, &workers, client.InNamespace(r.Namespace)); err != nil {
		return ctrl.Result{}, fmt.Errorf("list workers: %w", err)
	}
	fleet, err := r.readFleet(ctx, workers.Items)
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("cannot read the fleet, placing nothing this pass: %w", err)
	}

	return r.place(ctx, &worker, request, fleet)
}

func (r *OpenRLWorkerReconciler) place(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, request placement.Request, fleet *placement.Fleet) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	if request.Memory <= 0 {
		return ctrl.Result{}, r.fail(ctx, worker, "InvalidSpec", "spec.memory must be a positive quantity")
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
			// A previous incarnation's pod: a worker deleted and recreated
			// under the same name reaches here before garbage collection has
			// caught up. Adopting it would inherit a claim seat this CR never
			// booked -- the over-admission a recreate storm produces -- and
			// its dying phase would be reported as this worker's. Replace it.
			logger.Info("pod belongs to a previous incarnation; replacing it", "pod", podName, "worker", worker.Name)
			if err := r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID}); err != nil && !apierrors.IsNotFound(err) {
				return ctrl.Result{}, err
			}
			// No requeue: the deletion event re-enqueues through Owns.
			return ctrl.Result{}, r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
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

	if claimName == "" {
		claim, created, err := r.assign(ctx, worker, request, fleet)
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
		perDevice := request.PerDeviceBytes(claim.DeviceCount)

		verb := "SharedExistingClaim"
		if created {
			verb = fmt.Sprintf("CreatedClaim: %dx%s", claim.DeviceCount, gibQuantity(perDevice))
		}
		if err := r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
			s.Phase = openrlv1alpha1.PhasePlacing
			s.ClaimName = claimName
			s.Reason = verb
			recordFootprint(s, worker, request, claim.DeviceCount, perDevice)
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
		return ctrl.Result{}, r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
			s.Phase = openrlv1alpha1.PhasePlacing
			s.ClaimName = claimName
			s.Reason = "RecreatingPodOnNewClaim"
		})
	}

	if pod == nil {
		if err := r.createPod(ctx, worker, podName, claimName); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
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
		return ctrl.Result{RequeueAfter: r.RetryInterval}, r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
			s.Phase = openrlv1alpha1.PhasePending
			s.ClaimName, s.PodName, s.Reason = claimName, podName, reason
			setCondition(s, metav1.ConditionFalse, "Unschedulable", reason)
		})
	}

	return ctrl.Result{}, r.reportPod(ctx, worker, fleet, pod, claimName, podName)
}

// workerFinalizer is the deleted-worker seat guarantee: the CR -- and with
// it, the memory booking -- survives until its pod is verifiably gone.
const workerFinalizer = "openrl.io/placement"

// teardown drives a deleting worker: delete its pod, wait for the process to
// actually exit, then let the CR go. Claims are not touched here -- they are
// shared, and the reclaim sweep owns their end of life.
func (r *OpenRLWorkerReconciler) teardown(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker) (ctrl.Result, error) {
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
	controllerutil.RemoveFinalizer(worker, workerFinalizer)
	return ctrl.Result{}, r.Update(ctx, worker)
}

// assign cuts a new claim, or joins an existing one; the returned bool says
// which. The spread-before-share policy itself lives in placement.Decide.
func (r *OpenRLWorkerReconciler) assign(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, request placement.Request, fleet *placement.Fleet) (*placement.Claim, bool, error) {
	pool, join := placement.Decide(request, fleet)
	if join != nil {
		// Book immediately, so a worker reconciled straight after this one
		// sees the seat taken.
		join.Book(request.WorkerID, request.PerDeviceBytes(join.DeviceCount))
		return join, false, nil
	}
	if pool == nil {
		return nil, false, nil
	}

	perDevice := request.PerDeviceBytes(pool.DeviceCount)
	claim := &placement.Claim{
		Name:        claimNameFor(worker),
		DeviceCount: pool.DeviceCount,
		// Node stays empty until DRA decides; SizedAgainst reserves the
		// pool's devices in the meantime.
		SizedAgainst: pool.Node.Name,
	}
	claim.Book(request.WorkerID, perDevice)

	log.FromContext(ctx).Info("cutting a claim",
		"claim", claim.Name, "devices", pool.DeviceCount, "perDevice", gibQuantity(perDevice), "sizedAgainst", pool.Node.Name)

	body := r.buildClaim(worker, claim, perDevice, pool.Node.DeviceMemoryBytes)
	if err := r.Create(ctx, body); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return nil, false, fmt.Errorf("create claim %s: %w", claim.Name, err)
		}
		// Claim names are UID-derived, so an existing claim is this same
		// incarnation's earlier create. Adopt the cluster's copy -- it may
		// already be allocated.
		var existing resourcev1.ResourceClaim
		if err := r.fleetReader().Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: claim.Name}, &existing); err != nil {
			return nil, false, fmt.Errorf("read existing claim %s: %w", claim.Name, err)
		}
		adopted := r.claimFrom(ctx, &existing)
		if adopted == nil {
			return nil, false, fmt.Errorf("claim %s exists but its shape is unreadable", claim.Name)
		}
		adopted.Book(request.WorkerID, perDevice)
		fleet.Claims[adopted.Name] = adopted
		return adopted, true, nil
	}
	fleet.Claims[claim.Name] = claim
	return claim, true, nil
}

func (r *OpenRLWorkerReconciler) createPod(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, podName, claimName string) error {
	pod, err := r.renderPod(ctx, worker, podName, claimName)
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

// findPod returns the worker's pod, or nil if it has none. A terminal pod is
// reported, not replaced: whether a finished model still wants a worker is
// the gateway's call.
func (r *OpenRLWorkerReconciler) findPod(ctx context.Context, podName string) (*corev1.Pod, error) {
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

func (r *OpenRLWorkerReconciler) reportPod(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, fleet *placement.Fleet, pod *corev1.Pod, claimName, podName string) error {
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

	return r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
		s.Phase, s.ClaimName, s.PodName, s.NodeName, s.Reason = phase, claimName, podName, node, reason
		if phase == openrlv1alpha1.PhaseRunning {
			setCondition(s, metav1.ConditionTrue, "Placed", "worker is running on "+claimName)
		}
	})
}

// recordFootprint writes down the memory this placement implies. Parking a
// worker costs its whole accelerator footprint in host RAM, however it is
// spread, so the parked figure is the request's memory itself.
func recordFootprint(status *openrlv1alpha1.OpenRLWorkerStatus, worker *openrlv1alpha1.OpenRLWorker, request placement.Request, deviceCount int, perDevice int64) {
	status.DeviceCount = int32(deviceCount)
	status.MemoryPerDevice = gibQuantity(perDevice)
	status.HostMemoryWhenParked = gibQuantity(request.Memory)
	status.EstimatorVersion = worker.Spec.EstimatorVersion
}

// -- status -------------------------------------------------------------------

func (r *OpenRLWorkerReconciler) patchStatus(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, mutate func(*openrlv1alpha1.OpenRLWorkerStatus)) error {
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

func (r *OpenRLWorkerReconciler) markPending(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, reason string) error {
	return r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
		s.Phase, s.Reason = openrlv1alpha1.PhasePending, reason
		setCondition(s, metav1.ConditionFalse, "WaitingForCapacity", reason)
	})
}

func (r *OpenRLWorkerReconciler) fail(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, reason, message string) error {
	if r.Recorder != nil {
		r.Recorder.Event(worker, corev1.EventTypeWarning, reason, message)
	}
	return r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
		s.Phase, s.Reason = openrlv1alpha1.PhaseFailed, message
		setCondition(s, metav1.ConditionFalse, reason, message)
	})
}

// expired reports whether this worker has been waiting past the point where
// "not yet" should be called "no".
func (r *OpenRLWorkerReconciler) expired(worker *openrlv1alpha1.OpenRLWorker) bool {
	if r.PlacementTimeout <= 0 {
		return false
	}
	since := worker.CreationTimestamp.Time
	if condition := apimeta.FindStatusCondition(worker.Status.Conditions, openrlv1alpha1.ConditionPlaced); condition != nil {
		since = condition.LastTransitionTime.Time
	}
	return time.Since(since) > r.PlacementTimeout
}

func setCondition(status *openrlv1alpha1.OpenRLWorkerStatus, state metav1.ConditionStatus, reason, message string) {
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
func (r *OpenRLWorkerReconciler) SetupWithManager(mgr ctrl.Manager) error {
	r.reader = mgr.GetAPIReader()

	if err := mgr.Add(manager.RunnableFunc(r.runReclaim)); err != nil {
		return err
	}

	// Capacity changes are fleet-wide: a freed claim might unblock any
	// pending worker, so these events wake every worker rather than one.
	wakeAll := handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, _ client.Object) []reconcile.Request {
		var workers openrlv1alpha1.OpenRLWorkerList
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
		For(&openrlv1alpha1.OpenRLWorker{}).
		Owns(&corev1.Pod{}).
		Watches(&resourcev1.ResourceClaim{}, wakeAll, builder.WithPredicates(managedClaims())).
		Watches(&corev1.Node{}, wakeAll, builder.WithPredicates(nodeCapacityChanged)).
		WithOptions(controller.Options{MaxConcurrentReconciles: 1}).
		Complete(r)
}
