package controller

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	openrlv1alpha1 "github.com/gke-labs/open-rl/scheduler/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

const (
	testNamespace = "open-rl"
	testDriver    = "gpu.nvidia.com"
	testNode      = "node-a"
)

// testPodTemplate is a valid inline template: the API server's half of the
// pod, with no placement-shaped fields. Placement-owned fields are rejected
// at validation, which has its own test.
func testPodTemplate() corev1.PodTemplateSpec {
	return corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name:  "worker",
				Image: "template-image",
				Env:   []corev1.EnvVar{{Name: "KEEP_ME", Value: "1"}},
			}},
		},
	}
}

func testScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatalf("add client-go scheme: %v", err)
	}
	if err := openrlv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add openrl scheme: %v", err)
	}
	return scheme
}

// enabledNode is a pool the operator has opted in for both roles, described
// by a ResourceSlice with two 96Gi devices and 340Gi of allocatable host
// memory -- the one ceiling on how many workers may park here. Test workers
// request no host memory unless built by hungryWorker, so sharing is
// unbounded by default.
func enabledNode() []client.Object {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testNode,
			Labels: map[string]string{
				NodeLabelEnabled: "true",
				NodeLabelTrainer: "true",
				NodeLabelSampler: "true",
			},
		},
		Status: corev1.NodeStatus{
			Allocatable: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("340Gi")},
		},
	}

	nodeName := testNode
	product := "NVIDIA RTX PRO 6000 Blackwell"
	device := func(name string) resourcev1.Device {
		return resourcev1.Device{
			Name: name,
			Attributes: map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{
				"productName": {StringValue: &product},
			},
			Capacity: map[resourcev1.QualifiedName]resourcev1.DeviceCapacity{
				"memory": {Value: resource.MustParse("96Gi")},
			},
		}
	}
	slice := &resourcev1.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: "slice-a"},
		Spec: resourcev1.ResourceSliceSpec{
			Driver:   testDriver,
			NodeName: &nodeName,
			Pool:     resourcev1.ResourcePool{Name: testNode, ResourceSliceCount: 1},
			Devices:  []resourcev1.Device{device("gpu-0"), device("gpu-1")},
		},
	}

	return []client.Object{node, slice}
}

// worker is the whole of a request: a role, an id, how much accelerator memory
// it needs, and -- if it shares weights with anyone -- the owner it shares
// them with. Everything else the scheduler derives.
func worker(name, modelID string, role openrlv1alpha1.WorkerRole, memory string) *openrlv1alpha1.Workload {
	return &openrlv1alpha1.Workload{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         testNamespace,
			CreationTimestamp: metav1.Now(),
		},
		Spec: openrlv1alpha1.WorkloadSpec{
			Role:        role,
			ModelID:     modelID,
			Accelerator: openrlv1alpha1.AcceleratorSpec{Memory: resource.MustParse(memory)},
			Template:    testPodTemplate(),
		},
	}
}

// trainerWorker shares nothing, so it is an owner of one: it competes for
// turns alone against every other worker on its claim.
func trainerWorker(name, modelID string) *openrlv1alpha1.Workload {
	return worker(name, modelID, openrlv1alpha1.RoleTrainer, "24Gi")
}

// ownedWorker names the base model it serves. Workers naming the same owner
// share one fairness slot; they still take turns one at a time.
func ownedWorker(name, modelID, owner string) *openrlv1alpha1.Workload {
	w := worker(name, modelID, openrlv1alpha1.RoleTrainer, "24Gi")
	w.Spec.OwnerID = owner
	return w
}

// fillerWorker occupies one whole 96Gi device, forcing later workers to
// contend for the other.
func fillerWorker(name, modelID string) *openrlv1alpha1.Workload {
	return worker(name, modelID, openrlv1alpha1.RoleTrainer, "90Gi")
}

// hungryWorker requests real host memory, making the node's allocatable --
// the one ceiling on parked workers -- bite in the admission check.
func hungryWorker(name, modelID, hostRequest string) *openrlv1alpha1.Workload {
	w := trainerWorker(name, modelID)
	w.Spec.Template.Spec.Containers[0].Resources.Requests = corev1.ResourceList{
		corev1.ResourceMemory: resource.MustParse(hostRequest),
	}
	return w
}

func newReconciler(t *testing.T, objects ...client.Object) *WorkloadReconciler {
	t.Helper()
	c := fake.NewClientBuilder().
		WithScheme(testScheme(t)).
		WithObjects(objects...).
		WithStatusSubresource(&openrlv1alpha1.Workload{}, &corev1.Pod{}).
		Build()
	return &WorkloadReconciler{
		Client:           c,
		Namespace:        testNamespace,
		DeviceClass:      testDriver,
		DeviceDriver:     testDriver,
		RetryInterval:    time.Second,
		PlacementTimeout: time.Hour,
		// The scripted tests reason about one dedicated claim per worker, so
		// they pin spread. Binpack tests set it explicitly; the storm runs both.
		PlacementStrategy: placement.StrategySpread,
	}
}

func runReconcile(t *testing.T, r *WorkloadReconciler, name string) ctrl.Result {
	t.Helper()
	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: testNamespace, Name: name},
	})
	if err != nil {
		t.Fatalf("reconcile %s: %v", name, err)
	}
	return result
}

// settle places the worker and creates its pod: the claim is cut on one pass
// and the pod built on the next.
func settle(t *testing.T, r *WorkloadReconciler, names ...string) {
	t.Helper()
	for _, name := range names {
		runReconcile(t, r, name)
		runReconcile(t, r, name)
	}
}

// fallBackToSharing plays kube-scheduler declining the pod -- the one
// trigger for the sharing fallback -- and reconciles twice: once to move the
// seat and drop the stale pod, once to rebuild it on the shared claim. If no
// claim is joinable the worker stays dedicated.
func fallBackToSharing(t *testing.T, r *WorkloadReconciler, name string) {
	t.Helper()
	markUnschedulable(t, r, name, time.Now())
	runReconcile(t, r, name)
	runReconcile(t, r, name)
}

// markUnschedulable plays kube-scheduler's verdict: the pending pod cannot
// be placed, and has been refused since the given time.
func markUnschedulable(t *testing.T, r *WorkloadReconciler, name string, since time.Time) {
	t.Helper()
	pod := getPod(t, r, "orw-"+name)
	if pod == nil {
		return
	}
	pod.Status.Phase = corev1.PodPending
	pod.Status.Conditions = []corev1.PodCondition{{
		Type: corev1.PodScheduled, Status: corev1.ConditionFalse,
		Reason: "Unschedulable", Message: "0 nodes fit", LastTransitionTime: metav1.NewTime(since),
	}}
	if err := r.Status().Update(context.Background(), pod); err != nil {
		t.Fatal(err)
	}
}

func getWorker(t *testing.T, r *WorkloadReconciler, name string) *openrlv1alpha1.Workload {
	t.Helper()
	var w openrlv1alpha1.Workload
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: name}, &w); err != nil {
		t.Fatalf("get worker %s: %v", name, err)
	}
	return &w
}

func claimOf(t *testing.T, r *WorkloadReconciler, name string) string {
	t.Helper()
	claim := getWorker(t, r, name).Status.ClaimName
	if claim == "" {
		t.Fatalf("worker %s was not placed", name)
	}
	return claim
}

func getPod(t *testing.T, r *WorkloadReconciler, name string) *corev1.Pod {
	t.Helper()
	var pod corev1.Pod
	err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: name}, &pod)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		t.Fatalf("get pod %s: %v", name, err)
	}
	return &pod
}

func envOf(container corev1.Container, name string) string {
	for _, env := range container.Env {
		if env.Name == name {
			return env.Value
		}
	}
	return ""
}

// allocateClaim plays DRA: pins the claim to the test node, which is what
// makes it joinable. Sharing decisions only ever run against allocated claims.
func allocateClaim(t *testing.T, r *WorkloadReconciler, name string) {
	t.Helper()
	var claim resourcev1.ResourceClaim
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: name}, &claim); err != nil {
		t.Fatalf("get claim %s: %v", name, err)
	}
	claim.Status.Allocation = &resourcev1.AllocationResult{
		// One device: the allocated shape is read from these results, and
		// only single-device claims are joinable.
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
	}
	if err := r.Update(context.Background(), &claim); err != nil {
		t.Fatalf("allocate claim %s: %v", name, err)
	}
}

// requiresNodeLabels reports whether some affinity term demands exactly these
// label keys set to "true".
func requiresNodeLabels(pod *corev1.Pod, keys ...string) bool {
	if pod.Spec.Affinity == nil || pod.Spec.Affinity.NodeAffinity == nil ||
		pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		return false
	}
	for _, term := range pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms {
		matched := 0
		for _, want := range keys {
			for _, expr := range term.MatchExpressions {
				if expr.Key == want && expr.Operator == corev1.NodeSelectorOpIn && len(expr.Values) == 1 && expr.Values[0] == "true" {
					matched++
				}
			}
		}
		if matched == len(keys) && len(term.MatchExpressions) == len(keys) {
			return true
		}
	}
	return false
}

// A worker with nothing placed yet gets a claim and a pod, and the pod is
// rendered against the claim rather than against whatever the template said.
func TestReconcilePlacesUnplacedWorker(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	runReconcile(t, r, "w-a")

	placed := getWorker(t, r, "w-a")
	if placed.Status.Phase != openrlv1alpha1.PhasePlacing {
		t.Fatalf("phase = %q, want Placing", placed.Status.Phase)
	}
	claimName := placed.Status.ClaimName
	if claimName == "" {
		t.Fatal("no claim recorded on status")
	}

	var claim resourcev1.ResourceClaim
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: claimName}, &claim); err != nil {
		t.Fatalf("claim %s was not created: %v", claimName, err)
	}
	if claim.Labels[LabelManaged] != "true" {
		t.Errorf("claim is missing the managed label: %v", claim.Labels)
	}
	// The claim states ordered alternatives; on a one-size fleet there is one
	// tier, and its CEL carries both bounds: the floor is the worker's share,
	// the ceiling is the device size the tier was priced against, so DRA
	// cannot satisfy it with a bigger device placement never chose.
	tiers := claim.Spec.Devices.Requests[0].FirstAvailable
	if len(tiers) != 1 || tiers[0].Count != 1 {
		t.Fatalf("firstAvailable = %+v, want one single-device tier", tiers)
	}
	cel := tiers[0].Selectors[0].CEL.Expression
	if !strings.Contains(cel, fmt.Sprintf(`quantity("%d")) >= 0`, 24*placement.GiB)) || !strings.Contains(cel, `quantity("96Gi")) <= 0`) {
		t.Errorf("claim CEL = %q, want a 24Gi floor in exact bytes and a 96Gi ceiling", cel)
	}

	// The pod is created on the pass after the claim, so reconcile again.
	runReconcile(t, r, "w-a")
	pod := getPod(t, r, "orw-w-a")
	if pod == nil {
		t.Fatal("pod was not created")
	}
	if got := pod.Labels[LabelClaim]; got != claimName {
		t.Errorf("pod claim label = %q, want %q", got, claimName)
	}
	if len(pod.Spec.ResourceClaims) != 1 || *pod.Spec.ResourceClaims[0].ResourceClaimName != claimName {
		t.Errorf("pod resourceClaims = %+v, want the template's claim replaced by %q", pod.Spec.ResourceClaims, claimName)
	}
	if len(pod.Spec.NodeSelector) != 0 {
		t.Errorf("nodeSelector = %v, want none: the template's pin is dropped and affinity rules instead", pod.Spec.NodeSelector)
	}
	// One term for explicitly-labeled trainer nodes, one for nodes naming no
	// roles at all -- the documented default that both roles are allowed.
	if !requiresNodeLabels(pod, NodeLabelEnabled, NodeLabelTrainer) {
		t.Errorf("affinity %+v lacks the enabled+trainer term", pod.Spec.Affinity)
	}
	if terms := pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms; len(terms) != 2 {
		t.Errorf("affinity has %d terms, want 2: the second admits role-unlabeled nodes", len(terms))
	}

	// The ledger is the claim -- not a cluster-wide "trainers" bucket -- and a
	// worker that named no owner is an owner of one: it competes for turns
	// alone, under its own name.
	if pod.Labels[timeSliceEnabledLabel] != "true" || pod.Labels[timeSliceGroupLabel] != claimName {
		t.Errorf("time-slice labels = %v, want enabled with ledger %q", pod.Labels, claimName)
	}
	if got := pod.Labels[timeSliceOwnerLabel]; got != "w-a" {
		t.Errorf("owner label = %q, want the worker's own name", got)
	}
	container := pod.Spec.Containers[0]
	if got := envOf(container, timeSliceGroupEnv); got != claimName {
		t.Errorf("%s = %q, want %q", timeSliceGroupEnv, got, claimName)
	}
	// Identity env (owner, workload id) is the template's business now: the
	// API server writes it, the controller stamps only the ledger.
	if got := envOf(container, timeSliceOwnerEnv); got != "" {
		t.Errorf("%s = %q, want unset: identity env rides in the template", timeSliceOwnerEnv, got)
	}
	if got := envOf(container, "KEEP_ME"); got != "1" {
		t.Errorf("the template's own env was dropped: %v", container.Env)
	}
}

// Placement-owned fields in the template are rejected, not merged: an invalid
// template is terminal, because the spec is immutable.
func TestReconcileRejectsPlacementShapedTemplates(t *testing.T) {
	w := trainerWorker("w-pin", "model-a")
	w.Spec.Template.Spec.NodeSelector = map[string]string{"cloud.google.com/gke-accelerator": "nvidia-l4"}
	r := newReconciler(t, append(enabledNode(), w)...)

	runReconcile(t, r, "w-pin")

	status := getWorker(t, r, "w-pin").Status
	if status.Phase != openrlv1alpha1.PhaseFailed || status.Reason == "" {
		t.Fatalf("phase = %s (%s), want Failed with a reason naming the template", status.Phase, status.Reason)
	}
	if !strings.Contains(status.Reason, "nodeSelector") {
		t.Errorf("reason = %q, want it to name the offending field", status.Reason)
	}
}

// Every shipped runtime drives one device, so memory that fits no single
// device is unplaceable: placement never guesses a device count the process
// cannot use. (Multi-device claims return with a runtime that declares them;
// design doc section 12.)
func TestOversizedMemoryStaysPendingOnSingleDeviceRuntimes(t *testing.T) {
	big := worker("w-big", "model-big", openrlv1alpha1.RoleTrainer, "120Gi")
	r := newReconciler(t, append(enabledNode(), big)...)

	runReconcile(t, r, "w-big")

	status := getWorker(t, r, "w-big").Status
	if status.Phase != openrlv1alpha1.PhasePending {
		t.Errorf("phase = %s, want Pending for a 120Gi worker on single-device runtimes", status.Phase)
	}
	if status.ClaimName != "" {
		t.Errorf("claim = %q, want none: no single device fits 120Gi", status.ClaimName)
	}
}

// The regression this test exists for: spec.resourceClaims is immutable, so a
// worker re-placed onto a different claim can never be reached by its existing
// pod. The controller has to delete it, and the next pass has to build one
// against the new claim.
func TestReconcileRecreatesPodBoundToAStaleClaim(t *testing.T) {
	w := trainerWorker("w-a", "model-a")
	w.Status = openrlv1alpha1.WorkloadStatus{
		Phase:     openrlv1alpha1.PhaseRunning,
		ClaimName: "claim-gone",
		PodName:   "orw-w-a",
	}
	stale := "claim-gone"
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "orw-w-a",
			Namespace: testNamespace,
			Labels:    map[string]string{LabelClaim: stale, LabelWorker: "w-a"},
		},
		Spec: corev1.PodSpec{
			Containers:     []corev1.Container{{Name: "worker", Image: "template-image"}},
			ResourceClaims: []corev1.PodResourceClaim{{Name: podClaimName, ResourceClaimName: &stale}},
		},
	}
	r := newReconciler(t, append(enabledNode(), w, pod)...)

	runReconcile(t, r, "w-a")

	if getPod(t, r, "orw-w-a") != nil {
		t.Fatal("the pod bound to the vanished claim was left in place")
	}
	after := getWorker(t, r, "w-a")
	if after.Status.Reason != "RecreatingPodOnNewClaim" {
		t.Errorf("reason = %q, want RecreatingPodOnNewClaim", after.Status.Reason)
	}
	if after.Status.ClaimName == stale || after.Status.ClaimName == "" {
		t.Fatalf("claim = %q, want a freshly cut one", after.Status.ClaimName)
	}

	// Converge: the next pass builds a pod against the claim the worker now holds.
	runReconcile(t, r, "w-a")
	rebuilt := getPod(t, r, "orw-w-a")
	if rebuilt == nil {
		t.Fatal("no pod was rebuilt")
	}
	if got := *rebuilt.Spec.ResourceClaims[0].ResourceClaimName; got != after.Status.ClaimName {
		t.Errorf("rebuilt pod names claim %q, want %q", got, after.Status.ClaimName)
	}
}

// A worker deleted and recreated under the same name must not adopt its
// predecessor's still-terminating pod: the controller ownerRef UID tells the
// incarnations apart. Adoption would inherit a claim seat this CR never
// booked -- the over-admission a delete/recreate storm produces.
func TestReconcileReplacesAPredecessorsPod(t *testing.T) {
	w := trainerWorker("w-a", "model-a")
	w.UID = "new-incarnation"
	isController := true
	old := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "orw-w-a",
			Namespace: testNamespace,
			Labels:    map[string]string{LabelClaim: "claim-old", LabelWorker: "w-a"},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: "openrl.io/v1alpha1", Kind: "Workload",
				Name: "w-a", UID: "old-incarnation", Controller: &isController,
			}},
		},
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "worker", Image: "template-image"}}},
	}
	r := newReconciler(t, append(enabledNode(), w, old)...)

	runReconcile(t, r, "w-a")

	if getPod(t, r, "orw-w-a") != nil {
		t.Fatal("the predecessor's pod was adopted or left in place")
	}
	after := getWorker(t, r, "w-a")
	if after.Status.Reason != "ReplacingPredecessorPod" {
		t.Errorf("reason = %q, want ReplacingPredecessorPod", after.Status.Reason)
	}
	if after.Status.ClaimName == "claim-old" {
		t.Errorf("inherited the predecessor's claim %q", after.Status.ClaimName)
	}
}

// A pod that is already on the right claim is left alone. Without this, the
// delete branch above would restart every worker on every reconcile.
func TestReconcileLeavesAMatchingPodAlone(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	created := getPod(t, r, "orw-w-a")
	if created == nil {
		t.Fatal("pod was not created")
	}

	for i := 0; i < 3; i++ {
		runReconcile(t, r, "w-a")
	}
	still := getPod(t, r, "orw-w-a")
	if still == nil {
		t.Fatal("a settled pod was deleted")
	}
	if still.UID != created.UID {
		t.Errorf("pod was recreated (uid %q -> %q)", created.UID, still.UID)
	}
}

// Spread onto free capacity first: while unclaimed accelerators exist, each
// worker gets its own claim, because sharing under no contention just costs
// throughput. Only when the devices run out does a worker join an existing
// claim.
func TestReconcileSpreadsThenSharesUnderContention(t *testing.T) {
	r := newReconciler(t, append(enabledNode(),
		trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"), trainerWorker("w-c", "model-c"))...)

	runReconcile(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	runReconcile(t, r, "w-b")
	allocateClaim(t, r, claimOf(t, r, "w-b"))
	runReconcile(t, r, "w-c")

	// Everyone cuts a dedicated claim first; sharing is never decided at
	// admission. DRA satisfied a and b; c's stays pending.
	a, b, c := claimOf(t, r, "w-a"), claimOf(t, r, "w-b"), claimOf(t, r, "w-c")
	if a == b || c == a || c == b {
		t.Fatalf("claims %q/%q/%q, want three dedicated claims", a, b, c)
	}

	// kube-scheduler declines c's pod: c falls back to sharing.
	fallBackToSharing(t, r, "w-c")
	if got := claimOf(t, r, "w-c"); got != a && got != b {
		t.Errorf("w-c holds %q after the verdict, want one of the allocated claims", got)
	}

	// The fallback reclaimed the abandoned dedicated claim inline.
	var claims resourcev1.ResourceClaimList
	if err := r.List(context.Background(), &claims, client.InNamespace(testNamespace)); err != nil {
		t.Fatalf("list claims: %v", err)
	}
	if len(claims.Items) != 2 {
		t.Errorf("%d claims survive, want 2: one per device, the abandoned one reclaimed", len(claims.Items))
	}
}

// Under the binpack strategy the order flips: a worker seats on an eligible
// allocated claim before cutting one at all, even while a device sits free.
// Only a worker the node's host memory refuses gets a claim of its own.
func TestReconcileBinPacksOntoExistingClaims(t *testing.T) {
	// 128Gi each on a 340Gi node: two park together, a third does not.
	r := newReconciler(t, append(enabledNode(),
		hungryWorker("w-a", "model-a", "128Gi"), hungryWorker("w-b", "model-b", "128Gi"), hungryWorker("w-c", "model-c", "128Gi"))...)
	r.PlacementStrategy = placement.StrategyBinPack

	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	settle(t, r, "w-b")

	a := claimOf(t, r, "w-a")
	if b := claimOf(t, r, "w-b"); b != a {
		t.Fatalf("w-b landed on %q and w-a on %q, want w-b packed onto w-a's claim despite the free device", b, a)
	}

	// A third 128Gi request will not park on what is left of the node's
	// 340Gi, so w-c falls through to a dedicated claim.
	settle(t, r, "w-c")
	if c := claimOf(t, r, "w-c"); c == a {
		t.Fatal("w-c joined past the node's host memory, want a dedicated claim")
	}

	var claims resourcev1.ResourceClaimList
	if err := r.List(context.Background(), &claims, client.InNamespace(testNamespace)); err != nil {
		t.Fatalf("list claims: %v", err)
	}
	if len(claims.Items) != 2 {
		t.Errorf("%d claims exist, want 2: the packed one and w-c's own", len(claims.Items))
	}
}

// A trainer and a sampler on one accelerator, once the devices are contended.
// Role selects which node pools may host a worker and nothing else -- in
// particular it does not partition claims -- so on a node labelled for both,
// the two halves of the loop take turns on one GPU instead of one of them
// going unplaced.
//
// The filler holds the other device. Both existing claims hold one worker, so
// the tie breaks by name and the sampler joins the trainer's claim.
func TestReconcileRunsATrainerAndASamplerOnOneClaim(t *testing.T) {
	filler := fillerWorker("w-x", "model-x")
	trainer := trainerWorker("w-t", "model-t")
	sampler := worker("w-s", "model-s", openrlv1alpha1.RoleSampler, "24Gi")
	r := newReconciler(t, append(enabledNode(), filler, trainer, sampler)...)

	settle(t, r, "w-x")
	allocateClaim(t, r, claimOf(t, r, "w-x"))
	settle(t, r, "w-t")
	allocateClaim(t, r, claimOf(t, r, "w-t"))
	settle(t, r, "w-s")
	fallBackToSharing(t, r, "w-s")

	claim := claimOf(t, r, "w-t")
	if got := claimOf(t, r, "w-s"); got != claim {
		t.Fatalf("sampler landed on %q and trainer on %q, want one claim", got, claim)
	}

	// Same ledger, different owners: they share the bundle, not the weights.
	trainerPod, samplerPod := getPod(t, r, "orw-w-t"), getPod(t, r, "orw-w-s")
	if trainerPod == nil || samplerPod == nil {
		t.Fatalf("pods missing: trainer=%v sampler=%v", trainerPod != nil, samplerPod != nil)
	}
	if trainerPod.Labels[timeSliceGroupLabel] != samplerPod.Labels[timeSliceGroupLabel] {
		t.Errorf("time-slice groups differ: %q and %q", trainerPod.Labels[timeSliceGroupLabel], samplerPod.Labels[timeSliceGroupLabel])
	}
	if trainerPod.Labels[timeSliceOwnerLabel] == samplerPod.Labels[timeSliceOwnerLabel] {
		t.Errorf("both pods are in owner %q, but they share nothing", trainerPod.Labels[timeSliceOwnerLabel])
	}
	// The sampler still may not land on a pool the operator closed to samplers.
	if !requiresNodeLabels(samplerPod, NodeLabelEnabled, NodeLabelSampler) {
		t.Errorf("sampler affinity %+v lacks the enabled+sampler term", samplerPod.Spec.Affinity)
	}
}

// Workers under contention, two owners between them. The owner is a string
// the caller chooses and the scheduler only ever compares: same string means
// one fairness slot at runtime, different string means separate turns. It
// never shapes placement -- the controller has no table of which workload
// kinds may share, and contention spreads to the claim with the fewest
// assigned workers.
//
// The filler takes the second device, so everything after w-a shares.
func TestReconcileGroupsWorkersByTheirOwnerID(t *testing.T) {
	workers := []client.Object{
		fillerWorker("w-x", "model-x"),
		ownedWorker("w-a", "model-a", "Qwen/Qwen3-0.6B"),
		ownedWorker("w-b", "model-b", "Qwen/Qwen3-0.6B"),
		ownedWorker("w-c", "model-c", "meta-llama/Llama-3-8B"),
	}
	r := newReconciler(t, append(enabledNode(), workers...)...)

	settle(t, r, "w-x")
	allocateClaim(t, r, claimOf(t, r, "w-x"))
	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	settle(t, r, "w-b", "w-c")
	fallBackToSharing(t, r, "w-b")
	fallBackToSharing(t, r, "w-c")

	// w-b joins the least-loaded claim by name; w-c then finds the filler's
	// claim emptier. Owner never enters the placement decision.
	claim := claimOf(t, r, "w-a")
	if b := claimOf(t, r, "w-b"); b != claim {
		t.Fatalf("w-b landed on %q and w-a on %q, want the shared claim", b, claim)
	}
	if c := claimOf(t, r, "w-c"); c != claimOf(t, r, "w-x") {
		t.Fatalf("w-c landed on %q, want the filler's claim -- the one with the fewest workers", c)
	}

	owner := func(name string) string {
		pod := getPod(t, r, "orw-"+name)
		if pod == nil {
			t.Fatalf("no pod for %s", name)
		}
		return pod.Labels[timeSliceOwnerLabel]
	}
	if a, b := owner("w-a"), owner("w-b"); a != b {
		t.Errorf("owners %q and %q differ, but both named the same base model", a, b)
	}
	if a, c := owner("w-a"), owner("w-c"); a == c {
		t.Errorf("both pods are in owner %q, but they serve different base models", a)
	}
}

// Host memory is the one ceiling on sharing: a joiner whose pod request no
// longer fits beside the node's booked seats is turned away and stays on its
// dedicated claim, verdict or no verdict.
func TestReconcileDoesNotShareBeyondHostMemory(t *testing.T) {
	// 200Gi each on a 340Gi node: one parks, two would not.
	r := newReconciler(t, append(enabledNode(),
		hungryWorker("w-a", "model-a", "200Gi"), hungryWorker("w-b", "model-b", "200Gi"))...)

	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	settle(t, r, "w-b")
	dedicated := claimOf(t, r, "w-b")

	fallBackToSharing(t, r, "w-b")
	if got := claimOf(t, r, "w-b"); got != dedicated {
		t.Errorf("w-b moved to %q, but its 200Gi request cannot park beside w-a's on a 340Gi node", got)
	}
}

// Deleting a worker frees its memory booking only when its pod is verifiably
// gone: the finalizer holds the CR -- and with it the seat's host request --
// through the pod's termination grace, so the node's host-memory ceiling
// cannot break for the width of the garbage-collection window.
func TestDeletedWorkerHoldsItsSeatUntilThePodIsGone(t *testing.T) {
	// 120Gi each on a 340Gi node: two park, a third does not.
	r := newReconciler(t, append(enabledNode(),
		hungryWorker("w-a", "model-a", "120Gi"), hungryWorker("w-b", "model-b", "120Gi"), hungryWorker("w-c", "model-c", "120Gi"))...)

	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	settle(t, r, "w-b")
	allocateClaim(t, r, claimOf(t, r, "w-b"))

	// Pin w-a's pod the way a kubelet mid-termination would, then delete the
	// worker. The finalizer keeps the CR while the pod drains.
	pod := getPod(t, r, "orw-w-a")
	pod.Finalizers = append(pod.Finalizers, "test.openrl.io/hold")
	if err := r.Update(context.Background(), pod); err != nil {
		t.Fatal(err)
	}
	if err := r.Delete(context.Background(), getWorker(t, r, "w-a")); err != nil {
		t.Fatal(err)
	}
	runReconcile(t, r, "w-a")

	if getWorker(t, r, "w-a").DeletionTimestamp.IsZero() {
		t.Fatal("the worker should be terminating, held by its finalizer")
	}
	// w-c cuts its own dedicated claim, which DRA never satisfies. Even with
	// the verdict in it has nowhere to fall back to: the terminating worker's
	// seat still books 120Gi, and 360Gi will not park on a 340Gi node.
	settle(t, r, "w-c")
	dedicated := claimOf(t, r, "w-c")
	fallBackToSharing(t, r, "w-c")
	if got := claimOf(t, r, "w-c"); got != dedicated {
		t.Fatalf("w-c moved to %q, but the terminating worker still books its host memory", got)
	}

	// The process exits: the pod goes, then the worker, then the booking --
	// and w-a's seatless claim with them, reclaimed inline.
	pod = getPod(t, r, "orw-w-a")
	pod.Finalizers = nil
	if err := r.Update(context.Background(), pod); err != nil {
		t.Fatal(err)
	}
	runReconcile(t, r, "w-a")
	expectGone(t, r, &openrlv1alpha1.Workload{}, "w-a")
	fallBackToSharing(t, r, "w-c")
	if got, want := claimOf(t, r, "w-c"), claimOf(t, r, "w-b"); got != want {
		t.Fatalf("w-c landed on %q, want %q now that the freed memory admits it", got, want)
	}
}

// A pod whose process exits cleanly is Succeeded, not Running: the seat and
// claim stay held until the owner deletes the workload.
func TestSucceededPodReportsSucceeded(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	pod := getPod(t, r, "orw-w-a")
	pod.Status.Phase = corev1.PodSucceeded
	if err := r.Status().Update(context.Background(), pod); err != nil {
		t.Fatal(err)
	}
	runReconcile(t, r, "w-a")

	if got := getWorker(t, r, "w-a").Status.Phase; got != openrlv1alpha1.PhaseSucceeded {
		t.Fatalf("phase = %q, want Succeeded", got)
	}
	if findSeat(getLedger(t, r, ledgerNameFor(claimOf(t, r, "w-a"))), "w-a") == nil {
		t.Fatal("a succeeded worker lost its seat before deletion")
	}
}

// Zero or negative memory is a broken request, not a free placement.
func TestReconcileFailsNonPositiveMemory(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), worker("w-a", "model-a", openrlv1alpha1.RoleTrainer, "0"))...)

	runReconcile(t, r, "w-a")

	after := getWorker(t, r, "w-a")
	if after.Status.Phase != openrlv1alpha1.PhaseFailed {
		t.Fatalf("phase = %q, want Failed for memory: 0", after.Status.Phase)
	}
}

// A placed worker never expires: the Placed condition's age is time spent
// running, not time spent waiting. When its pod goes unschedulable after a
// long healthy run -- node death, or turnover holding memory between pod
// incarnations -- the worker rides out the wedge grace; it is not declared
// Failed on the first verdict.
func TestLongRunningWorkerSurvivesAnUnschedulableBlip(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	claimName := claimOf(t, r, "w-a")
	allocateClaim(t, r, claimName)

	// Placed well past PlacementTimeout ago, running ever since.
	worker := getWorker(t, r, "w-a")
	worker.Status.Phase = openrlv1alpha1.PhaseRunning
	worker.Status.Conditions = []metav1.Condition{{
		Type: openrlv1alpha1.ConditionPlaced, Status: metav1.ConditionTrue,
		Reason: "Placed", Message: "worker is running on " + claimName,
		LastTransitionTime: metav1.NewTime(time.Now().Add(-2 * r.PlacementTimeout)),
	}}
	if err := r.Status().Update(context.Background(), worker); err != nil {
		t.Fatal(err)
	}

	// The verdict arrived moments ago: a transient, well inside the grace.
	markUnschedulable(t, r, "w-a", time.Now())
	runReconcile(t, r, "w-a")

	after := getWorker(t, r, "w-a")
	if after.Status.Phase == openrlv1alpha1.PhaseFailed {
		t.Fatalf("a transient verdict failed a long-running worker: %s", after.Status.Reason)
	}
	if after.Status.Phase != openrlv1alpha1.PhasePending {
		t.Fatalf("phase = %q, want Pending while the verdict is debounced", after.Status.Phase)
	}
	if after.Status.ClaimName != claimName {
		t.Fatalf("claim = %q, want %q retained through the grace", after.Status.ClaimName, claimName)
	}
}

// The pod carries the tier order as weighted node preferences: the smallest
// adequate pool's name at weight 100, no term for the largest shape -- the
// baseline nothing needs to outrank. firstAvailable orders alternatives
// within a node; these weights are what order the nodes.
func TestPodPrefersTheSmallestAdequateNode(t *testing.T) {
	nodeName := "node-small"
	small := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   nodeName,
			Labels: map[string]string{NodeLabelEnabled: "true", NodeLabelTrainer: "true", NodeLabelSampler: "true"},
		},
		Status: corev1.NodeStatus{Allocatable: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("60Gi")}},
	}
	slice := &resourcev1.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: "slice-small"},
		Spec: resourcev1.ResourceSliceSpec{
			Driver:   testDriver,
			NodeName: &nodeName,
			Pool:     resourcev1.ResourcePool{Name: nodeName, ResourceSliceCount: 1},
			Devices: []resourcev1.Device{{
				Name:     "gpu-0",
				Capacity: map[resourcev1.QualifiedName]resourcev1.DeviceCapacity{"memory": {Value: resource.MustParse("24Gi")}},
			}},
		},
	}
	r := newReconciler(t, append(enabledNode(), small, slice, trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")

	pod := getPod(t, r, "orw-w-a")
	if pod == nil {
		t.Fatal("no pod for w-a")
	}
	terms := pod.Spec.Affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution
	if len(terms) != 1 {
		t.Fatalf("preferred terms = %+v, want exactly one: the small pool", terms)
	}
	if terms[0].Weight != 100 || terms[0].Preference.MatchFields[0].Values[0] != nodeName {
		t.Errorf("term = %+v, want %s at weight 100", terms[0], nodeName)
	}
}

// Claim names derive from the worker's UID -- unique per incarnation, stable
// within one -- and stay label-legal however long the worker's name is.
func TestClaimNamesAreUIDUniqueAndLabelSafe(t *testing.T) {
	first := trainerWorker(strings.Repeat("w", 100), "model-a")
	first.UID = "11111111-aaaa-bbbb-cccc-dddddddddddd"
	name := claimNameFor(first)
	if len(name) > 63 {
		t.Fatalf("claim name %q is %d chars; labels cap at 63", name, len(name))
	}
	if claimNameFor(first) != name {
		t.Error("one incarnation must converge on one claim name")
	}

	reborn := trainerWorker(strings.Repeat("w", 100), "model-a")
	reborn.UID = "22222222-aaaa-bbbb-cccc-dddddddddddd"
	if claimNameFor(reborn) == name {
		t.Error("a recreated worker collided with its predecessor's claim name")
	}
}

// A worker asking for more than any registered pool can provide is reported as
// pending with an explanation, not silently dropped or placed anyway.
func TestReconcileReportsAnUnplaceableWorkerAsPending(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), worker("w-a", "model-a", openrlv1alpha1.RoleTrainer, "4000Gi"))...)

	result := runReconcile(t, r, "w-a")
	if result.RequeueAfter != r.RetryInterval {
		t.Errorf("requeueAfter = %v, want %v", result.RequeueAfter, r.RetryInterval)
	}
	after := getWorker(t, r, "w-a")
	if after.Status.Phase != openrlv1alpha1.PhasePending {
		t.Fatalf("phase = %q, want Pending", after.Status.Phase)
	}
	if after.Status.Reason == "" {
		t.Error("a pending worker should carry an explanation")
	}
	if getPod(t, r, "orw-w-a") != nil {
		t.Error("a pod was created for a worker that was never placed")
	}
}

// The CEL floor is exact bytes: a share rounded up to whole GiB would ask a
// 79.65Gi device for 80Gi whenever the request lands in its last fraction,
// and the claim would never allocate. The ceiling still rounds up.
func TestClaimFloorIsExactBytes(t *testing.T) {
	r := newReconciler(t, enabledNode()...)
	device := 80*placement.GiB - 360*1024*1024 // ~79.65Gi
	floor := 79*placement.GiB + 512*1024*1024  // 79.5Gi: fits the device, rounds up past it
	claim := r.buildClaim("claim-x", []placement.Tier{{Name: "t1x80", Count: 1, FloorBytes: floor, CeilingBytes: device}})

	expr := claim.Spec.Devices.Requests[0].FirstAvailable[0].Selectors[0].CEL.Expression
	if want := fmt.Sprintf(`quantity("%d")) >= 0`, floor); !strings.Contains(expr, want) {
		t.Errorf("floor: %s\nwant it to contain %s", expr, want)
	}
	if want := `quantity("80Gi")) <= 0`; !strings.Contains(expr, want) {
		t.Errorf("ceiling: %s\nwant it to contain %s", expr, want)
	}
}

// A role label set to "false" names the role and denies it, exactly as the
// pod affinity reads it; only a node naming neither role takes both.
func TestRoleLabelFalseDeniesTheRole(t *testing.T) {
	r := newReconciler(t)
	roles := func(labels map[string]string) (trainer, sampler bool) {
		t.Helper()
		objects := enabledNode()
		node := objects[0].(*corev1.Node)
		slice := objects[1].(*resourcev1.ResourceSlice)
		node.Labels = map[string]string{NodeLabelEnabled: "true"}
		for k, v := range labels {
			node.Labels[k] = v
		}
		pool := r.poolsFrom(context.Background(), []resourcev1.ResourceSlice{*slice}, []corev1.Node{*node})[testNode]
		if pool == nil {
			t.Fatal("the enabled node was dropped from the fleet")
		}
		return pool.Accepts("trainer"), pool.Accepts("sampler")
	}

	if tr, sa := roles(nil); !tr || !sa {
		t.Errorf("no role labels: trainer=%v sampler=%v, want both", tr, sa)
	}
	if tr, sa := roles(map[string]string{NodeLabelTrainer: "false", NodeLabelSampler: "true"}); tr || !sa {
		t.Errorf("trainer=false sampler=true: trainer=%v sampler=%v, want sampler only", tr, sa)
	}
	if tr, sa := roles(map[string]string{NodeLabelTrainer: "false"}); tr || sa {
		t.Errorf("trainer=false alone: trainer=%v sampler=%v, want neither -- the affinity admits neither", tr, sa)
	}
}
