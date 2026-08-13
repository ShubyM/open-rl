package controller

import (
	"context"
	"strconv"
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

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
)

const (
	testNamespace = "open-rl"
	testDriver    = "gpu.nvidia.com"
	testNode      = "node-a"
	testTemplate  = "trainer-pod-template"
)

// podTemplateYAML is deliberately hostile to the controller: it pins a
// ResourceClaim and a nodeSelector of its own. Both must be overwritten, since
// picking hardware is the whole job of this controller.
const podTemplateYAML = `
apiVersion: v1
kind: Pod
spec:
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-l4
  resourceClaims:
  - name: gpu
    resourceClaimName: someone-elses-claim
  containers:
  - name: worker
    image: template-image
    env:
    - name: KEEP_ME
      value: "1"
`

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

// enabledNode is a pool the operator has opted in for both roles, described by
// a ResourceSlice with two 96Gi devices. maxWorkers is how many time-sliced
// workers the operator will let share one claim here; 1 means no sharing, which
// is also what an unlabelled node gets.
func enabledNode(maxWorkers int) []client.Object {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testNode,
			Labels: map[string]string{
				NodeLabelEnabled:            "true",
				NodeLabelTrainer:            "true",
				NodeLabelSampler:            "true",
				NodeLabelMaxWorkersPerClaim: strconv.Itoa(maxWorkers),
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

	// The key is deliberately not the controller's "pod.yaml" default: the
	// deployed templates are shared with the static worker manager and keep
	// its key names, so every test also exercises the lone-key fallback.
	return []client.Object{node, slice, &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: testTemplate, Namespace: testNamespace},
		Data:       map[string]string{"trainer-worker-pod.yaml": podTemplateYAML},
	}}
}

// worker is the whole of a request: a role, an id, how much accelerator memory
// it needs, and -- if it shares weights with anyone -- the owner it shares
// them with. Everything else the scheduler derives.
func worker(name, modelID string, role openrlv1alpha1.WorkerRole, memory string) *openrlv1alpha1.OpenRLWorker {
	return &openrlv1alpha1.OpenRLWorker{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         testNamespace,
			CreationTimestamp: metav1.Now(),
		},
		Spec: openrlv1alpha1.OpenRLWorkerSpec{
			Role:    role,
			ModelID: modelID,
			Memory:  resource.MustParse(memory),
		},
	}
}

// trainerWorker shares nothing, so it is an owner of one: it competes for
// turns alone against every other worker on its claim.
func trainerWorker(name, modelID string) *openrlv1alpha1.OpenRLWorker {
	return worker(name, modelID, openrlv1alpha1.RoleTrainer, "24Gi")
}

// ownedWorker names the base model it serves. Workers naming the same owner
// share one fairness slot; they still take turns one at a time.
func ownedWorker(name, modelID, owner string) *openrlv1alpha1.OpenRLWorker {
	w := worker(name, modelID, openrlv1alpha1.RoleTrainer, "24Gi")
	w.Spec.OwnerID = owner
	return w
}

// fillerWorker occupies one whole 96Gi device, forcing later workers to
// contend for the other.
func fillerWorker(name, modelID string) *openrlv1alpha1.OpenRLWorker {
	return worker(name, modelID, openrlv1alpha1.RoleTrainer, "90Gi")
}

func newReconciler(t *testing.T, objects ...client.Object) *OpenRLWorkerReconciler {
	t.Helper()
	c := fake.NewClientBuilder().
		WithScheme(testScheme(t)).
		WithObjects(objects...).
		WithStatusSubresource(&openrlv1alpha1.OpenRLWorker{}).
		Build()
	return &OpenRLWorkerReconciler{
		Client:       c,
		Namespace:    testNamespace,
		DeviceClass:  testDriver,
		DeviceDriver: testDriver,
		DefaultPodTemplates: map[openrlv1alpha1.WorkerRole]string{
			openrlv1alpha1.RoleTrainer: testTemplate,
			openrlv1alpha1.RoleSampler: testTemplate,
		},
		RetryInterval:    time.Second,
		PlacementTimeout: time.Hour,
		ReclaimInterval:  time.Minute,
	}
}

func runReconcile(t *testing.T, r *OpenRLWorkerReconciler, name string) ctrl.Result {
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
func settle(t *testing.T, r *OpenRLWorkerReconciler, names ...string) {
	t.Helper()
	for _, name := range names {
		runReconcile(t, r, name)
		runReconcile(t, r, name)
	}
}

func getWorker(t *testing.T, r *OpenRLWorkerReconciler, name string) *openrlv1alpha1.OpenRLWorker {
	t.Helper()
	var w openrlv1alpha1.OpenRLWorker
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: name}, &w); err != nil {
		t.Fatalf("get worker %s: %v", name, err)
	}
	return &w
}

func claimOf(t *testing.T, r *OpenRLWorkerReconciler, name string) string {
	t.Helper()
	claim := getWorker(t, r, name).Status.ClaimName
	if claim == "" {
		t.Fatalf("worker %s was not placed", name)
	}
	return claim
}

func getPod(t *testing.T, r *OpenRLWorkerReconciler, name string) *corev1.Pod {
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
func allocateClaim(t *testing.T, r *OpenRLWorkerReconciler, name string) {
	t.Helper()
	var claim resourcev1.ResourceClaim
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: name}, &claim); err != nil {
		t.Fatalf("get claim %s: %v", name, err)
	}
	claim.Status.Allocation = &resourcev1.AllocationResult{
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
	r := newReconciler(t, append(enabledNode(4), trainerWorker("w-a", "model-a"))...)

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
	if claim.Labels[LabelDeviceMemory] != "96Gi" {
		t.Errorf("device-memory label = %q, want the sized device's 96Gi", claim.Labels[LabelDeviceMemory])
	}
	// The CEL carries both bounds: the floor is the worker's share, the
	// ceiling is the device size the claim was priced against, so DRA cannot
	// satisfy this claim with a bigger device placement never chose.
	cel := claim.Spec.Devices.Requests[0].Exactly.Selectors[0].CEL.Expression
	if !strings.Contains(cel, `quantity("24Gi")) >= 0`) || !strings.Contains(cel, `quantity("96Gi")) <= 0`) {
		t.Errorf("claim CEL = %q, want a 24Gi floor and a 96Gi ceiling", cel)
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

	// The group is the claim -- not a cluster-wide "trainers" bucket -- and a
	// worker that named no owner is an owner of one: it competes for turns
	// alone, under its own name.
	if pod.Labels[timeSliceEnabledLabel] != "true" || pod.Labels[timeSliceGroupLabel] != claimName {
		t.Errorf("time-slice labels = %v, want enabled with group %q", pod.Labels, claimName)
	}
	if got := pod.Labels[timeSliceOwnerLabel]; got != "w-a" {
		t.Errorf("owner label = %q, want the worker's own name", got)
	}
	container := pod.Spec.Containers[0]
	if got := envOf(container, timeSliceGroupEnv); got != claimName {
		t.Errorf("%s = %q, want %q", timeSliceGroupEnv, got, claimName)
	}
	if got := envOf(container, timeSliceOwnerEnv); got != "w-a" {
		t.Errorf("%s = %q, want %q", timeSliceOwnerEnv, got, "w-a")
	}
	if got := envOf(container, "KEEP_ME"); got != "1" {
		t.Errorf("the template's own env was dropped: %v", container.Env)
	}
}

// The device count is derived from memory, never asked for. There is no
// sharding: the model is laid out layer by layer over whatever it is given, so
// 120Gi on 96Gi devices needs two of them and half the model sits on each.
func TestReconcileDerivesTheDeviceCountFromMemory(t *testing.T) {
	big := worker("w-big", "model-big", openrlv1alpha1.RoleTrainer, "120Gi")
	r := newReconciler(t, append(enabledNode(4), big)...)

	runReconcile(t, r, "w-big")

	status := getWorker(t, r, "w-big").Status
	if status.DeviceCount != 2 {
		t.Errorf("deviceCount = %d, want 2: 120Gi does not fit one 96Gi device", status.DeviceCount)
	}
	if status.MemoryPerDevice != "60Gi" {
		t.Errorf("memoryPerDevice = %q, want 60Gi", status.MemoryPerDevice)
	}
	// Parking moves the whole footprint to host RAM, however it was spread.
	if status.HostMemoryWhenParked != "120Gi" {
		t.Errorf("hostMemoryWhenParked = %q, want 120Gi", status.HostMemoryWhenParked)
	}
}

// The regression this test exists for: spec.resourceClaims is immutable, so a
// worker re-placed onto a different claim can never be reached by its existing
// pod. The controller has to delete it, and the next pass has to build one
// against the new claim.
func TestReconcileRecreatesPodBoundToAStaleClaim(t *testing.T) {
	w := trainerWorker("w-a", "model-a")
	w.Status = openrlv1alpha1.OpenRLWorkerStatus{
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
	r := newReconciler(t, append(enabledNode(4), w, pod)...)

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
				APIVersion: "openrl.io/v1alpha1", Kind: "OpenRLWorker",
				Name: "w-a", UID: "old-incarnation", Controller: &isController,
			}},
		},
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "worker", Image: "template-image"}}},
	}
	r := newReconciler(t, append(enabledNode(4), w, old)...)

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
	r := newReconciler(t, append(enabledNode(4), trainerWorker("w-a", "model-a"))...)

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
	r := newReconciler(t, append(enabledNode(4),
		trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"), trainerWorker("w-c", "model-c"))...)

	runReconcile(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	runReconcile(t, r, "w-b")
	allocateClaim(t, r, claimOf(t, r, "w-b"))
	runReconcile(t, r, "w-c")

	a, b, c := claimOf(t, r, "w-a"), claimOf(t, r, "w-b"), claimOf(t, r, "w-c")
	if a == b {
		t.Errorf("both trainers landed on claim %q while a device was still free", a)
	}
	if c != a && c != b {
		t.Errorf("third worker cut claim %q, but both devices were taken and it should have shared", c)
	}

	var claims resourcev1.ResourceClaimList
	if err := r.List(context.Background(), &claims, client.InNamespace(testNamespace)); err != nil {
		t.Fatalf("list claims: %v", err)
	}
	if len(claims.Items) != 2 {
		t.Errorf("cut %d claims, want 2: one per device, and the third worker sharing", len(claims.Items))
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
	r := newReconciler(t, append(enabledNode(4), filler, trainer, sampler)...)

	settle(t, r, "w-x")
	allocateClaim(t, r, claimOf(t, r, "w-x"))
	settle(t, r, "w-t")
	allocateClaim(t, r, claimOf(t, r, "w-t"))
	settle(t, r, "w-s")

	claim := claimOf(t, r, "w-t")
	if got := claimOf(t, r, "w-s"); got != claim {
		t.Fatalf("sampler landed on %q and trainer on %q, want one claim", got, claim)
	}

	// Same group, different owners: they share the bundle, not the weights.
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
	r := newReconciler(t, append(enabledNode(4), workers...)...)

	settle(t, r, "w-x")
	allocateClaim(t, r, claimOf(t, r, "w-x"))
	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	settle(t, r, "w-b", "w-c")

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

// The same two trainers on a pool that seats one worker per claim get a claim
// each. openrl.io/max-workers-per-claim is what turns sharing on, and a node
// without the label defaults to no sharing rather than to unlimited.
func TestReconcileDoesNotShareWhenTheNodeSeatsOneResident(t *testing.T) {
	r := newReconciler(t, append(enabledNode(1), trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"))...)

	runReconcile(t, r, "w-a")
	runReconcile(t, r, "w-b")

	if a, b := claimOf(t, r, "w-a"), claimOf(t, r, "w-b"); a == b {
		t.Errorf("both workers landed on claim %q, but the pool seats one resident", a)
	}
}

// Deleting a worker frees its seat only when its pod is verifiably gone:
// the finalizer holds the CR -- and with it the real memory booking --
// through the pod's termination grace, so max-workers-per-claim cannot break
// for the width of the garbage-collection window.
func TestDeletedWorkerHoldsItsSeatUntilThePodIsGone(t *testing.T) {
	r := newReconciler(t, append(enabledNode(1),
		trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"), trainerWorker("w-c", "model-c"))...)

	settle(t, r, "w-a")
	claimA := claimOf(t, r, "w-a")
	allocateClaim(t, r, claimA)
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
	// Both devices claimed, both single seats booked (w-a's still counts):
	// w-c has nowhere to go while the process drains.
	runReconcile(t, r, "w-c")
	if phase := getWorker(t, r, "w-c").Status.Phase; phase != openrlv1alpha1.PhasePending {
		t.Fatalf("w-c is %q, want Pending: the terminating worker still holds its seat", phase)
	}

	// The process exits: the pod goes, then the worker, then the seat.
	pod = getPod(t, r, "orw-w-a")
	pod.Finalizers = nil
	if err := r.Update(context.Background(), pod); err != nil {
		t.Fatal(err)
	}
	runReconcile(t, r, "w-a")
	var gone openrlv1alpha1.OpenRLWorker
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: "w-a"}, &gone); !apierrors.IsNotFound(err) {
		t.Fatalf("worker still present after its pod died: %v", err)
	}
	runReconcile(t, r, "w-c")
	if got := claimOf(t, r, "w-c"); got != claimA {
		t.Fatalf("w-c landed on %q, want the freed seat on %q", got, claimA)
	}
}

// Zero or negative memory is a broken request, not a free placement.
func TestReconcileFailsNonPositiveMemory(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4), worker("w-a", "model-a", openrlv1alpha1.RoleTrainer, "0"))...)

	runReconcile(t, r, "w-a")

	after := getWorker(t, r, "w-a")
	if after.Status.Phase != openrlv1alpha1.PhaseFailed {
		t.Fatalf("phase = %q, want Failed for memory: 0", after.Status.Phase)
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
	r := newReconciler(t, append(enabledNode(4), worker("w-a", "model-a", openrlv1alpha1.RoleTrainer, "4000Gi"))...)

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
