package controller

import (
	"context"
	"fmt"
	"math/rand"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
)

// The placement storm: random fleets, random workload churn, a random DRA,
// random grace expiries, and reconciles racing on goroutines -- with the
// seat-ledger invariants checked after every round. The point is the paths
// no scripted test walks: bursts that over-cut, moves racing teardowns,
// sweeps racing joins. Deterministic per seed; failures name theirs.

const stormSeeds = 12

func TestPlacementStormKeepsTheLedgerSound(t *testing.T) {
	seeds := stormSeeds
	if testing.Short() {
		seeds = 3
	}
	for seed := 0; seed < seeds; seed++ {
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			runStorm(t, rand.New(rand.NewSource(int64(seed))))
		})
	}
}

// pool is the sim's view of one node: what DRA and kube-scheduler know.
type pool struct {
	name       string
	sizeGiB    int64
	devices    int
	maxWorkers int
}

func makePool(p pool) []client.Object {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: p.name,
			Labels: map[string]string{
				NodeLabelEnabled:            "true",
				NodeLabelTrainer:            "true",
				NodeLabelSampler:            "true",
				NodeLabelMaxWorkersPerClaim: strconv.Itoa(p.maxWorkers),
			},
		},
		Status: corev1.NodeStatus{
			Allocatable: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("4Ti")},
		},
	}
	nodeName := p.name
	devices := make([]resourcev1.Device, p.devices)
	for i := range devices {
		devices[i] = resourcev1.Device{
			Name: fmt.Sprintf("gpu-%d", i),
			Capacity: map[resourcev1.QualifiedName]resourcev1.DeviceCapacity{
				"memory": {Value: resource.MustParse(fmt.Sprintf("%dGi", p.sizeGiB))},
			},
		}
	}
	slice := &resourcev1.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: "slice-" + p.name},
		Spec: resourcev1.ResourceSliceSpec{
			Driver:   testDriver,
			NodeName: &nodeName,
			Pool:     resourcev1.ResourcePool{Name: p.name, ResourceSliceCount: 1},
			Devices:  devices,
		},
	}
	return []client.Object{node, slice}
}

func randomFleet(rng *rand.Rand) []pool {
	sizes := []int64{15, 23, 80}
	n := 1 + rng.Intn(3)
	pools := make([]pool, n)
	for i := range pools {
		pools[i] = pool{
			name:       fmt.Sprintf("node-%d", i),
			sizeGiB:    sizes[rng.Intn(len(sizes))],
			devices:    1 + rng.Intn(3),
			maxWorkers: 1 + rng.Intn(4),
		}
	}
	return pools
}

var celGi = regexp.MustCompile(`quantity\("(\d+)Gi"\)`)

// playDRA allocates pending claims the way the real allocator would: walk
// each claim's tiers in order and satisfy the first whose bounds admit a
// pool with enough free devices. Randomized claim order, so allocation
// interleaves differently per seed.
func playDRA(t *testing.T, r *OpenRLWorkloadReconciler, rng *rand.Rand, pools []pool) {
	t.Helper()
	ctx := context.Background()
	var claims resourcev1.ResourceClaimList
	if err := r.List(ctx, &claims, client.InNamespace(testNamespace)); err != nil {
		t.Fatal(err)
	}

	used := map[string]int{}
	for i := range claims.Items {
		if claims.Items[i].Status.Allocation != nil {
			used[allocatedNode(&claims.Items[i])] += len(claims.Items[i].Status.Allocation.Devices.Results)
		}
	}

	order := rng.Perm(len(claims.Items))
	for _, idx := range order {
		claim := &claims.Items[idx]
		if claim.Status.Allocation != nil || len(claim.Spec.Devices.Requests) == 0 {
			continue
		}
	tiers:
		for _, tier := range claim.Spec.Devices.Requests[0].FirstAvailable {
			bounds := celGi.FindAllStringSubmatch(tier.Selectors[0].CEL.Expression, 2)
			floor, _ := strconv.ParseInt(bounds[0][1], 10, 64)
			ceiling, _ := strconv.ParseInt(bounds[1][1], 10, 64)
			for _, p := range pools {
				if p.sizeGiB < floor || p.sizeGiB > ceiling {
					continue
				}
				if p.devices-used[p.name] < int(tier.Count) {
					continue
				}
				results := make([]resourcev1.DeviceRequestAllocationResult, tier.Count)
				for i := range results {
					results[i] = resourcev1.DeviceRequestAllocationResult{
						Request: "gpu/" + tier.Name, Driver: testDriver, Pool: p.name, Device: fmt.Sprintf("gpu-%d", used[p.name]+i),
					}
				}
				claim.Status.Allocation = &resourcev1.AllocationResult{
					Devices: resourcev1.DeviceAllocationResult{Results: results},
					NodeSelector: &corev1.NodeSelector{NodeSelectorTerms: []corev1.NodeSelectorTerm{{
						MatchFields: []corev1.NodeSelectorRequirement{{
							Key: "metadata.name", Operator: corev1.NodeSelectorOpIn, Values: []string{p.name},
						}},
					}}},
				}
				if err := r.Status().Update(ctx, claim); err != nil {
					if isBenignRaceError(err) {
						// The sweep or a teardown deleted the claim between
						// the list and this write: DRA loses that race in
						// real clusters too.
						break tiers
					}
					t.Fatal(err)
				}
				used[p.name] += int(tier.Count)
				break tiers
			}
		}
	}
}

// reconcileAll drives every named worker once, on goroutines when asked --
// the fake-client stand-in for MaxConcurrentReconciles. Distinct keys only,
// as controller-runtime guarantees.
func reconcileAll(t *testing.T, r *OpenRLWorkloadReconciler, rng *rand.Rand, names []string, concurrent bool) {
	t.Helper()
	order := rng.Perm(len(names))
	if !concurrent {
		for _, i := range order {
			reconcileIgnoringConflicts(t, r, names[i])
		}
		return
	}
	var wg sync.WaitGroup
	for _, i := range order {
		wg.Add(1)
		go func(name string) {
			defer wg.Done()
			reconcileIgnoringConflicts(t, r, name)
		}(names[i])
	}
	wg.Wait()
}

// reconcileIgnoringConflicts is runReconcile minus the fatality: under
// deliberate races, optimistic-concurrency conflicts are the design working,
// and the next round retries. Anything else still fails the test.
func reconcileIgnoringConflicts(t *testing.T, r *OpenRLWorkloadReconciler, name string) {
	t.Helper()
	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: testNamespace, Name: name},
	})
	if err != nil && !isBenignRaceError(err) {
		t.Errorf("reconcile %s: %v", name, err)
	}
}

func isBenignRaceError(err error) bool {
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "conflict") || strings.Contains(msg, "not found") || strings.Contains(msg, "already exists")
}

func runStorm(t *testing.T, rng *rand.Rand) {
	pools := randomFleet(rng)
	var objects []client.Object
	for _, p := range pools {
		objects = append(objects, makePool(p)...)
	}
	r := newReconciler(t, objects...)
	r.ScaleOutGracePeriod = time.Hour // moves fire only when the storm backdates a claim

	ctx := context.Background()
	live := map[string]bool{}
	nextID := 0
	memories := []string{"5Gi", "10Gi", "22Gi", "60Gi", "90Gi"}

	for round := 0; round < 40; round++ {
		// Arrivals: a burst of 0-3 workers.
		for range rng.Intn(4) {
			name := fmt.Sprintf("w-%d", nextID)
			nextID++
			w := worker(name, "model-"+name, openrlv1alpha1.RoleTrainer, memories[rng.Intn(len(memories))])
			if rng.Intn(3) == 0 {
				w.Spec.Accelerator.MaxDeviceCount = 2
			}
			if rng.Intn(3) == 0 {
				w.Spec.OwnerID = fmt.Sprintf("owner-%d", rng.Intn(3))
			}
			if err := r.Create(ctx, w); err != nil {
				t.Fatal(err)
			}
			live[name] = true
		}

		// Departures: sometimes a worker leaves mid-anything.
		if len(live) > 0 && rng.Intn(3) == 0 {
			for name := range live {
				_ = r.Delete(ctx, &openrlv1alpha1.OpenRLWorkload{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: testNamespace}})
				delete(live, name)
				break
			}
		}

		// The grace clock: age a random pending claim past it.
		if rng.Intn(2) == 0 {
			var claims resourcev1.ResourceClaimList
			if err := r.List(ctx, &claims, client.InNamespace(testNamespace)); err != nil {
				t.Fatal(err)
			}
			for i := range claims.Items {
				if claims.Items[i].Status.Allocation == nil && rng.Intn(2) == 0 {
					claims.Items[i].CreationTimestamp = metav1.NewTime(time.Now().Add(-2 * r.ScaleOutGracePeriod))
					_ = r.Update(ctx, &claims.Items[i])
				}
			}
		}

		stampUnschedulable(t, r, rng)
		names := allWorkerNames(t, r)
		reconcileAll(t, r, rng, names, rng.Intn(2) == 0)
		if rng.Intn(2) == 0 {
			playDRA(t, r, rng, pools)
		}
		if rng.Intn(3) == 0 {
			if err := r.reclaimIdleClaims(ctx); err != nil {
				t.Fatal(err)
			}
		}
		checkLedger(t, r, round)
		if t.Failed() {
			return
		}
	}

	// Quiescence: let everything settle, then hold the strong invariants.
	for range 8 {
		stampUnschedulable(t, r, rng)
		names := allWorkerNames(t, r)
		reconcileAll(t, r, rng, names, false)
		playDRA(t, r, rng, pools)
		_ = r.reclaimIdleClaims(ctx)
	}
	checkLedger(t, r, -1)
	checkQuiescent(t, r)
}

// stampUnschedulable plays kube-scheduler's verdict on every pending pod --
// the precondition for both the sharing fallback and the wedged-claim
// abandon. A quarter of verdicts are backdated past any grace, so abandons
// fire during the storm too.
func stampUnschedulable(t *testing.T, r *OpenRLWorkloadReconciler, rng *rand.Rand) {
	t.Helper()
	var pods corev1.PodList
	if err := r.List(context.Background(), &pods, client.InNamespace(testNamespace)); err != nil {
		t.Fatal(err)
	}
	for i := range pods.Items {
		pod := &pods.Items[i]
		if (pod.Status.Phase != "" && pod.Status.Phase != corev1.PodPending) || unschedulableMessage(pod) != "" {
			continue
		}
		since := metav1.Now()
		if rng.Intn(4) == 0 {
			since = metav1.NewTime(time.Now().Add(-3 * time.Hour))
		}
		pod.Status.Phase = corev1.PodPending
		pod.Status.Conditions = append(pod.Status.Conditions, corev1.PodCondition{
			Type: corev1.PodScheduled, Status: corev1.ConditionFalse, Message: "sim: no fit", LastTransitionTime: since,
		})
		_ = r.Status().Update(context.Background(), pod)
	}
}

func allWorkerNames(t *testing.T, r *OpenRLWorkloadReconciler) []string {
	t.Helper()
	var workers openrlv1alpha1.OpenRLWorkloadList
	if err := r.List(context.Background(), &workers, client.InNamespace(testNamespace)); err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(workers.Items))
	for i := range workers.Items {
		names = append(names, workers.Items[i].Name)
	}
	return names
}

// checkLedger holds the invariants that must be true between rounds, however
// the storm interleaved.
func checkLedger(t *testing.T, r *OpenRLWorkloadReconciler, round int) {
	t.Helper()
	ctx := context.Background()

	var groups openrlv1alpha1.OpenRLClaimGroupList
	if err := r.List(ctx, &groups, client.InNamespace(testNamespace)); err != nil {
		t.Fatal(err)
	}
	var workers openrlv1alpha1.OpenRLWorkloadList
	if err := r.List(ctx, &workers, client.InNamespace(testNamespace)); err != nil {
		t.Fatal(err)
	}
	var claims resourcev1.ResourceClaimList
	if err := r.List(ctx, &claims, client.InNamespace(testNamespace)); err != nil {
		t.Fatal(err)
	}
	claimNames := map[string]bool{}
	for i := range claims.Items {
		claimNames[claims.Items[i].Name] = true
	}
	workerByName := map[string]*openrlv1alpha1.OpenRLWorkload{}
	for i := range workers.Items {
		workerByName[workers.Items[i].Name] = &workers.Items[i]
	}

	seatsOf := map[string][]string{} // workload -> groups seating it
	for g := range groups.Items {
		group := &groups.Items[g]
		if group.Spec.MaxSeats > 0 && len(group.Spec.Seats) > int(group.Spec.MaxSeats) {
			t.Errorf("round %d: group %s holds %d seats over its ceiling %d",
				round, group.Name, len(group.Spec.Seats), group.Spec.MaxSeats)
		}
		if groupNameFor(group.Spec.ClaimName) != group.Name {
			t.Errorf("round %d: group %s pairs with claim %s, names drifted", round, group.Name, group.Spec.ClaimName)
		}
		if len(group.Spec.Seats) > 0 && !claimNames[group.Spec.ClaimName] {
			// Tolerated only briefly: the founder books before creating the
			// claim. By checkpoint time the claim must exist unless the
			// founder is mid-flight; flag it, it should never persist.
			if w := workerByName[group.Spec.Seats[0].Workload]; w == nil {
				t.Errorf("round %d: group %s seats %v but its claim is gone and so is the founder",
					round, group.Name, group.Spec.Seats)
			}
		}
		for _, seat := range group.Spec.Seats {
			seatsOf[seat.Workload] = append(seatsOf[seat.Workload], group.Name)
			w := workerByName[seat.Workload]
			if w == nil {
				t.Errorf("round %d: group %s seats %q, but no such workload exists", round, group.Name, seat.Workload)
				continue
			}
			if string(w.UID) != seat.WorkloadUID && seat.WorkloadUID != "" {
				t.Errorf("round %d: group %s seats %q under UID %s, live workload is %s",
					round, group.Name, seat.Workload, seat.WorkloadUID, w.UID)
			}
		}
	}
	for workload, where := range seatsOf {
		if len(where) > 1 {
			t.Errorf("round %d: workload %s holds seats in %d groups at once: %v", round, workload, len(where), where)
		}
	}
}

// checkQuiescent holds the strong end-state invariants once nothing is in
// flight: status, seat, and pod stamps all tell one story.
func checkQuiescent(t *testing.T, r *OpenRLWorkloadReconciler) {
	t.Helper()
	ctx := context.Background()

	var workers openrlv1alpha1.OpenRLWorkloadList
	if err := r.List(ctx, &workers, client.InNamespace(testNamespace)); err != nil {
		t.Fatal(err)
	}
	for i := range workers.Items {
		w := &workers.Items[i]
		if w.Status.ClaimName == "" {
			continue
		}
		group := getGroup(t, r, groupNameFor(w.Status.ClaimName))
		seat := findSeat(group, w.Name)
		if seat == nil {
			t.Errorf("placed worker %s has no seat on %s", w.Name, group.Name)
			continue
		}
		if seat.AssignmentID != w.Status.AssignmentID {
			t.Errorf("worker %s: status assignment %s, seat says %s", w.Name, w.Status.AssignmentID, seat.AssignmentID)
		}
		pod := getPod(t, r, workerPodName(w))
		if pod == nil {
			continue // next reconcile builds it; not a ledger violation
		}
		if got := envOf(pod.Spec.Containers[0], assignmentIDEnv); got != "" && got != w.Status.AssignmentID {
			t.Errorf("worker %s: pod stamped with assignment %s, status says %s", w.Name, got, w.Status.AssignmentID)
		}
	}
}
