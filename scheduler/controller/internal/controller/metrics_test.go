package controller

import (
	"context"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	openrlv1alpha1 "github.com/gke-labs/open-rl/scheduler/controller/api/v1alpha1"
)

// heldGPUs scrapes openrl_gpu_held the way the metrics endpoint does and
// renders each series as "run role pod node UUID=value".
func heldGPUs(t *testing.T, r *WorkloadReconciler) []string {
	t.Helper()
	registry := prometheus.NewPedanticRegistry()
	registry.MustRegister(gpuHeldCollector{reader: r.Client, namespace: testNamespace})
	families, err := registry.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	var series []string
	for _, family := range families {
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			series = append(series, fmt.Sprintf("%s %s %s %s %s=%v",
				labels["run_id"], labels["role"], labels["pod"], labels["node"], labels["UUID"], metric.GetGauge().GetValue()))
		}
	}
	return series
}

// The placement record: an observed allocation stamps the GPUs and the time,
// the metric reports the hold, and teardown closes the interval and drops it.
func TestPlacementRecordsGPUsUntilTeardown(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	if got := heldGPUs(t, r); len(got) != 0 {
		t.Fatalf("series before allocation = %v, want none", got)
	}
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	runReconcile(t, r, "w-a")

	placed := getWorker(t, r, "w-a").Status
	if placed.NodeName != testNode || !slices.Equal(placed.GPUUUIDs, []string{"GPU-gpu-0"}) || placed.PlacedAt == nil || placed.ReleasedAt != nil {
		t.Fatalf("status = node %q uuids %v placedAt %v releasedAt %v, want node-a, GPU-gpu-0, placed, not released",
			placed.NodeName, placed.GPUUUIDs, placed.PlacedAt, placed.ReleasedAt)
	}
	want := []string{"model-a trainer orw-w-a node-a GPU-gpu-0=1"}
	if got := heldGPUs(t, r); !slices.Equal(got, want) {
		t.Fatalf("series = %v, want %v", got, want)
	}
	runReconcile(t, r, "w-a")
	if again := getWorker(t, r, "w-a").Status.PlacedAt; !again.Equal(placed.PlacedAt) {
		t.Fatalf("placedAt moved from %v to %v on a pass that changed nothing", placed.PlacedAt, again)
	}

	// A second finalizer keeps the workload readable after teardown finishes.
	w := getWorker(t, r, "w-a")
	w.Finalizers = append(w.Finalizers, "test.openrl.io/hold")
	if err := r.Update(context.Background(), w); err != nil {
		t.Fatal(err)
	}
	if err := r.Delete(context.Background(), w); err != nil {
		t.Fatal(err)
	}
	runReconcile(t, r, "w-a") // deletes the pod
	if got := heldGPUs(t, r); !slices.Equal(got, want) {
		t.Fatalf("series while the pod terminates = %v, want %v", got, want)
	}
	runReconcile(t, r, "w-a") // pod gone: seats released

	released := getWorker(t, r, "w-a").Status
	if released.ReleasedAt == nil || released.NodeName != testNode || !slices.Equal(released.GPUUUIDs, placed.GPUUUIDs) {
		t.Fatalf("status = node %q uuids %v releasedAt %v, want the placement kept and released",
			released.NodeName, released.GPUUUIDs, released.ReleasedAt)
	}
	if got := heldGPUs(t, r); len(got) != 0 {
		t.Fatalf("series after release = %v, want none", got)
	}
}

// Abandoning a wedged claim releases the placement, and the next allocation
// opens a new one.
func TestAbandonedClaimReleasesThenReplaces(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	runReconcile(t, r, "w-a")
	markUnschedulable(t, r, "w-a", time.Now().Add(-2*wedgeGracePeriod))
	runReconcile(t, r, "w-a")

	released := getWorker(t, r, "w-a").Status
	if released.ReleasedAt == nil || released.NodeName != "" || len(released.GPUUUIDs) != 0 {
		t.Fatalf("status = node %q uuids %v releasedAt %v, want released with no node or GPUs",
			released.NodeName, released.GPUUUIDs, released.ReleasedAt)
	}
	if got := heldGPUs(t, r); len(got) != 0 {
		t.Fatalf("series after abandoning = %v, want none", got)
	}

	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	runReconcile(t, r, "w-a")

	replaced := getWorker(t, r, "w-a").Status
	if replaced.NodeName != testNode || replaced.PlacedAt == nil || replaced.ReleasedAt != nil {
		t.Fatalf("status = node %q placedAt %v releasedAt %v, want placed again and not released",
			replaced.NodeName, replaced.PlacedAt, replaced.ReleasedAt)
	}
	if got := heldGPUs(t, r); len(got) != 1 {
		t.Fatalf("series after re-placing = %v, want one", got)
	}
}

// A placed worker whose claim is deleted out from under it stops holding its
// GPUs as soon as the controller sees the claim gone, whether re-placement
// waits on the old claim's deletion or finds no capacity.
func TestLostClaimReleasesBeforeRePlacement(t *testing.T) {
	cases := map[string]func(t *testing.T, r *WorkloadReconciler, claimName string){
		"old claim still terminating": func(t *testing.T, r *WorkloadReconciler, claimName string) {
			var claim resourcev1.ResourceClaim
			if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: claimName}, &claim); err != nil {
				t.Fatal(err)
			}
			claim.Finalizers = append(claim.Finalizers, "test.openrl.io/hold")
			if err := r.Update(context.Background(), &claim); err != nil {
				t.Fatal(err)
			}
			if err := r.Delete(context.Background(), &claim); err != nil {
				t.Fatal(err)
			}
		},
		"no capacity left": func(t *testing.T, r *WorkloadReconciler, claimName string) {
			claim := &resourcev1.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, Name: claimName}}
			if err := r.Delete(context.Background(), claim); err != nil {
				t.Fatal(err)
			}
			var node corev1.Node
			if err := r.Get(context.Background(), types.NamespacedName{Name: testNode}, &node); err != nil {
				t.Fatal(err)
			}
			node.Labels[NodeLabelEnabled] = "false"
			if err := r.Update(context.Background(), &node); err != nil {
				t.Fatal(err)
			}
		},
	}
	for name, loseClaim := range cases {
		t.Run(name, func(t *testing.T) {
			r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)
			settle(t, r, "w-a")
			claimName := claimOf(t, r, "w-a")
			allocateClaim(t, r, claimName)
			runReconcile(t, r, "w-a")

			loseClaim(t, r, claimName)
			runReconcile(t, r, "w-a")

			status := getWorker(t, r, "w-a").Status
			if status.Phase != openrlv1alpha1.PhasePending || status.ReleasedAt == nil || status.NodeName != "" || len(status.GPUUUIDs) != 0 {
				t.Fatalf("status = phase %q node %q uuids %v releasedAt %v, want Pending, released, no node or GPUs",
					status.Phase, status.NodeName, status.GPUUUIDs, status.ReleasedAt)
			}
			if got := heldGPUs(t, r); len(got) != 0 {
				t.Fatalf("series after the claim was lost = %v, want none", got)
			}
		})
	}
}

// A lost dedicated claim is re-cut under the same name. The GPUs reported for
// the worker are the new allocation's, even though its pod kept its node.
func TestRecutClaimReportsItsOwnGPUs(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)
	settle(t, r, "w-a")
	claimName := claimOf(t, r, "w-a")
	allocateClaim(t, r, claimName)
	runReconcile(t, r, "w-a")

	pod := getPod(t, r, "orw-w-a")
	pod.Spec.NodeName = testNode
	if err := r.Update(context.Background(), pod); err != nil {
		t.Fatal(err)
	}
	if err := r.Delete(context.Background(), &resourcev1.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, Name: claimName}}); err != nil {
		t.Fatal(err)
	}
	runReconcile(t, r, "w-a")
	if again := claimOf(t, r, "w-a"); again != claimName {
		t.Fatalf("re-cut claim = %q, want the same name %q", again, claimName)
	}
	allocateClaimOn(t, r, claimName, "gpu-1")
	runReconcile(t, r, "w-a")

	status := getWorker(t, r, "w-a").Status
	if !slices.Equal(status.GPUUUIDs, []string{"GPU-gpu-1"}) || status.ReleasedAt != nil {
		t.Fatalf("status = uuids %v releasedAt %v, want GPU-gpu-1 and not released", status.GPUUUIDs, status.ReleasedAt)
	}
	want := []string{"model-a trainer orw-w-a node-a GPU-gpu-1=1"}
	if got := heldGPUs(t, r); !slices.Equal(got, want) {
		t.Fatalf("series = %v, want %v", got, want)
	}
}
