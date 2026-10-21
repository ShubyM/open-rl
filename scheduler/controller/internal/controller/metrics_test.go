package controller

import (
	"context"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
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
