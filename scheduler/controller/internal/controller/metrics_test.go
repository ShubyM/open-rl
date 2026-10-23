package controller

import (
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	openrlv1alpha1 "github.com/gke-labs/open-rl/scheduler/controller/api/v1alpha1"
)

// heldGPUs scrapes openrl_gpu_held as the metrics endpoint does, one
// "run role pod node UUID=value" string per series.
func heldGPUs(t *testing.T, r *WorkloadReconciler) []string {
	t.Helper()
	registry := prometheus.NewPedanticRegistry()
	registry.MustRegister(gpuHeldCollector{reader: r.Client, namespace: testNamespace})
	families, err := registry.Gather()
	must(t, err)
	var series []string
	for _, family := range families {
		for _, metric := range family.GetMetric() {
			l := map[string]string{}
			for _, pair := range metric.GetLabel() {
				l[pair.GetName()] = pair.GetValue()
			}
			series = append(series, fmt.Sprintf("%s %s %s %s %s=%v", l["run_id"], l["role"], l["pod"], l["node"], l["UUID"], metric.GetGauge().GetValue()))
		}
	}
	return series
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// placedWorker returns a reconciler whose worker w-a holds gpu-0, and its claim.
// The observed allocation stamps the GPUs and the time, and the metric reports
// the hold.
func placedWorker(t *testing.T) (*WorkloadReconciler, string) {
	t.Helper()
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)
	settle(t, r, "w-a")
	claimName := claimOf(t, r, "w-a")
	allocateClaim(t, r, claimName)
	runReconcile(t, r, "w-a")
	placed := getWorker(t, r, "w-a").Status
	if placed.NodeName != testNode || !slices.Equal(placed.GPUUUIDs, []string{"GPU-gpu-0"}) || placed.PlacedAt == nil || placed.ReleasedAt != nil {
		t.Fatalf("status = node %q uuids %v placedAt %v releasedAt %v, want node-a, GPU-gpu-0, placed, not released",
			placed.NodeName, placed.GPUUUIDs, placed.PlacedAt, placed.ReleasedAt)
	}
	if got, want := heldGPUs(t, r), []string{"model-a trainer orw-w-a node-a GPU-gpu-0=1"}; !slices.Equal(got, want) {
		t.Fatalf("series = %v, want %v", got, want)
	}
	return r, claimName
}

func expectReleased(t *testing.T, r *WorkloadReconciler) {
	t.Helper()
	status := getWorker(t, r, "w-a").Status
	if status.Phase != openrlv1alpha1.PhasePending || status.ReleasedAt == nil || status.NodeName != "" || len(status.GPUUUIDs) != 0 {
		t.Fatalf("status = phase %q node %q uuids %v releasedAt %v, want Pending, released, no node or GPUs",
			status.Phase, status.NodeName, status.GPUUUIDs, status.ReleasedAt)
	}
	if got := heldGPUs(t, r); len(got) != 0 {
		t.Fatalf("series after release = %v, want none", got)
	}
}

// Abandoning a wedged claim releases the placement.
func TestAbandonedClaimReleasesThePlacement(t *testing.T) {
	r, _ := placedWorker(t)
	markUnschedulable(t, r, "w-a", time.Now().Add(-2*wedgeGracePeriod))
	runReconcile(t, r, "w-a")
	expectReleased(t, r)
}
