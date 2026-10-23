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
func placedWorker(t *testing.T) (*WorkloadReconciler, string) {
	t.Helper()
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)
	settle(t, r, "w-a")
	claimName := claimOf(t, r, "w-a")
	allocateClaim(t, r, claimName)
	runReconcile(t, r, "w-a")
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

// An observed allocation stamps the GPUs and the time, the metric reports the
// hold, and teardown closes the interval once the pod is gone.
func TestPlacementRecordsGPUsUntilTeardown(t *testing.T) {
	ctx := context.Background()
	r, _ := placedWorker(t)
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
	must(t, r.Update(ctx, w))
	must(t, r.Delete(ctx, w))
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

// A placed worker stops holding its GPUs as soon as the controller sees its
// claim abandoned or gone, whether re-placement waits on the old claim's
// deletion or finds no capacity.
func TestLostClaimReleasesBeforeRePlacement(t *testing.T) {
	ctx := context.Background()
	cases := map[string]func(t *testing.T, r *WorkloadReconciler, claimName string){
		"wedged claim abandoned": func(t *testing.T, r *WorkloadReconciler, _ string) {
			markUnschedulable(t, r, "w-a", time.Now().Add(-2*wedgeGracePeriod))
		},
		"old claim still terminating": func(t *testing.T, r *WorkloadReconciler, claimName string) {
			var claim resourcev1.ResourceClaim
			must(t, r.Get(ctx, types.NamespacedName{Namespace: testNamespace, Name: claimName}, &claim))
			claim.Finalizers = append(claim.Finalizers, "test.openrl.io/hold")
			must(t, r.Update(ctx, &claim))
			must(t, r.Delete(ctx, &claim))
		},
		"no capacity left": func(t *testing.T, r *WorkloadReconciler, claimName string) {
			must(t, r.Delete(ctx, &resourcev1.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, Name: claimName}}))
			var node corev1.Node
			must(t, r.Get(ctx, types.NamespacedName{Name: testNode}, &node))
			node.Labels[NodeLabelEnabled] = "false"
			must(t, r.Update(ctx, &node))
		},
	}
	for name, loseClaim := range cases {
		t.Run(name, func(t *testing.T) {
			r, claimName := placedWorker(t)
			loseClaim(t, r, claimName)
			runReconcile(t, r, "w-a")
			expectReleased(t, r)
		})
	}
}

// A lost dedicated claim is re-cut under the same name. The GPUs reported for
// the worker are the new allocation's, even though its pod kept its node.
func TestRecutClaimReportsItsOwnGPUs(t *testing.T) {
	ctx := context.Background()
	r, claimName := placedWorker(t)
	pod := getPod(t, r, "orw-w-a")
	pod.Spec.NodeName = testNode
	must(t, r.Update(ctx, pod))
	must(t, r.Delete(ctx, &resourcev1.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Namespace: testNamespace, Name: claimName}}))
	runReconcile(t, r, "w-a")
	if again := claimOf(t, r, "w-a"); again != claimName {
		t.Fatalf("re-cut claim = %q, want the same name %q", again, claimName)
	}
	allocateClaimOn(t, r, claimName, "gpu-1")
	runReconcile(t, r, "w-a")

	want := []string{"model-a trainer orw-w-a node-a GPU-gpu-1=1"}
	if status := getWorker(t, r, "w-a").Status; !slices.Equal(status.GPUUUIDs, []string{"GPU-gpu-1"}) || status.ReleasedAt != nil {
		t.Fatalf("status = uuids %v releasedAt %v, want GPU-gpu-1 and not released", status.GPUUUIDs, status.ReleasedAt)
	}
	if got := heldGPUs(t, r); !slices.Equal(got, want) {
		t.Fatalf("series = %v, want %v", got, want)
	}
}
