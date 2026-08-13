package controller

import (
	"context"
	"fmt"
	"sort"
	"strconv"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/controller/internal/placement"
)

// Labels the controller stamps on the objects it owns, and reads back to
// rebuild its own state after a restart.
const (
	LabelManaged      = "openrl.io/managed"
	LabelRole         = "openrl.io/role"
	LabelAccelCount   = "openrl.io/accelerator-count"
	LabelDeviceMemory = "openrl.io/device-memory"
	LabelClaim        = "openrl.io/claim"
	LabelWorker       = "openrl.io/worker"
	LabelSizedAgainst = "openrl.io/sized-against"
)

// Node labels the operator sets to opt a pool in. Policy, not hardware: the
// DRA driver's ResourceSlices report what the devices actually are.
//
// Opting a node in means opting it in exclusively: the controller counts a
// device as free unless one of its own claims holds it, so GPU workloads it
// does not manage on an enabled node would be invisible to placement.
const (
	NodeLabelEnabled            = "openrl.io/enabled"
	NodeLabelMaxWorkersPerClaim = "openrl.io/max-workers-per-claim"
	NodeLabelTrainer            = "openrl.io/trainer"
	NodeLabelSampler            = "openrl.io/sampler"
)

var nodeRoleLabel = map[openrlv1alpha1.WorkerRole]string{
	openrlv1alpha1.RoleTrainer: NodeLabelTrainer,
	openrlv1alpha1.RoleSampler: NodeLabelSampler,
}

// readFleet folds ResourceSlices, node labels, managed ResourceClaims and the
// workers already assigned to them into one picture to decide against.
func (r *OpenRLWorkerReconciler) readFleet(ctx context.Context, workers []openrlv1alpha1.OpenRLWorker) (*placement.Fleet, error) {
	var nodes corev1.NodeList
	if err := r.List(ctx, &nodes, client.MatchingLabels{NodeLabelEnabled: "true"}); err != nil {
		return nil, fmt.Errorf("list nodes: %w", err)
	}

	var slices resourcev1.ResourceSliceList
	if err := r.List(ctx, &slices); err != nil {
		return nil, fmt.Errorf("list resourceslices: %w", err)
	}

	fleet := placement.NewFleet()
	fleet.Nodes = r.poolsFrom(ctx, slices.Items, nodes.Items)

	// Consistent reader: a claim created for the previous worker in a burst
	// must be joinable by this one, and the cache may not have caught up.
	var claims resourcev1.ResourceClaimList
	if err := r.fleetReader().List(ctx, &claims, client.InNamespace(r.Namespace), client.MatchingLabels{LabelManaged: "true"}); err != nil {
		return nil, fmt.Errorf("list resourceclaims: %w", err)
	}
	for i := range claims.Items {
		if c := r.claimFrom(ctx, &claims.Items[i]); c != nil {
			fleet.Claims[c.Name] = c
		}
	}

	// Occupancy comes from worker statuses, not pods -- and the finalizer is
	// what makes that sound: a deleting worker keeps its CR, and with it the
	// real memory booking, until its pod is verifiably gone.
	for i := range workers {
		bookWorker(fleet, &workers[i])
	}
	return fleet, nil
}

// poolsFrom merges what the driver publishes with what the operator allowed.
// Devices accumulate across slices; where memory differs the smallest wins,
// because the fit must hold for whichever devices DRA picks.
func (r *OpenRLWorkerReconciler) poolsFrom(ctx context.Context, slices []resourcev1.ResourceSlice, nodes []corev1.Node) map[string]*placement.Node {
	logger := log.FromContext(ctx)

	devices := map[string]*placement.Node{}
	for _, i := range latestCompletePools(ctx, slices, r.DeviceDriver) {
		spec := slices[i].Spec
		name := *spec.NodeName
		for j := range spec.Devices {
			device := spec.Devices[j]
			capacity, ok := device.Capacity["memory"]
			if !ok {
				continue
			}
			memory := capacity.Value.Value()
			pool, seen := devices[name]
			if !seen {
				product := ""
				if attr, ok := device.Attributes["productName"]; ok && attr.StringValue != nil {
					product = *attr.StringValue
				}
				devices[name] = &placement.Node{Name: name, DeviceCount: 1, DeviceMemoryBytes: memory, Product: product}
				continue
			}
			pool.DeviceCount++
			pool.DeviceMemoryBytes = min(pool.DeviceMemoryBytes, memory)
		}
	}

	pools := map[string]*placement.Node{}
	for i := range nodes {
		node := &nodes[i]
		pool, ok := devices[node.Name]
		if !ok {
			logger.Info("node is enabled but no ResourceSlice from this driver describes it; skipping it for placement",
				"node", node.Name, "driver", r.DeviceDriver)
			continue
		}
		// A node naming no role labels allows both roles.
		pool.Roles = map[string]bool{}
		for role, label := range nodeRoleLabel {
			if node.Labels[label] == "true" {
				pool.Roles[string(role)] = true
			}
		}
		if len(pool.Roles) == 0 {
			for role := range nodeRoleLabel {
				pool.Roles[string(role)] = true
			}
		}
		pool.MaxWorkersPerClaim = max(1, labelInt(ctx, node.Labels, NodeLabelMaxWorkersPerClaim, 1, "node "+node.Name))
		pool.HostMemoryBytes = node.Status.Allocatable.Memory().Value()
		if pool.HostMemoryBytes == 0 {
			logger.Info("node reports no allocatable memory; parked-worker capacity cannot be checked here",
				"node", node.Name)
		}
		pools[node.Name] = pool
	}
	return pools
}

// latestCompletePools filters slices to the latest complete generation of
// each (node, pool) -- what the ResourceSlice contract requires of consumers.
// During a driver update, mixing generations double-counts devices and a
// partial generation under-counts them; an incomplete pool is skipped and
// picked up on a later reconcile.
func latestCompletePools(ctx context.Context, slices []resourcev1.ResourceSlice, driver string) []int {
	type poolKey struct{ node, pool string }
	byPool := map[poolKey][]int{}
	for i := range slices {
		spec := slices[i].Spec
		if spec.NodeName == nil || *spec.NodeName == "" || spec.Driver != driver {
			continue
		}
		key := poolKey{*spec.NodeName, spec.Pool.Name}
		byPool[key] = append(byPool[key], i)
	}

	var keep []int
	for key, indices := range byPool {
		latest := int64(-1)
		for _, i := range indices {
			if g := slices[i].Spec.Pool.Generation; g > latest {
				latest = g
			}
		}
		var current []int
		for _, i := range indices {
			if slices[i].Spec.Pool.Generation == latest {
				current = append(current, i)
			}
		}
		if int64(len(current)) != slices[current[0]].Spec.Pool.ResourceSliceCount {
			log.FromContext(ctx).Info("skipping incomplete ResourceSlice pool",
				"node", key.node, "pool", key.pool, "have", len(current), "want", slices[current[0]].Spec.Pool.ResourceSliceCount)
			continue
		}
		keep = append(keep, current...)
	}
	sort.Ints(keep)
	return keep
}

// claimFrom reads back the shape the controller stamped on a claim it created.
func (r *OpenRLWorkerReconciler) claimFrom(ctx context.Context, claim *resourcev1.ResourceClaim) *placement.Claim {
	count := labelInt(ctx, claim.Labels, LabelAccelCount, 0, "claim "+claim.Name)
	if count < 1 {
		log.FromContext(ctx).Info("skipping claim with unusable accelerator-count label", "claim", claim.Name)
		return nil
	}
	return &placement.Claim{
		Name:        claim.Name,
		DeviceCount: count,
		Node:        allocatedNode(claim),
		// Only read while unallocated: the reservation that keeps a burst from
		// cutting more claims than the pool has devices.
		SizedAgainst: claim.Labels[LabelSizedAgainst],
	}
}

// bookWorker charges a worker's assignment against the claim it names,
// re-deriving memory from the spec (the status string is Gi-rounded).
func bookWorker(fleet *placement.Fleet, worker *openrlv1alpha1.OpenRLWorker) {
	if claim, ok := fleet.Claims[worker.Status.ClaimName]; ok {
		request := requestFrom(worker)
		claim.Book(request.WorkerID, request.PerDeviceBytes(claim.DeviceCount))
	}
}

// allocatedNode is the node a claim was allocated to, or "" if DRA has not
// decided. DRA reports placement as a node selector; a GPU allocation pins to
// exactly one hostname.
func allocatedNode(claim *resourcev1.ResourceClaim) string {
	if claim.Status.Allocation == nil || claim.Status.Allocation.NodeSelector == nil {
		return ""
	}
	for _, term := range claim.Status.Allocation.NodeSelector.NodeSelectorTerms {
		for _, expr := range term.MatchExpressions {
			if isHostnameKey(expr.Key) && len(expr.Values) > 0 {
				return expr.Values[0]
			}
		}
		for _, field := range term.MatchFields {
			if isHostnameKey(field.Key) && len(field.Values) > 0 {
				return field.Values[0]
			}
		}
	}
	return ""
}

func isHostnameKey(key string) bool {
	return key == corev1.LabelHostname || key == "metadata.name"
}

// requestFrom is the placement Request an OpenRLWorker spec is asking for.
// Validation is the CRD schema's job.
func requestFrom(worker *openrlv1alpha1.OpenRLWorker) placement.Request {
	spec := worker.Spec
	return placement.Request{
		Role:   string(spec.Role),
		Memory: spec.Memory.Value(),
		// Raw: the spec calls the owner ID opaque, and sanitizing here would
		// merge distinct owners ("A/B" and "a-b") into one fairness slot.
		// Labels sanitize at the stamping site; env vars carry this exactly.
		Owner: spec.OwnerID,
		// The CR name: the one identity Kubernetes already guarantees unique.
		// Model id is model configuration, not object identity.
		WorkerID: worker.Name,
	}
}

// labelInt reads a non-negative integer label, falling back to a default if it
// is missing or nonsense.
func labelInt(ctx context.Context, labels map[string]string, key string, fallback int, subject string) int {
	raw, ok := labels[key]
	if !ok {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 {
		log.FromContext(ctx).Info("ignoring unparseable label", "subject", subject, "label", key, "value", raw, "using", fallback)
		return fallback
	}
	return value
}

// gibQuantity renders a byte count as the Gi string the CRD reports.
func gibQuantity(bytes int64) string {
	return resource.NewQuantity(placement.CeilGiB(bytes)*placement.GiB, resource.BinarySI).String()
}
