// Package placement is the scheduling decision: pure functions over a Request
// and a Fleet, no Kubernetes imports. A claim is a bundle of accelerators;
// several workers may be assigned to it; exactly one is resident at a time.
package placement

import (
	"fmt"
	"sort"
)

// GiB is the unit every memory figure here is reported in.
const GiB int64 = 1 << 30

// CeilGiB rounds a byte count up to whole GiB. The one rounding rule for
// every figure the scheduler reports or writes into a CEL selector.
func CeilGiB(bytes int64) int64 {
	return (bytes + GiB - 1) / GiB
}

// HostMemoryHeadroom is the share of a node's allocatable memory left for the
// kubelet, the DRA driver, page cache, and the resident worker's own host-side
// allocations. Only the remainder may hold suspended workers.
const HostMemoryHeadroom = 0.15

// Node is one accelerator pool: hardware from the driver's ResourceSlice,
// policy from the operator's node labels.
type Node struct {
	Name string
	// DeviceCount and DeviceMemoryBytes come from the DRA driver.
	DeviceCount       int
	DeviceMemoryBytes int64
	// HostMemoryBytes is the node's allocatable memory, which bounds how many
	// workers can be parked here at once.
	HostMemoryBytes int64
	// Roles is the set of worker roles the operator allowed on this pool.
	Roles map[string]bool
	// MaxWorkersPerClaim is the openrl.io/max-workers-per-claim ceiling. It is
	// a policy cap on queueing and switch overhead, not a capacity check, and
	// it never permits multiple residents.
	MaxWorkersPerClaim int
	Product            string
}

// Accepts reports whether the operator allowed this role on this pool.
func (n *Node) Accepts(role string) bool { return n.Roles[role] }

// Describe renders the pool's hardware for an error message.
func (n *Node) Describe() string {
	hardware := fmt.Sprintf("%dGi x %d", n.DeviceMemoryBytes/GiB, n.DeviceCount)
	if n.Product == "" {
		return hardware
	}
	return hardware + " " + n.Product
}

// HostBudget is how much of this node's memory may hold suspended workers.
// Suspension does not spill to disk -- it parks device memory in the process's
// own host address space -- so exceeding this OOM-kills the node. Zero means
// the node did not report allocatable memory; the check is skipped rather
// than guessed at.
func (n *Node) HostBudget() int64 {
	return int64(float64(n.HostMemoryBytes) * (1 - HostMemoryHeadroom))
}

// Claim is a ResourceClaim, plus what is already sitting on it. Claims are
// not partitioned by role or workload type: anything the node accepts may
// join and take turns.
type Claim struct {
	Name        string
	DeviceCount int
	// Node is where the claim was allocated, empty until DRA has decided.
	Node string
	// SizedAgainst is the pool a not-yet-allocated claim was cut for. Until
	// DRA decides, the claim reserves devices there, so a burst stops cutting
	// claims once a pool's devices are spoken for and starts joining instead.
	SizedAgainst string
	// booked is per-device bytes per assigned worker -- deliberately not one
	// total, because nothing is ever summed against the device.
	booked map[string]int64
}

// Allocated reports whether the scheduler has said where this claim landed.
func (c *Claim) Allocated() bool { return c.Node != "" }

// Workers is how many workers are assigned to this claim.
func (c *Claim) Workers() int { return len(c.booked) }

// Book accepts a placement: one more assigned worker and its per-device bytes.
func (c *Claim) Book(workerID string, perDeviceBytes int64) {
	if c.booked == nil {
		c.booked = map[string]int64{}
	}
	c.booked[workerID] = perDeviceBytes
}

// Release gives back a worker's seat and the memory that came with it.
func (c *Claim) Release(workerID string) {
	delete(c.booked, workerID)
}

// ParkedBytesWith is the host memory this claim's suspended workers would
// hold if one more worker of perDeviceBytes joined: the conservative case is
// the smallest worker resident and every other one parked, each holding what
// it had on every device.
func (c *Claim) ParkedBytesWith(perDeviceBytes int64) int64 {
	total, smallest := perDeviceBytes, perDeviceBytes
	for _, bytes := range c.booked {
		total += bytes
		if bytes < smallest {
			smallest = bytes
		}
	}
	return (total - smallest) * int64(c.DeviceCount)
}

// Fleet is everything placement decides against.
type Fleet struct {
	Nodes  map[string]*Node
	Claims map[string]*Claim
}

// NewFleet returns an empty Fleet.
func NewFleet() *Fleet {
	return &Fleet{Nodes: map[string]*Node{}, Claims: map[string]*Claim{}}
}

// FreeDevices is how many of a node's accelerators no claim has taken yet.
// An unallocated claim counts against the pool it was sized for: without that
// reservation, a burst reconciling before DRA decides would cut one claim per
// worker and leave most of them unsatisfiable.
func (f *Fleet) FreeDevices(node *Node) int {
	free := node.DeviceCount
	for _, claim := range f.Claims {
		if claim.Node == node.Name || (!claim.Allocated() && claim.SizedAgainst == node.Name) {
			free -= claim.DeviceCount
		}
	}
	return free
}

// Request is one worker's needs, parsed out of its spec once.
type Request struct {
	// Role selects node pools and nothing else; it does not partition claims.
	Role string
	// Memory is the total accelerator memory the worker needs, across however
	// many devices it ends up on.
	Memory int64
	// Owner is the runtime fairness unit. Placement never reads it; the
	// timeslicer does.
	Owner string
	// WorkerID identifies the worker. Required, and required to be unique.
	WorkerID string
}

// OwnerKey is the fairness unit the timeslicer serves this worker under; a
// worker naming no owner is an owner of one.
func (r Request) OwnerKey() string {
	if r.Owner != "" {
		return r.Owner
	}
	return r.WorkerID
}

// DevicesOn is how many of a node's devices this workload needs, or 0 if the
// pool cannot hold it. Plain ceiling division: there is no sharding, so any
// count will do and aggregate memory is the only thing that has to add up.
func (r Request) DevicesOn(n *Node) int {
	if n.DeviceMemoryBytes <= 0 {
		return 0
	}
	count := int((r.Memory + n.DeviceMemoryBytes - 1) / n.DeviceMemoryBytes)
	if count < 1 {
		count = 1
	}
	if count > n.DeviceCount {
		return 0
	}
	return count
}

// PerDeviceBytes is the workload's share of each device when spread over
// deviceCount of them. An even split that a layer-wise layout only
// approximates; erring high is the safe direction.
func (r Request) PerDeviceBytes(deviceCount int) int64 {
	if deviceCount < 1 {
		panic(fmt.Sprintf("deviceCount must be >= 1, got %d", deviceCount))
	}
	return (r.Memory + int64(deviceCount) - 1) / int64(deviceCount)
}

// candidateNodes is every pool that accepts the role and fits the workload,
// with the device count it would take there.
func candidateNodes(req Request, fleet *Fleet) map[string]int {
	fits := map[string]int{}
	for name, node := range fleet.Nodes {
		if !node.Accepts(req.Role) {
			continue
		}
		if count := req.DevicesOn(node); count > 0 {
			fits[name] = count
		}
	}
	return fits
}

// Decide is the placement policy in one sentence: spread onto a free pool
// while one exists, share an existing claim only under contention. Exactly
// one result is non-nil, or both are nil when nothing will have the worker.
func Decide(req Request, fleet *Fleet) (*Pool, *Claim) {
	if pool := ChoosePool(req, fleet); pool != nil {
		return pool, nil
	}
	return nil, SelectClaim(req, fleet)
}

// SelectClaim is the claim this worker should join, or nil if none will have
// it -- the sharing half of Decide.
//
// Only allocated claims are joinable: an unallocated claim's node is unknown,
// so nothing about it -- device memory, host memory, role policy, the worker
// ceiling -- can be checked against anything real. A burst does not scatter
// while it waits, because pending claims reserve their pool (FreeDevices);
// the joiners simply retry once DRA has decided.
//
// Three checks govern joining: the worker fits a device by itself (nothing
// is summed -- only one worker is ever resident), the claim is below
// max-workers-per-claim, and the node has host memory for the workers that
// may be parked. Preference: fewest assigned workers, then name, so the
// choice is deterministic.
func SelectClaim(req Request, fleet *Fleet) *Claim {
	var best *Claim
	for _, claim := range fleet.Claims {
		node := fleet.Nodes[claim.Node]
		if !claim.Allocated() || node == nil || !node.Accepts(req.Role) || req.DevicesOn(node) != claim.DeviceCount {
			continue
		}
		if claim.Workers() >= node.MaxWorkersPerClaim {
			continue
		}
		perDevice := req.PerDeviceBytes(claim.DeviceCount)
		if perDevice > node.DeviceMemoryBytes {
			continue
		}
		// A zero budget means the node reported no allocatable memory; skip
		// the check rather than refuse every claim on it.
		if budget := node.HostBudget(); budget > 0 && claim.ParkedBytesWith(perDevice) > budget {
			continue
		}
		if best == nil || claim.Workers() < best.Workers() ||
			(claim.Workers() == best.Workers() && claim.Name < best.Name) {
			best = claim
		}
	}
	return best
}

// Pool is the node a new claim is sized against and how wide it would be.
type Pool struct {
	Node        *Node
	DeviceCount int
}

// ChoosePool picks the pool a new claim is sized against -- best fit by
// wasted memory among pools with unclaimed accelerators -- or nil if none has
// room. Sizing only: DRA picks where the claim actually lands.
func ChoosePool(req Request, fleet *Fleet) *Pool {
	var best *Pool
	var bestWaste int64
	names := make([]string, 0, len(fleet.Nodes))
	for name := range fleet.Nodes {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		node := fleet.Nodes[name]
		if !node.Accepts(req.Role) {
			continue
		}
		count := req.DevicesOn(node)
		if count == 0 || count > fleet.FreeDevices(node) {
			continue
		}
		waste := int64(count)*node.DeviceMemoryBytes - req.Memory
		if best == nil || waste < bestWaste || (waste == bestWaste && count < best.DeviceCount) {
			best, bestWaste = &Pool{Node: node, DeviceCount: count}, waste
		}
	}
	return best
}

// Explain says why this worker is not running. "Busy, retry" and "too small,
// never" are deliberately different answers; detail carries the caller's own
// words about what failed.
func Explain(req Request, fleet *Fleet, detail string) string {
	var pools []*Node
	for _, node := range fleet.Nodes {
		if node.Accepts(req.Role) {
			pools = append(pools, node)
		}
	}

	var reason string
	switch {
	case len(pools) == 0:
		reason = fmt.Sprintf("NoCapacity: no enabled node accepts %s workers", req.Role)
	case len(candidateNodes(req, fleet)) > 0:
		// The hardware exists; it is busy. Retrying is the right move.
		reason = "WaitingForCapacity: a pool fits this workload but none has a free seat or a free accelerator"
	default:
		biggest := pools[0]
		for _, node := range pools[1:] {
			if int64(node.DeviceCount)*node.DeviceMemoryBytes > int64(biggest.DeviceCount)*biggest.DeviceMemoryBytes {
				biggest = node
			}
		}
		reason = fmt.Sprintf("NoCapacity: needs %dGi in total; largest pool offers %s",
			CeilGiB(req.Memory), biggest.Describe())
	}

	if detail != "" {
		reason += ". " + detail
	}
	return truncate(reason, 1024)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
