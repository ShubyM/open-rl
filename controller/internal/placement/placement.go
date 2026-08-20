// Package placement is the scheduling decision: pure functions over a Request
// and a Fleet, no Kubernetes imports. A claim is a bundle of accelerators;
// several workers may be assigned to it; exactly one is resident at a time.
//
// Placement never surveys free capacity. A new worker's claim states an
// ordered set of acceptable device shapes (Tiers) and DRA decides whether one
// is free -- kube-scheduler's allocation cycle is the mutex, not this
// package. What remains here is the shape catalog (Tiers), the sharing
// decision for a worker whose dedicated claim went unsatisfied (SelectClaim),
// and the explanation when neither can help (Explain).
package placement

import (
	"fmt"
	"time"
)

// GiB is the unit every memory figure here is reported in.
const GiB int64 = 1 << 30

// CeilGiB rounds a byte count up to whole GiB. The one rounding rule for
// every figure the scheduler reports or writes into a CEL selector.
func CeilGiB(bytes int64) int64 {
	return (bytes + GiB - 1) / GiB
}

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

// booking is one assigned worker's footprint on a claim: its owner (the
// fairness unit) and its pod's host-memory request. No accelerator figure:
// only one worker is ever resident, so nothing is summed against the device.
type booking struct {
	owner     string
	hostBytes int64
}

// Claim is a ResourceClaim, plus what is already sitting on it. Claims are
// not partitioned by role or workload type: anything the node accepts may
// join and take turns.
type Claim struct {
	Name string
	// DeviceCount is how many devices DRA allocated, 0 until it has.
	DeviceCount int
	// Node is where the claim was allocated, empty until DRA has decided.
	Node string
	// Created is when the claim was cut: the clock the scale-out grace
	// period runs against while the claim waits for DRA.
	Created time.Time
	// booked is each assigned worker's footprint -- deliberately never one
	// total, because nothing is ever summed against the device.
	booked map[string]booking
}

// Allocated reports whether the scheduler has said where this claim landed.
func (c *Claim) Allocated() bool { return c.Node != "" }

// Workers is how many workers are assigned to this claim.
func (c *Claim) Workers() int { return len(c.booked) }

// Book accepts a placement: one more assigned worker and its footprint.
func (c *Claim) Book(workerID, owner string, hostBytes int64) {
	if c.booked == nil {
		c.booked = map[string]booking{}
	}
	c.booked[workerID] = booking{owner: owner, hostBytes: hostBytes}
}

// Release gives back a worker's seat and the memory that came with it.
func (c *Claim) Release(workerID string) {
	delete(c.booked, workerID)
}

// Owners is how many distinct fairness units are assigned to this claim.
func (c *Claim) Owners() int {
	owners := map[string]bool{}
	for _, b := range c.booked {
		owners[b.owner] = true
	}
	return len(owners)
}

// HostBytesWith is the host memory the claim's node must satisfy if one more
// pod requesting hostBytes joins: the sum of every assigned pod's request.
// The pods all sit on the claim's node whether resident or parked, so their
// requests -- which carry the parking headroom -- must fit together.
func (c *Claim) HostBytesWith(hostBytes int64) int64 {
	total := hostBytes
	for _, b := range c.booked {
		total += b.hostBytes
	}
	return total
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

// NodeHostBytes is the host memory every worker already assigned to this
// node's claims will request from it. Summed across claims, because the
// node's allocatable memory is one pool however many claims sit on it.
func (f *Fleet) NodeHostBytes(node *Node) int64 {
	var total int64
	for _, claim := range f.Claims {
		if claim.Node != node.Name {
			continue
		}
		for _, b := range claim.booked {
			total += b.hostBytes
		}
	}
	return total
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
	// MaxDevices is the widest claim the runtime can drive; placement never
	// sizes one wider. Zero means one: no shipped runtime is multi-device,
	// and a wider claim is memory the pod can see but the process won't use.
	MaxDevices int
	// HostRequestBytes is the pod template's memory request: what this
	// worker's pod asks the node for, resident or parked.
	HostRequestBytes int64
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
// pool cannot hold it within the shape the runtime declared. Ceiling
// division bounded by MaxDevices: a pool whose devices are too small to
// satisfy the worker within that many devices is ineligible, never padded
// out with devices the process cannot drive.
func (r Request) DevicesOn(n *Node) int {
	if n.DeviceMemoryBytes <= 0 {
		return 0
	}
	count := devicesFor(r.Memory, n.DeviceMemoryBytes)
	if count > r.widest() || count > n.DeviceCount {
		return 0
	}
	return count
}

// widest is the most devices the runtime declared it can drive; zero means one.
func (r Request) widest() int { return max(1, r.MaxDevices) }

// devicesFor is how many devices of one size hold the memory: ceiling
// division, at least one.
func devicesFor(memory, deviceBytes int64) int {
	return max(1, int((memory+deviceBytes-1)/deviceBytes))
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

// SelectClaim picks the allocated claim a worker should join, or nil:
// single-device only, role / host-memory / worker-ceiling checked, preferring
// fewest owners, then fewest workers, then name. Advisory -- the group's CAS
// booking is what makes it stick.
func SelectClaim(req Request, fleet *Fleet) *Claim {
	var best *Claim
	for _, claim := range fleet.Claims {
		node := fleet.Nodes[claim.Node]
		if !claim.Allocated() || node == nil || !node.Accepts(req.Role) {
			continue
		}
		if claim.DeviceCount != 1 || req.DevicesOn(node) != 1 {
			continue
		}
		if claim.Workers() >= node.MaxWorkersPerClaim {
			continue
		}
		if req.Memory > node.DeviceMemoryBytes {
			continue
		}
		// Node-wide: every pod on the node draws from one allocatable pool.
		// A node reporting no memory skips the check rather than refusing all.
		if node.HostMemoryBytes > 0 && fleet.NodeHostBytes(node)+req.HostRequestBytes > node.HostMemoryBytes {
			continue
		}
		if best == nil || claimLess(claim, best) {
			best = claim
		}
	}
	return best
}

// claimLess orders eligible claims: fewer assigned owners first (joining the
// least-contended fairness pool), then fewer workers, then name.
func claimLess(a, b *Claim) bool {
	if a.Owners() != b.Owners() {
		return a.Owners() < b.Owners()
	}
	if a.Workers() != b.Workers() {
		return a.Workers() < b.Workers()
	}
	return a.Name < b.Name
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
		reason = fmt.Sprintf("NoCapacity: needs %dGi across at most %d device(s); largest pool offers %s",
			CeilGiB(req.Memory), req.widest(), biggest.Describe())
	}

	if detail != "" {
		reason += ". " + detail
	}
	if len(reason) > 1024 {
		reason = reason[:1024]
	}
	return reason
}
