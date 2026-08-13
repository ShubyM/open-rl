// What we expect of placement, end to end, played through Decide the way the
// controller plays it -- with DRA's answer arriving instantly. The memory
// figures are the estimator's real outputs: the gateway's tier table says
// 10Gi for the L4 tier and 60Gi for the 80Gi tier, and the hardware shapes
// are the ones we run (2x L4 24Gi dev box, single 80Gi devices).
//
//   - free GPUs are used before anyone shares
//   - workers share a claim whether or not their sums fit; only independent
//     fit matters
//   - a full claim makes the next worker wait, and a departure frees the seat
//   - role labels are policy: they hide hardware from workers
//   - host memory bounds how many suspended workers a node carries
//   - a worker no single node can hold stays waiting, with a reason
//   - placed workers stay put when capacity frees; new capacity serves new
//     work
//
// Who is resident at any moment is the timeslicer's job, tested in
// tests/test_accel_timeslicer.py. The pipeline against a real API server is
// hack/kind-smoke.sh.
package placement

import (
	"strings"
	"testing"
)

// cluster drives Decide exactly as the controller's assign step does, with
// the one simulation liberty that a new claim binds to its pool immediately
// (the real one waits for DRA; pending claims are covered by the unit tests).
type cluster struct {
	t      *testing.T
	fleet  *Fleet
	placed map[string]*Claim
}

func newCluster(t *testing.T, nodes ...*Node) *cluster {
	t.Helper()
	fleet := NewFleet()
	for _, n := range nodes {
		fleet.Nodes[n.Name] = n
	}
	return &cluster{t: t, fleet: fleet, placed: map[string]*Claim{}}
}

// arrive places one worker and reports where it landed: a claim name, or ""
// while it waits.
func (c *cluster) arrive(req Request) string {
	c.t.Helper()
	pool, join := Decide(req, c.fleet)
	switch {
	case pool != nil:
		claim := &Claim{Name: "claim-" + req.WorkerID, DeviceCount: pool.DeviceCount, Node: pool.Node.Name}
		claim.Book(req.WorkerID, req.PerDeviceBytes(pool.DeviceCount))
		c.fleet.Claims[claim.Name] = claim
		c.placed[req.WorkerID] = claim
		return claim.Name
	case join != nil:
		join.Book(req.WorkerID, req.PerDeviceBytes(join.DeviceCount))
		c.placed[req.WorkerID] = join
		return join.Name
	}
	return ""
}

// leave releases a worker's seat; an emptied claim returns its accelerators.
func (c *cluster) leave(workerID string) {
	c.t.Helper()
	claim := c.placed[workerID]
	if claim == nil {
		c.t.Fatalf("%s was never placed", workerID)
	}
	claim.Release(workerID)
	delete(c.placed, workerID)
	if claim.Workers() == 0 {
		delete(c.fleet.Claims, claim.Name)
	}
}

// waitingReason is why an unplaced worker is waiting.
func (c *cluster) waitingReason(req Request) string {
	return Explain(req, c.fleet, "")
}

func l4Box(name string) *Node {
	return &Node{
		Name: name, DeviceCount: 2, DeviceMemoryBytes: gib(24), HostMemoryBytes: gib(94),
		Roles: map[string]bool{"trainer": true, "sampler": true}, MaxWorkersPerClaim: 4, Product: "NVIDIA L4",
	}
}

func gpu80(name string, maxWorkers int, roles ...string) *Node {
	allowed := map[string]bool{}
	for _, role := range roles {
		allowed[role] = true
	}
	return &Node{
		Name: name, DeviceCount: 1, DeviceMemoryBytes: gib(80), HostMemoryBytes: gib(180),
		Roles: allowed, MaxWorkersPerClaim: maxWorkers,
	}
}

func TestWorkersSpreadAcrossFreeGPUs(t *testing.T) {
	c := newCluster(t, gpu80("node-a", 2, "trainer"), gpu80("node-b", 2, "trainer"))

	// Two 80Gi-tier estimates (the gateway's table says 60Gi) while both
	// GPUs are free: a claim each, on different nodes.
	a := c.arrive(trainer("w1", 60))
	b := c.arrive(trainer("w2", 60))
	if a == "" || b == "" || a == b {
		t.Fatalf("placed on %q and %q, want separate claims while devices are free", a, b)
	}
	if c.placed["w1"].Node == c.placed["w2"].Node {
		t.Errorf("both claims landed on %s, want separate nodes", c.placed["w1"].Node)
	}
}

// An FFT trainer and its sampler under contention: whether their sum would
// fit (45+25 on 80Gi) or not (55+35), V1 places them identically, because
// each only has to fit alone -- they take exclusive turns at runtime.
func TestWorkersShareAClaimWhetherOrNotTheirSumFits(t *testing.T) {
	for name, pair := range map[string][2]int64{
		"sum fits":     {45, 25},
		"sum does not": {55, 35},
	} {
		t.Run(name, func(t *testing.T) {
			c := newCluster(t, gpu80("gpu", 2, "trainer", "sampler"))
			first := c.arrive(trainer("trainer", pair[0]))
			second := c.arrive(Request{Role: "sampler", WorkerID: "sampler", Memory: gib(pair[1])})
			if first == "" || first != second {
				t.Fatalf("placed on %q and %q, want one shared claim", first, second)
			}
		})
	}
}

// max-workers-per-claim is a seat count, not a memory rule: the third worker
// fits the GPU but waits, and takes the seat the moment one frees.
func TestAFullClaimMakesTheNextWorkerWaitForASeat(t *testing.T) {
	c := newCluster(t, gpu80("gpu", 2, "trainer"))
	c.arrive(trainer("j1", 50))
	shared := c.arrive(trainer("j2", 50))

	if got := c.arrive(trainer("j3", 20)); got != "" {
		t.Fatalf("j3 was seated on %q, but the claim is at max-workers-per-claim", got)
	}
	c.leave("j1")
	if got := c.arrive(trainer("j3", 20)); got != shared {
		t.Fatalf("j3 landed on %q, want the seat freed on %q -- the claim outlives its first worker", got, shared)
	}
}

// Labels decide what OpenRL may use: a free GPU on a trainer-only node is
// invisible to a sampler, and the reason names the policy.
func TestRoleLabelsHideFreeHardware(t *testing.T) {
	c := newCluster(t, gpu80("trainer-only", 2, "trainer"))

	sampler := Request{Role: "sampler", WorkerID: "sampler", Memory: gib(10)}
	if got := c.arrive(sampler); got != "" {
		t.Fatalf("sampler landed on %q, but the operator allowed no sampler nodes", got)
	}
	if reason := c.waitingReason(sampler); !strings.Contains(reason, "no enabled node accepts sampler") {
		t.Errorf("reason = %q, want the policy named, not capacity", reason)
	}
}

// GPU memory fits but host memory does not: suspended workers park in host
// RAM, so a thin-host node refuses the worker that would overflow it -- until
// a departure frees the memory.
func TestHostMemoryBoundsAdmission(t *testing.T) {
	thin := gpu80("thin-host", 8, "trainer")
	thin.HostMemoryBytes = gib(40) // budget 34Gi after headroom
	c := newCluster(t, thin)

	c.arrive(trainer("w1", 20))
	if got := c.arrive(trainer("w2", 20)); got == "" {
		t.Fatal("w2 parks only 20Gi, inside the budget, and should have joined")
	}
	if got := c.arrive(trainer("w3", 20)); got != "" {
		t.Fatalf("w3 was seated on %q, but parking two 20Gi workers exceeds the 34Gi budget", got)
	}
	c.leave("w1")
	if got := c.arrive(trainer("w3", 20)); got == "" {
		t.Fatal("w3 still refused after the departure freed host memory")
	}
}

// Capacity that exists only across nodes places nothing; the same memory on
// one node places as a multi-device claim.
func TestAWorkerNoSingleNodeCanHoldStaysWaiting(t *testing.T) {
	across := newCluster(t, gpu80("node-a", 2, "trainer"), gpu80("node-b", 2, "trainer"))
	big := trainer("big", 140)
	if got := across.arrive(big); got != "" {
		t.Fatalf("140Gi landed on %q, but no single node can hold it", got)
	}
	if reason := across.waitingReason(big); !strings.Contains(reason, "NoCapacity") {
		t.Errorf("reason = %q, want NoCapacity: waiting would never help", reason)
	}

	oneNode := newCluster(t, &Node{
		Name: "wide", DeviceCount: 2, DeviceMemoryBytes: gib(80), HostMemoryBytes: gib(360),
		Roles: map[string]bool{"trainer": true}, MaxWorkersPerClaim: 2,
	})
	if oneNode.arrive(big) == "" || oneNode.placed["big"].DeviceCount != 2 {
		t.Fatalf("placed = %+v, want a two-device claim on the one node that fits it", oneNode.placed["big"])
	}
}

// V1 does not rebalance: the pair placed under contention keeps sharing after
// a GPU frees, and the freed GPU serves new work instead.
func TestPlacedWorkersStayPutWhenAGPUFrees(t *testing.T) {
	c := newCluster(t, gpu80("gpu-a", 2, "trainer"), gpu80("gpu-b", 2, "trainer"))
	c.arrive(trainer("z-blocker", 60))
	shared := c.arrive(trainer("w1", 50))
	if got := c.arrive(trainer("w2", 50)); got != shared {
		t.Fatalf("w2 on %q, want it sharing %q while both GPUs were taken", got, shared)
	}

	c.leave("z-blocker")
	if got := c.placed["w2"]; got.Name != shared {
		t.Fatalf("w2 moved to %q; V1 never migrates a placed worker", got.Name)
	}
	fresh := c.arrive(trainer("fresh", 50))
	if fresh == "" || c.placed["fresh"].Node != "gpu-a" {
		t.Fatalf("fresh = %+v, want it on the GPU the blocker freed", c.placed["fresh"])
	}
}

// The dev box, using the estimator's own tier outputs: 10Gi lora-tier
// workers on 2x L4. Spread first, then share, then wait for a seat.
func TestTierTableWorkloadsOnTheDevBox(t *testing.T) {
	c := newCluster(t, l4Box("box"))

	a := c.arrive(trainer("lora-1", 10))
	b := c.arrive(trainer("lora-2", 10))
	if a == b {
		t.Fatalf("both loras landed on %q while an L4 was free", a)
	}
	// The 80Gi tier's 60Gi estimate does not fit any L4, whole box or not.
	if got := c.arrive(trainer("fft-8b", 60)); got != "" {
		t.Fatalf("a 60Gi estimate landed on %q, but the box's devices are 24Gi", got)
	}
	// More loras double up on the existing claims instead.
	if got := c.arrive(trainer("lora-3", 10)); got != a {
		t.Fatalf("lora-3 landed on %q, want the fewest-workers claim %q", got, a)
	}
}
