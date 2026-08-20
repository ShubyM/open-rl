// The spec's end-to-end scenarios live in behavior_test.go, which drives
// these same functions with a simulated DRA. What lives here is only what
// the sim cannot reach: the window where claims exist but DRA has not
// allocated them, the deterministic orderings, and the arithmetic the spec
// pins in its appendix.
package placement

import (
	"strings"
	"testing"
)

func gib(n int64) int64 { return n * GiB }

// l4Node is the dev box: g2-standard-24 with 2x L4 24Gi and 94Gi allocatable.
func l4Node(name string, maxWorkers int, roles ...string) *Node {
	allowed := map[string]bool{}
	for _, role := range roles {
		allowed[role] = true
	}
	return &Node{
		Name:               name,
		DeviceCount:        2,
		DeviceMemoryBytes:  gib(24),
		HostMemoryBytes:    gib(94),
		Roles:              allowed,
		MaxWorkersPerClaim: maxWorkers,
		Product:            "NVIDIA L4",
	}
}

// bigNode is a pool of any shape, for the cases where a workload has to choose.
func bigNode(name string, devices int, deviceGiB int64, maxWorkers int, roles ...string) *Node {
	allowed := map[string]bool{}
	for _, role := range roles {
		allowed[role] = true
	}
	return &Node{
		Name:               name,
		DeviceCount:        devices,
		DeviceMemoryBytes:  gib(deviceGiB),
		HostMemoryBytes:    gib(340),
		Roles:              allowed,
		MaxWorkersPerClaim: maxWorkers,
	}
}

func trainer(id string, memoryGiB int64) Request {
	// MaxDevices 8: most of these tests exercise the arithmetic of a runtime
	// that declared it can shard; the single-device default is pinned where
	// it matters. The host request mirrors the memory figure: the template
	// asks the node for at least the footprint it may park.
	return Request{Role: "trainer", WorkerID: id, Memory: gib(memoryGiB), MaxDevices: 8, HostRequestBytes: gib(memoryGiB)}
}

// booked is a claim with workers already assigned to it, charged worker by
// worker the way readFleet rebuilds one from its group's seats.
func booked(c *Claim, workers ...string) *Claim {
	for _, worker := range workers {
		c.Book(worker, worker, 0)
	}
	return c
}

func name(c *Claim) string {
	if c == nil {
		return ""
	}
	return c.Name
}

// There is no sharding: the model is laid out layer by layer over whatever
// devices it gets, so aggregate memory is the only thing that has to add up and
// any device count will do -- including three.
func TestDevicesOnIsPlainCeilingDivision(t *testing.T) {
	node := bigNode("n", 8, 24, 1, "trainer")
	for _, tc := range []struct {
		memoryGiB int64
		want      int
	}{
		{10, 1},
		{24, 1},
		{25, 2},
		{60, 3}, // not a power of two, and that is fine
		{192, 8},
		{193, 0}, // more than the pool holds
	} {
		if got := trainer("w", tc.memoryGiB).DevicesOn(node); got != tc.want {
			t.Errorf("DevicesOn(%dGi) = %d, want %d", tc.memoryGiB, got, tc.want)
		}
	}
}

func TestPerDeviceBytesRoundsUp(t *testing.T) {
	req := trainer("w", 30)
	if got, want := req.PerDeviceBytes(2), gib(15); got != want {
		t.Errorf("PerDeviceBytes(2) = %d, want %d", got, want)
	}
	// 30Gi over 4 devices is 7.5Gi each; erring high is the safe direction.
	if got, want := req.PerDeviceBytes(4), (gib(30)+3)/4; got != want {
		t.Errorf("PerDeviceBytes(4) = %d, want %d", got, want)
	}
}

func TestOwnerKey(t *testing.T) {
	for _, tc := range []struct {
		name string
		req  Request
		want string
	}{
		{"a named owner is the owner", Request{Owner: "qwen3-0-6b", WorkerID: "job-a"}, "qwen3-0-6b"},
		{"no owner means an owner of one", Request{WorkerID: "job-a"}, "job-a"},
	} {
		if got := tc.req.OwnerKey(); got != tc.want {
			t.Errorf("%s: OwnerKey() = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// A pending claim's node is unknown, so nothing about it can be checked
// against anything real: it is never joined. The worker waits for DRA or
// for its own grace-period fallback instead.
func TestPendingClaimsAreNotJoined(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = &Node{
		Name: "n", DeviceCount: 1, DeviceMemoryBytes: gib(24), HostMemoryBytes: gib(94),
		Roles: map[string]bool{"trainer": true}, MaxWorkersPerClaim: 4,
	}
	fleet.Claims["pending"] = booked(&Claim{Name: "pending"}, "job-a")

	if got := SelectClaim(trainer("job-b", 6), fleet); got != nil {
		t.Errorf("joined %q, but an unallocated claim must not be joined", name(got))
	}
}

// max-workers-per-claim is enforced at the join: a full allocated claim
// refuses however much device memory remains.
func TestSelectClaimRejectsFullClaims(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", 2, "trainer")
	fleet.Claims["full"] = booked(&Claim{Name: "full", DeviceCount: 1, Node: "n"}, "job-a", "job-b")

	if got := SelectClaim(trainer("job-c", 6), fleet); got != nil {
		t.Errorf("joined %q, but max-workers-per-claim is 2 and both seats are taken", name(got))
	}
}

// The catalog is shapes, never free counts: however many claims exist, the
// same tiers come back, because whether anything is free is DRA's question.
func TestCatalogIgnoresOccupancy(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = bigNode("n", 2, 80, 2, "trainer")
	fleet.Claims["c1"] = booked(&Claim{Name: "c1", DeviceCount: 1, Node: "n"}, "job-a")
	fleet.Claims["c2"] = booked(&Claim{Name: "c2", DeviceCount: 1, Node: "n"}, "job-b")

	tiers := Tiers(trainer("w", 10), Catalog(fleet, "trainer"))
	if len(tiers) != 1 || tiers[0].Count != 1 {
		t.Errorf("Tiers = %+v, want the one 80Gi shape regardless of what is booked", tiers)
	}
	if got := Catalog(fleet, "sampler"); len(got) != 0 {
		t.Errorf("Catalog(sampler) = %+v, want empty: the pool is trainer-only", got)
	}
}

// The spec's preference order: fewest assigned workers, then claim name --
// deterministic across reconciles.
func TestSelectClaimPrefersFewestWorkersThenName(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", 4, "trainer")
	fleet.Claims["quiet"] = booked(&Claim{Name: "quiet", DeviceCount: 1, Node: "n"}, "job-b")
	fleet.Claims["crowded"] = booked(&Claim{Name: "crowded", DeviceCount: 1, Node: "n"}, "job-c", "job-d", "job-e")

	if got := SelectClaim(trainer("job-new", 6), fleet); name(got) != "quiet" {
		t.Errorf("joined %q, want the claim with the fewest workers", name(got))
	}

	fleet.Claims["a-quiet"] = booked(&Claim{Name: "a-quiet", DeviceCount: 1, Node: "n"}, "job-f")
	if got := SelectClaim(trainer("job-new", 6), fleet); name(got) != "a-quiet" {
		t.Errorf("joined %q, want the tie broken by name", name(got))
	}
}

// Tight fit lives in the tier order: the tier wasting the least device
// memory leads, so small work prefers small devices -- a preference DRA
// honors when it can, never a survey of what is free.
func TestTiersPreferTheTightestFit(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["l4"] = bigNode("l4", 4, 24, 4, "trainer")
	fleet.Nodes["big"] = bigNode("big", 4, 96, 4, "trainer")

	// 20Gi wastes 4Gi on an L4 and 76Gi on the big pool: both are offered,
	// the L4 shape first.
	tiers := Tiers(trainer("w", 20), Catalog(fleet, "trainer"))
	if len(tiers) != 2 || tiers[0].CeilingBytes != gib(24) || tiers[0].Count != 1 {
		t.Fatalf("Tiers = %+v, want the 24Gi shape leading", tiers)
	}

	// 200Gi does not fit four L4s at all, so the big shape is the only tier.
	tiers = Tiers(trainer("w", 200), Catalog(fleet, "trainer"))
	if len(tiers) != 1 || tiers[0].CeilingBytes != gib(96) || tiers[0].Count != 3 {
		t.Fatalf("Tiers = %+v, want only 3x96Gi", tiers)
	}
}

// Host admission sums the assigned pods' memory requests -- what the node
// must actually satisfy, resident or parked -- and fairness counts distinct
// owners, not workers.
func TestHostBytesSumTheAssignedPods(t *testing.T) {
	claim := &Claim{Name: "c", DeviceCount: 1, Node: "n"}
	claim.Book("trainer-a", "job-a", gib(40))
	claim.Book("sampler-a", "job-a", gib(20))

	if got, want := claim.HostBytesWith(gib(30)), gib(90); got != want {
		t.Errorf("HostBytesWith = %d, want %d", got, want)
	}
	if got := claim.Owners(); got != 1 {
		t.Errorf("Owners = %d, want 1: a trainer and sampler of one job are one fairness unit", got)
	}
	if got := claim.Workers(); got != 2 {
		t.Errorf("Workers = %d, want 2", got)
	}
}

// The single-device default: a worker that declares no shape is never given
// a claim wider than one device, and never joins a wide claim.
func TestDefaultShapeIsSingleDevice(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["l4"] = bigNode("l4", 4, 24, 4, "trainer")

	// 60Gi would take three L4s, but an undeclared runtime drives one device.
	undeclared := Request{Role: "trainer", WorkerID: "w", Memory: gib(60)}
	if tiers := Tiers(undeclared, Catalog(fleet, "trainer")); len(tiers) != 0 {
		t.Fatalf("Tiers = %+v for a single-device runtime that fits no single device", tiers)
	}

	// Declaring the shape is what unlocks the wider claim.
	declared := undeclared
	declared.MaxDevices = 4
	if tiers := Tiers(declared, Catalog(fleet, "trainer")); len(tiers) != 1 || tiers[0].Count != 3 {
		t.Fatalf("Tiers = %+v, want 3 devices once the runtime declares them", tiers)
	}
}

// A pending worker's reason distinguishes "busy, retry" from "too small,
// never" -- the spec's actionable-reason requirement.
func TestExplain(t *testing.T) {
	empty := NewFleet()
	if got := Explain(trainer("w", 6), empty, ""); !strings.HasPrefix(got, "NoCapacity") {
		t.Errorf("Explain = %q, want NoCapacity for a fleet with no pools", got)
	}

	tooSmall := NewFleet()
	tooSmall.Nodes["n"] = l4Node("n", 4, "trainer") // 2x24Gi
	got := Explain(trainer("w", 200), tooSmall, "")
	if !strings.HasPrefix(got, "NoCapacity") || !strings.Contains(got, "200Gi") {
		t.Errorf("Explain = %q, want NoCapacity naming the 200Gi it could not fit", got)
	}

	full := NewFleet()
	full.Nodes["n"] = l4Node("n", 1, "trainer")
	full.Claims["c"] = booked(&Claim{Name: "c", DeviceCount: 1, Node: "n"}, "job-a")
	got = Explain(trainer("w", 6), full, "pod is unschedulable")
	if !strings.HasPrefix(got, "WaitingForCapacity") || !strings.Contains(got, "pod is unschedulable") {
		t.Errorf("Explain = %q, want WaitingForCapacity carrying the caller's detail", got)
	}
}
