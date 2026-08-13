// The spec's end-to-end scenarios are tested in internal/sim (sim_test.go),
// which drives these same functions. What lives here is only what the sim
// cannot reach: the burst window where claims exist but DRA has not
// allocated them (the sim binds immediately), the deterministic orderings,
// and the arithmetic the spec pins in its appendix.
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
	return Request{Role: "trainer", WorkerID: id, Memory: gib(memoryGiB)}
}

// booked is a claim with workers already assigned to it, charged worker by
// worker the way bookWorker rebuilds one from the workers that reference it.
func booked(c *Claim, perDevice int64, workers ...string) *Claim {
	for _, worker := range workers {
		c.Book(worker, perDevice)
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
// against anything real: it is reserved (FreeDevices), never joined. The
// burst waits a retry instead of scattering -- and a 70Gi worker can no
// longer join a claim whose stored selector only guarantees 10Gi devices.
func TestPendingClaimsAreReservedNotJoined(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = &Node{
		Name: "n", DeviceCount: 1, DeviceMemoryBytes: gib(24), HostMemoryBytes: gib(94),
		Roles: map[string]bool{"trainer": true}, MaxWorkersPerClaim: 4,
	}
	fleet.Claims["pending"] = booked(&Claim{Name: "pending", DeviceCount: 1, SizedAgainst: "n"}, gib(6), "job-a")

	if got := SelectClaim(trainer("job-b", 6), fleet); got != nil {
		t.Errorf("joined %q, but an unallocated claim must not be joined", name(got))
	}
	if pool, join := Decide(trainer("job-b", 6), fleet); pool != nil || join != nil {
		t.Errorf("Decide = (%+v, %v), want the worker to wait for the allocation", pool, name(join))
	}
}

// max-workers-per-claim is enforced at the join: a full allocated claim
// refuses however much device memory remains.
func TestSelectClaimRejectsFullClaims(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", 2, "trainer")
	fleet.Claims["full"] = booked(&Claim{Name: "full", DeviceCount: 1, Node: "n"}, gib(2), "job-a", "job-b")

	if got := SelectClaim(trainer("job-c", 6), fleet); got != nil {
		t.Errorf("joined %q, but max-workers-per-claim is 2 and both seats are taken", name(got))
	}
}

// An unallocated claim reserves the pool it was sized against, so a burst
// stops cutting claims once a pool's devices are spoken for.
func TestUnallocatedClaimsReserveTheirPool(t *testing.T) {
	fleet := NewFleet()
	node := bigNode("n", 2, 80, 2, "trainer")
	fleet.Nodes["n"] = node
	fleet.Claims["c1"] = &Claim{Name: "c1", DeviceCount: 1, SizedAgainst: "n"}
	fleet.Claims["c2"] = &Claim{Name: "c2", DeviceCount: 1, SizedAgainst: "n"}

	if free := fleet.FreeDevices(node); free != 0 {
		t.Errorf("FreeDevices = %d, want 0: both devices are reserved by pending claims", free)
	}
	if pool := ChoosePool(trainer("w", 10), fleet); pool != nil {
		t.Errorf("ChoosePool = %+v, want nil so the worker joins instead of cutting a third claim", pool)
	}
}

// Decide is the one place the spread-before-share order lives: a free pool
// wins over a joinable claim; under contention the claim wins over waiting.
func TestDecideSpreadsBeforeSharing(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = bigNode("n", 2, 24, 4, "trainer")
	fleet.Claims["c"] = booked(&Claim{Name: "c", DeviceCount: 1, Node: "n"}, gib(6), "job-a")

	if pool, join := Decide(trainer("job-b", 6), fleet); pool == nil || join != nil {
		t.Errorf("Decide = (%+v, %v), want the free device, not the shared claim", pool, name(join))
	}

	fleet.Claims["c2"] = booked(&Claim{Name: "c2", DeviceCount: 1, Node: "n"}, gib(6), "job-b")
	if pool, join := Decide(trainer("job-c", 6), fleet); pool != nil || join == nil {
		t.Errorf("Decide = (%+v, %v), want a shared claim once no device is free", pool, name(join))
	}
}

// The spec's preference order: fewest assigned workers, then claim name --
// deterministic across reconciles.
func TestSelectClaimPrefersFewestWorkersThenName(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", 4, "trainer")
	fleet.Claims["quiet"] = booked(&Claim{Name: "quiet", DeviceCount: 1, Node: "n"}, gib(2), "job-b")
	fleet.Claims["crowded"] = booked(&Claim{Name: "crowded", DeviceCount: 1, Node: "n"}, gib(2), "job-c", "job-d", "job-e")

	if got := SelectClaim(trainer("job-new", 6), fleet); name(got) != "quiet" {
		t.Errorf("joined %q, want the claim with the fewest workers", name(got))
	}

	fleet.Claims["a-quiet"] = booked(&Claim{Name: "a-quiet", DeviceCount: 1, Node: "n"}, gib(2), "job-f")
	if got := SelectClaim(trainer("job-new", 6), fleet); name(got) != "a-quiet" {
		t.Errorf("joined %q, want the tie broken by name", name(got))
	}
}

// New claims are sized best-fit by wasted memory, so small work stays off the
// big pool while an L4 can hold it.
func TestChoosePoolPrefersTheTightestFit(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["l4"] = bigNode("l4", 4, 24, 4, "trainer")
	fleet.Nodes["big"] = bigNode("big", 4, 96, 4, "trainer")

	// 20Gi wastes 4Gi on an L4 and 76Gi on the big pool.
	pool := ChoosePool(trainer("w", 20), fleet)
	if pool == nil || pool.Node.Name != "l4" || pool.DeviceCount != 1 {
		t.Fatalf("ChoosePool picked %+v, want 1 device on the L4 pool", pool)
	}

	// 200Gi does not fit four L4s at all, so the big pool is the only answer.
	pool = ChoosePool(trainer("w", 200), fleet)
	if pool == nil || pool.Node.Name != "big" || pool.DeviceCount != 3 {
		t.Fatalf("ChoosePool picked %+v, want 3 devices on the big pool", pool)
	}
}

// Appendix A's host-memory rule, pinned as arithmetic: the conservative case
// parks every assigned worker except the smallest one.
func TestParkedBytesLeaveTheSmallestWorkerResident(t *testing.T) {
	claim := booked(&Claim{Name: "c", DeviceCount: 2, Node: "n"}, gib(10), "job-a")
	claim.Book("job-b", gib(4))

	// Joining with 6Gi: workers are 10, 4, 6; the 4Gi one stays resident, so
	// 16Gi parks per device, over 2 devices.
	if got, want := claim.ParkedBytesWith(gib(6)), gib(32); got != want {
		t.Errorf("ParkedBytesWith = %d, want %d", got, want)
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
	full.Claims["c"] = booked(&Claim{Name: "c", DeviceCount: 1, Node: "n"}, gib(20), "job-a")
	got = Explain(trainer("w", 6), full, "pod is unschedulable")
	if !strings.HasPrefix(got, "WaitingForCapacity") || !strings.Contains(got, "pod is unschedulable") {
		t.Errorf("Explain = %q, want WaitingForCapacity carrying the caller's detail", got)
	}
}
