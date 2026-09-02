package placement

import (
	"math/rand"
	"testing"
)

// Property fuzz over the tier compiler: whatever the catalog and request,
// every emitted tier must be individually satisfiable and the list must be
// ordered the way the claim relies on -- fewest devices, then least waste.
// 5000 random cases per run, deterministic seed.
func TestTiersProperties(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for i := 0; i < 5000; i++ {
		catalog := make([]Size, 1+rng.Intn(4))
		for j := range catalog {
			catalog[j] = Size{
				DeviceMemoryBytes: gib(int64(1 + rng.Intn(96))),
				MaxPerNode:        rng.Intn(9), // 0 is legal input: an empty pool contributes nothing
			}
		}
		req := Request{
			Role:       "trainer",
			WorkerID:   "w",
			Memory:     gib(int64(1 + rng.Intn(200))),
			MaxDevices: rng.Intn(5), // 0 means one
		}
		tiers := Tiers(req, catalog)

		widest := req.MaxDevices
		if widest < 1 {
			widest = 1
		}
		maxPerSize := map[int64]int{}
		for _, size := range catalog {
			if size.MaxPerNode > maxPerSize[size.DeviceMemoryBytes] {
				maxPerSize[size.DeviceMemoryBytes] = size.MaxPerNode
			}
		}

		names := map[string]bool{}
		for k, tier := range tiers {
			if tier.Count < 1 || tier.Count > widest {
				t.Fatalf("case %d: tier %+v has count outside [1,%d]", i, tier, widest)
			}
			if tier.Count > maxPerSize[tier.CeilingBytes] {
				t.Fatalf("case %d: tier %+v asks one node for more devices than any node of that size offers (%d)",
					i, tier, maxPerSize[tier.CeilingBytes])
			}
			if int64(tier.Count)*tier.CeilingBytes < req.Memory {
				t.Fatalf("case %d: tier %+v cannot hold %d bytes even full", i, tier, req.Memory)
			}
			if tier.FloorBytes > tier.CeilingBytes {
				t.Fatalf("case %d: tier %+v floor exceeds its ceiling", i, tier)
			}
			if int64(tier.Count)*tier.FloorBytes < req.Memory {
				t.Fatalf("case %d: tier %+v floor share does not cover the request", i, tier)
			}
			if names[tier.Name] {
				t.Fatalf("case %d: duplicate tier name %q in %+v", i, tier.Name, tiers)
			}
			names[tier.Name] = true

			if k == 0 {
				continue
			}
			prev := tiers[k-1]
			if tier.Count < prev.Count {
				t.Fatalf("case %d: tiers out of order by count: %+v before %+v", i, prev, tier)
			}
			if tier.Count == prev.Count {
				prevWaste := int64(prev.Count)*prev.CeilingBytes - req.Memory
				waste := int64(tier.Count)*tier.CeilingBytes - req.Memory
				if waste < prevWaste {
					t.Fatalf("case %d: tiers out of order by waste: %+v before %+v", i, prev, tier)
				}
			}
		}
	}
}
