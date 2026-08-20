package placement

import (
	"fmt"
	"sort"
)

// Size is one device shape present in the fleet: how big the device is, and
// the most of them any single node offers. This is the catalog -- what
// exists, never what is free. A stale catalog costs a dead or missing tier;
// it can never double-book, which is why catalog reads need no lock and no
// consistent reader.
type Size struct {
	DeviceMemoryBytes int64
	MaxPerNode        int
}

// Tier is one acceptable outcome for a claim, in preference order: count
// devices, each within [floor, ceiling]. The ceiling is the device size the
// tier is priced against, so the allocator cannot substitute a bigger
// device placement never chose.
type Tier struct {
	Name         string
	Count        int
	FloorBytes   int64
	CeilingBytes int64
}

// Catalog is the fleet's device shapes as seen by one role: what exists,
// never what is free. Sizes merge across nodes; the per-node device count
// bound keeps a tier from asking one node for more devices than any single
// node of that size offers.
func Catalog(fleet *Fleet, role string) []Size {
	bySize := map[int64]int{}
	for _, node := range fleet.Nodes {
		if !node.Accepts(role) || node.DeviceMemoryBytes <= 0 || node.DeviceCount < 1 {
			continue
		}
		if node.DeviceCount > bySize[node.DeviceMemoryBytes] {
			bySize[node.DeviceMemoryBytes] = node.DeviceCount
		}
	}
	catalog := make([]Size, 0, len(bySize))
	for memory, count := range bySize {
		catalog = append(catalog, Size{DeviceMemoryBytes: memory, MaxPerNode: count})
	}
	return catalog
}

// Tiers compiles the placement preference into the claim itself: one tier
// per device size the workload's declared shape can use, ordered fewest
// devices first, then least unused memory, then size. This is the old
// pool survey with the free-device half deleted -- the catalog says what
// could satisfy the workload; whether anything is free is the allocator's
// question, answered at allocation time.
func Tiers(req Request, catalog []Size) []Tier {
	seen := map[string]bool{}
	var tiers []Tier
	for _, size := range catalog {
		if size.DeviceMemoryBytes <= 0 || size.MaxPerNode < 1 {
			continue
		}
		count := devicesFor(req.Memory, size.DeviceMemoryBytes)
		// Never a width the runtime cannot drive, and claims stay one-node.
		if count > req.widest() || count > size.MaxPerNode {
			continue
		}
		tier := Tier{
			Name:         fmt.Sprintf("t%dx%d", count, CeilGiB(size.DeviceMemoryBytes)),
			Count:        count,
			FloorBytes:   req.PerDeviceBytes(count),
			CeilingBytes: size.DeviceMemoryBytes,
		}
		if seen[tier.Name] {
			continue
		}
		seen[tier.Name] = true
		tiers = append(tiers, tier)
	}

	sort.Slice(tiers, func(i, j int) bool {
		a, b := tiers[i], tiers[j]
		if a.Count != b.Count {
			return a.Count < b.Count
		}
		wasteA := int64(a.Count)*a.CeilingBytes - req.Memory
		wasteB := int64(b.Count)*b.CeilingBytes - req.Memory
		if wasteA != wasteB {
			return wasteA < wasteB
		}
		return a.CeilingBytes < b.CeilingBytes
	})
	return tiers
}
