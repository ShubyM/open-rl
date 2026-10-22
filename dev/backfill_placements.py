"""Write placement records for runs from before the API server recorded them, using the old reconstruction.

The dashboard used to rebuild where a run ran from worker records, Workloads, DRA claims and command records.
This runs that reconstruction once and stores each lane as the record the API server now writes
(server.telemetry.placement_of), so older runs stay loadable. Run it with the source of 48296c07, the last commit
with the reconstruction, on the path, against the store to backfill:

  mkdir /tmp/openrl-base && git archive 48296c07 src | tar -x -C /tmp/openrl-base
  OPEN_RL_OPS_TTL_DAYS=0 PYTHONPATH=/tmp/openrl-base/src REDIS_URL=redis://HOST:PORT/0 python dev/backfill_placements.py

Kubernetes comes from kubeconfig, for placements still live. Records already in the store are kept, so a rerun adds nothing.
"""

import asyncio
import json

from server import telemetry
from server.dashboard import inspection, snapshot
from server.store import get_state_store
from server.worker_manager import owner_id

OPEN_PLACEMENTS = "open_rl:placements:open"  # as in server.telemetry, which 48296c07 predates
KEEP = 10**6  # a list limit no backfill reaches


def placements_key(model_id: str) -> str:
  return f"open_rl:placements:{model_id}"


def placement(lane: dict, runtime: str, uuids: dict[str, str]) -> dict:
  """A reconstructed lane as a placement record. A lane names a GPU by DRA device or by UUID."""
  named = lane.get("device_uuids") or {}
  model_id = lane.get("runtime_id") or runtime
  return {
    "workload": lane["name"],
    "role": lane.get("role"),
    "model_id": model_id,
    "owner": owner_id(model_id),
    "pod": lane.get("pod"),
    "node": lane.get("node") or None,
    "gpu_uuids": [uuid for device in lane["devices"] if (uuid := named.get(device) or uuids.get(device) or device).startswith("GPU-")],
    "placed_at": lane["start"],
    "released_at": lane["end"] if lane["ended"] else None,
  }


async def main() -> None:
  store = get_state_store()
  snapshot.HISTORY_WINDOW_SECONDS = float("inf")  # every worker record, not the dashboard's seven days
  state = await snapshot.Snapshot().read(store)
  uuids = {d["id"]: d["uuid"] for node in state["cluster"]["nodes"] for d in node["devices"] if d.get("uuid")}
  found: dict[tuple, dict] = {}
  for run in state["runs"]:
    run_id = run["run_id"]
    samples = [*await store.read_samples(telemetry.ops_key(run_id)), *await store.read_samples(telemetry.samples_key(run_id))]
    for lane in inspection.sources(state, run_id, samples=samples):
      if lane.get("role") != "dispatcher":  # a pool's CPU pod is a Deployment, not a placement
        record = placement(lane, run["runtime_id"], uuids)
        found[record["model_id"], record["workload"], record["placed_at"]] = record
  added = {}
  for model_id in sorted({key[0] for key in found}):
    known = {(r["workload"], r["placed_at"]) for r in await store.read_samples(placements_key(model_id))}
    new = sorted((r for (m, *_), r in found.items() if m == model_id and (r["workload"], r["placed_at"]) not in known), key=lambda r: r["placed_at"])
    for record in new:
      await store.append_sample(placements_key(model_id), record, KEEP, telemetry.TTL_SECONDS)
    if any(r["released_at"] is None for r in new):
      await store.add_to_set(OPEN_PLACEMENTS, model_id)
    added[model_id] = len(new)
  print(json.dumps({"runs": len(state["runs"]), "records_found": len(found), "added": added, "cluster_error": state["error"]}, indent=2))


if __name__ == "__main__":
  asyncio.run(main())
