"""Fictional API fixtures for the real dashboard. No gateway, store, or cluster calls."""

import argparse
import math
import time
from datetime import UTC, datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from server.dashboard.gke import time_range
from server.dashboard.router import asset, dashboard, inspection_index


def iso(at):
  return datetime.fromtimestamp(at, UTC).isoformat()


def fixtures(now):
  nodes = [
    {
      "name": f"gpu-node-{suffix}",
      "gpu_capacity": 8 if i < 2 else 1,
      "accelerator": "nvidia-l4" if i == 3 else "nvidia-h100",
      "ready": i != 9,
      "devices": [{"id": f"pool-{i}/gpu-{j}", "name": f"GPU {j}"} for j in range(8 if i < 2 else 1)],
    }
    for i, suffix in enumerate(["a3f7", "b8c2", "c1d9", "d4e6", "e7a2", "f2b8", "19c4", "28d5", "37e6", "46f7"])
  ]
  runs = []
  for i, (run_id, name) in enumerate(
    [
      ("8ad9c2e1-55f1-4a2a-9b0c-0b40749f9ce7", None),
      ("34e512fa-a8b7-428b-8be0-2a08e36db814", "text2sql_try_17_lr1e-5_batch32_seed42__rerun_after_oom_2026-09-09"),
      ("ff4d98b2-7c36-4b13-97e1-5e29e79fc185", "eval"),
      ("17ab2d84-0986-4e9e-834f-b5ccdf819982", "train_final_FINAL_v3_actually_final"),
      ("61ba9c24-6e43-418b-9ad8-1b527fc0a883", None),
      ("28fe03ac-28ad-47f2-a67c-658a4f657010", "eval"),
      ("c16e49b7-23e8-44e2-958a-0d34eaf2d007", "wip"),
      ("c609bd82-54d4-4cc1-958b-934a812799cb", "baseline"),
      ("d817fa51-20cd-4406-9efc-98e671c380e8", "queued experiment"),
      ("d286ae71-3e1e-4a09-b13e-1892238ad4ea", "train_final_FINAL_v2"),
    ]
  ):
    model = "google/gemma-3-4b-it" if i in (3, 9) else "Qwen/Qwen3-8B"
    runs.append(
      {
        "run_id": run_id,
        "model": model,
        "name": name or f"{model.split('/')[-1]} · {run_id[:8]}",
        "display_name": name,
        "recipe_name": {0: "math_rl", 3: "gsm8k_sft"}.get(i),
        "fine_tuning_type": "lora" if i in (1, 3) else "full",
        "status": {7: "completed", 9: "failed"}.get(i, "active"),
        "display_status": {7: "Completed", 8: "Queued", 9: "Failed"}.get(i, "Running"),
        "created_at": now - (480 if i == 8 else 9600 + i * 420),
        "completed_at": now - 60 if i in (7, 9) else None,
        "steps": 0 if i in (1, 2, 5, 6, 8) else 42 + i * 30,
        "shared_runtime": i in (1, 3),
        "pods": [],
        "nodes": [],
        "workloads": [],
      }
    )
  allocations = []
  # run, node, first GPU, GPU count, start/end minutes within the fixture window.
  for ordinal, (r, n, first, count, start, end) in enumerate(
    [(0, 0, 0, 4, 0, 30), (1, 0, 4, 2, 8, 30), (0, 1, 0, 2, 0, 30)]
    + [(i, i, 0, 1, start, 30) for i, start in zip(range(2, 7), [13, 9, 3, 14, 18], strict=True)]
    + [(7, 7, 0, 1, 1, 18), (9, 9, 0, 1, 5, 25)]
  ):
    run, node = runs[r], nodes[n]
    role = "trainer" if r in (0, 3, 4, 7, 9) else "sampler"
    uid = f"demo-workload-{ordinal}"
    run["runtime_id"] = run["model"] if run["shared_runtime"] else run["run_id"]
    started, finished = now - 1800 + start * 60, now - 1800 + end * 60
    pod = {
      "uid": f"demo-pod-{ordinal}",
      "name": f"{role}-{run['run_id'][:8]}-{ordinal}",
      "node": node["name"],
      "role": role,
      "phase": "Running" if end == 30 else "Succeeded" if r == 7 else "Failed",
      "restarts": 1 if r == 3 else 0,
      "problem": "Recovered after OOMKilled" if r == 3 else "Failed" if r == 9 else None,
      "events": [],
      "runtime_id": run["runtime_id"],
      "shared_runtime": run["shared_runtime"],
      "owner_references": [{"uid": uid}],
      "created_at": iso(started),
      "containers": [{"name": role, "state": "running", "restart_count": 0, "started_at": iso(now - 1800 + start * 60)}],
    }
    if end < 30:
      run["completed_at"] = finished
      pod["containers"][0].update(
        state="terminated", reason="Completed" if r == 7 else "OOMKilled", exit_code=0 if r == 7 else 137, finished_at=iso(finished)
      )
    if r == 3:
      pod["containers"][0].update(
        restart_count=1,
        started_at=iso(now - 30),
        last_termination={"reason": "OOMKilled", "exit_code": 137, "finished_at": iso(now - 39)},
      )
    workload = {
      "uid": uid,
      "name": pod["name"],
      "owner_id": run["run_id"],
      "model_id": run["runtime_id"],
      "role": role,
      "training_kind": "lora" if run["fine_tuning_type"] == "lora" else "fft",
      "node_name": node["name"],
      "phase": "Completed" if pod["phase"] == "Succeeded" else pod["phase"],
      "requested_memory": f"{count * 40}Gi",
      "placed_message": "Assigned to claim",
      "exclusive": r != 1,
    }
    run["pods"].append(pod)
    run["nodes"].append(node["name"])
    run["workloads"].append(workload)
    allocations.append(
      {
        "id": uid,
        "label": run["name"],
        "runtime_id": run["runtime_id"],
        "owner_id": run["run_id"],
        "run_ids": [run["run_id"]],
        "node": node["name"],
        "role": role,
        "device_count": count,
        "devices": [d["id"] for d in node["devices"][first : first + count]],
        "phase": "Running",
        "exclusive": workload["exclusive"],
        "start": now - 1800 + start * 60,
        "end": finished + (15 if end == 30 else 0),
      }
    )
  runs[8]["workloads"] = [
    {
      "uid": "demo-pending",
      "name": "pending-trainer",
      "owner_id": runs[8]["run_id"],
      "role": "trainer",
      "phase": "Pending",
      "requested_memory": "160Gi",
      "placed_message": "Waiting for two GPUs with sufficient unreserved memory",
    }
  ]
  for run in runs:
    run.setdefault("runtime_id", run["run_id"])
    run["runtime_run_ids"] = [r["run_id"] for r in runs if r.get("runtime_id") == run["runtime_id"]]

  def at_time(at):
    return [p for p in allocations if p["start"] <= at < p["end"]]

  return {
    "schema_version": 1,
    "demo": True,
    "observed_at": iso(now),
    "coverage": "Fictional data; 30-minute fixture window",
    "runs": runs,
    "placements": at_time(now),
    "history": [{"at": at, "available": True, "placements": at_time(at)} for at in range(now - 1800, now + 1, 15)],
    "cluster": {
      "available": True,
      "nodes": nodes,
      "pods": [p for r in runs for p in r["pods"]],
      "scheduler": {
        "available": True,
        "workloads": [w for r in runs for w in r["workloads"]],
        "ledgers": [
          {
            "name": p["id"],
            "claim_name": f"claim-{p['id']}",
            "seats": [
              {
                "workload_uid": p["id"],
                "workload": p["label"],
                "owner": p["owner_id"],
                "exclusive": p["exclusive"],
              }
            ],
          }
          for p in at_time(now)
        ],
      },
    },
  }, allocations


def run_log_records(run, start, end):
  records = []
  for pod in run["pods"]:
    container = pod["containers"][0]
    born = datetime.fromisoformat(pod["created_at"]).timestamp()
    stopped = min(NOW, datetime.fromisoformat(container["finished_at"]).timestamp()) if container.get("finished_at") else NOW
    entries = [(born + 10, "[WORKER] Model loaded; request polling started.")]
    if pod["role"] == "sampler":
      entries += [(stopped - 10, "[SAMPLER] Sample request completed; generated 128 tokens.")]
    else:
      entries += [(stopped - 69, f"Saved checkpoint /tmp/open-rl/checkpoints/{run['run_id']}/step-{run['steps']}")]
    if pod["restarts"] or container.get("reason") == "OOMKilled":
      failed_at = datetime.fromisoformat(container.get("last_termination", container)["finished_at"]).timestamp()
      entries += [(failed_at, "Error in training requests processor: CUDA out of memory. Tried to allocate 256.00 MiB.")]
    if pod["restarts"]:
      entries += [(datetime.fromisoformat(container["started_at"]).timestamp(), "[WORKER] Requests processor started.")]
    records += [
      {
        "timestamp": iso(at),
        "pod": pod["name"],
        "pod_uid": pod["uid"],
        "container": pod["role"],
        "node": pod["node"],
        "role": pod["role"],
        "attempt": int(pod["restarts"] > 0 and at >= datetime.fromisoformat(container["started_at"]).timestamp()),
        "severity": "ERROR" if "out of memory" in message else "INFO",
        "message": message,
      }
      for at, message in entries
      if start <= at <= end and born <= at <= stopped
    ]
  return sorted(records, key=lambda record: record["timestamp"], reverse=True)


NOW = int(time.time())
STATE, ALLOCATIONS = fixtures(NOW)
app = FastAPI(title="OpenRL demo — fictional data")


@app.get("/")
@app.get("/preview.html")
async def preview():
  return RedirectResponse("/dashboard")


@app.get("/api/v1/dashboard")
async def index():
  return {**await inspection_index(), "demo": True, "scope": {"namespace": "fictional", "identity": "demo"}}


@app.get("/api/v1/dashboard/{path:path}", include_in_schema=False)
async def fixture_api(path: str, request: Request):
  parts, query = path.split("/"), request.query_params
  if path == "snapshot":
    return STATE
  try:
    start, end = time_range(query.get("since", iso(NOW - 1800)), query.get("until", iso(NOW)))
    start, end = datetime.fromisoformat(start).timestamp(), datetime.fromisoformat(end).timestamp()
    limit = min(1000, max(1, int(query.get("tail", query.get("limit", "200")))))
    offset = max(0, int(query.get("cursor", "0")))
  except ValueError as exc:
    raise HTTPException(400, str(exc)) from exc
  if len(parts) == 3 and parts[0] == "allocations" and parts[2] == "metrics":
    placement = next((p for p in ALLOCATIONS if p["id"] == parts[1]), None)
    if placement is None:
      raise HTTPException(404)
    devices = []
    for i, device in enumerate(placement["devices"]):
      points = [
        [at, round(max(0, min(100, (70 + 16 * math.sin(at / 90 + i) + 5 * math.sin(at / 25)) * (0.48 if i == 3 else 1))), 1)]
        for at in range(NOW - 1800, NOW + 1, 15)
        if start <= at <= end and placement["start"] <= at <= placement["end"]
      ]
      devices.append({"id": device, "utilization": points, "memory_mib": [[at, 55424 + i * 128] for at, _ in points]})
    return {
      "demo": True,
      "available": any(device["utilization"] for device in devices),
      "devices": devices,
      "mfu": {"value": 0.404, "estimated": True, "scope": "run", "device_count": 6}
      if any(device["utilization"] for device in devices) and placement["run_ids"] == [STATE["runs"][0]["run_id"]]
      else None,
    }
  if len(parts) in (2, 3) and parts[0] == "runs":
    run = next((r for r in STATE["runs"] if r["run_id"] == parts[1]), None)
    if run is None:
      raise HTTPException(404)
    if len(parts) == 2:
      return {"demo": True, "schema_version": 1, "observed_at": iso(NOW), "coverage": STATE["coverage"], **run}
    if parts[2] == "metrics":
      placements = [p for p in ALLOCATIONS if run["run_id"] in p["run_ids"]]
      role = run["pods"][0]["role"] if run["pods"] else None
      samples = [
        {
          "at": at,
          "operation": "sample" if role == "sampler" else "optim_step",
          "role": role,
          "run_id": run["run_id"],
          "runtime_id": run["runtime_id"],
          "request_id": f"demo-request-{i}",
          "status": "succeeded",
          "elapsed_seconds": round(0.28 + i * 0.01, 2) if role == "sampler" else round(2.1 + i * 0.04 + (8 if i == 23 else 0), 2),
          "metrics": {}
          if role == "sampler"
          else {"loss:mean": round(1.32 - i * 0.0174, 3), "grad_norm": round(0.68 - i * 0.0086 + (3.2 if i == 23 else 0), 3)},
        }
        for i, at in enumerate(range(NOW - 1800, NOW + 1, 60))
        if start <= at <= end and any(p["start"] < at < p["end"] for p in placements)
      ]
      for sample in samples:
        sample["started_at"] = sample["at"] - sample["elapsed_seconds"]
      return {"demo": True, "available": bool(samples), "samples": samples, "coverage": STATE["coverage"]}
    if parts[2] == "logs":
      records = [
        record
        for record in run_log_records(run, start, end)
        if query.get("q", "").lower() in record["message"].lower()
        and all(not query.get(key) or str(record[key]) == query[key] for key in ("pod", "container", "node", "role", "attempt", "severity"))
      ]
      return {
        "demo": True,
        "source": "demo",
        "records": records[offset : offset + limit],
        "next_cursor": str(offset + limit) if offset + limit < len(records) else None,
        "sources": [],
        "coverage": {"source": "fictional"},
        "shared_runtime": run["shared_runtime"],
        "runtime_run_ids": run["runtime_run_ids"],
      }
  if len(parts) == 3 and parts[0] == "pods" and parts[2] == "logs":
    run = next((run for run in STATE["runs"] if any(pod["name"] == parts[1] for pod in run["pods"])), None)
    if run is None:
      raise HTTPException(404)
    pod = next(pod for pod in run["pods"] if pod["name"] == parts[1])
    previous = query.get("previous", "false").lower() == "true"
    records = [
      record
      for record in run_log_records(run, start, end)
      if record["pod"] == pod["name"]
      and record["attempt"] == pod["restarts"] - int(previous)
      and (not query.get("container") or record["container"] == query["container"])
    ]
    return {
      "demo": True,
      "pod": pod["name"],
      "container": query.get("container"),
      "previous": previous,
      "text": "\n".join(f"{record['timestamp']} {record['message']}" for record in reversed(records[:limit])),
    }
  # Never fall through to a real cluster/store endpoint, even for unknown paths.
  raise HTTPException(404, "No fixture for this endpoint")


# Only mount the production UI handlers: live API routes are never registered here.
for path, endpoint in [("/dashboard", dashboard), ("/dashboard/", dashboard), ("/dashboard/assets/{name}", asset)]:
  app.add_api_route(path, endpoint, include_in_schema=False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=9017)
  args = parser.parse_args()
  uvicorn.run(app, host=args.host, port=args.port)
