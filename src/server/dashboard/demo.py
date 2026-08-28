# Fictional demo payloads for the dashboard. Everything here is invented so the UI can be
# developed and demoed without a cluster; every payload carries demo=True and the UI is
# required to label it as fictional.

import math
import time

from server.dashboard.data import derive_problems, run_diagnostics, scheduler_run_diagnostics

DEMO_NOTICE = "Demo data — every machine, pod, and run on this page is fictional."


def demo_duty_series(capacity: int, jobs: dict[str, tuple[int, int]], seed: int) -> dict:
  """A plausible fictional per-job duty series: 10 minutes of samples every 10s. `jobs` maps
  each job to (base GPUs, wobble amplitude); claims are clamped to pool capacity."""
  now = int(time.time())
  series = []
  for i in range(60):
    claims = {}
    for j, (job, (base, amplitude)) in enumerate(jobs.items()):
      value = round(base + amplitude * math.sin((i + seed * 5 + j * 9) / 6))
      if value > 0:
        claims[job] = value
    while sum(claims.values()) > capacity:
      biggest = max(claims, key=claims.get)
      claims[biggest] -= 1
    series.append([now - (59 - i) * 10, {job: gpus for job, gpus in claims.items() if gpus > 0}])
  current = round(sum(series[-1][1].values()) / capacity, 4)
  return {"capacity": capacity, "current": current, "jobs": list(jobs), "series": series}


def demo_scheduler() -> dict:
  workloads = [
    {
      "name": "trainer-demo-run-1",
      "uid": "demo-workload-1",
      "created_at": "2026-07-29T06:02:00+00:00",
      "age_seconds": 540,
      "deleting": False,
      "generation": 1,
      "role": "trainer",
      "model_id": "demo-run-1",
      "owner_id": "demo-run-1",
      "training_kind": "fft",
      "requested_memory": "72Gi",
      "max_devices": 1,
      "phase": "Running",
      "reason": "Allocated",
      "claim_name": "openrl-h100-demo-1",
      "assignment_id": "assignment-demo-1",
      "pod_name": "open-rl-trainer-demo-run-1",
      "node_name": "demo-h100-node-1",
      "device_count": 1,
      "memory_per_device": "72Gi",
      "observed_generation": 1,
      "generation_current": True,
      "placed": True,
      "placed_reason": "Allocated",
      "placed_message": "claim allocated on demo-h100-node-1",
      "placed_transition_at": "2026-07-29T06:02:18+00:00",
    },
    {
      "name": "trainer-demo-run-2",
      "uid": "demo-workload-2",
      "created_at": "2026-07-29T07:41:00+00:00",
      "age_seconds": 420,
      "deleting": False,
      "generation": 1,
      "role": "trainer",
      "model_id": "demo-run-2",
      "owner_id": "demo-run-2",
      "training_kind": "fft",
      "requested_memory": "40Gi",
      "max_devices": 1,
      "phase": "Pending",
      "reason": "No tier has a free claim and host memory is full",
      "claim_name": None,
      "assignment_id": None,
      "pod_name": "open-rl-trainer-demo-run-2",
      "node_name": None,
      "device_count": 0,
      "memory_per_device": None,
      "observed_generation": 1,
      "generation_current": True,
      "placed": False,
      "placed_reason": "NoCapacity",
      "placed_message": "waiting for a free claim",
      "placed_transition_at": "2026-07-29T07:41:00+00:00",
    },
  ]
  ledgers = [
    {
      "name": "openrl-h100-demo-1",
      "created_at": "2026-07-29T06:02:00+00:00",
      "age_seconds": 540,
      "claim_name": "openrl-h100-demo-1",
      "seat_count": 1,
      "owners": ["demo-run-1"],
      "seats": [
        {
          "workload": "trainer-demo-run-1",
          "workload_uid": "demo-workload-1",
          "assignment_id": "assignment-demo-1",
          "owner": "demo-run-1",
          "host_request": "24Gi",
        }
      ],
    }
  ]
  return {
    "installed": True,
    "available": True,
    "error": None,
    "workloads": workloads,
    "ledgers": ledgers,
    "summary": {"workloads": 2, "phase_counts": {"Running": 1, "Pending": 1}, "ledgers": 1, "seats": 1, "shared_ledgers": 0},
  }


def demo_cluster() -> dict:
  snapshot = {
    "demo": True,
    "notice": DEMO_NOTICE,
    "kubernetes": {
      "available": True,
      "namespace": "open-rl-demo",
      "error": None,
      "metrics": {
        "installed": True,
        "available": True,
        "error": None,
        "pods_available": True,
        "nodes_available": True,
        "pods_observed": 5,
        "nodes_observed": 5,
      },
    },
    "gateway": {
      "title": "open-rl gateway",
      "mode": "distributed",
      "fft_enabled": True,
      "redis_configured": True,
      "vllm_url": None,
      "sampler_backend": "vllm",
    },
    "scheduler": demo_scheduler(),
    "services": [
      {"id": "redis", "label": "Redis", "configured": True, "ok": True, "detail": "redis://demo-redis:6379"},
      {"id": "storage", "label": "Shared storage", "configured": True, "ok": True, "detail": "/mnt/shared/open-rl"},
    ],
    "edges": [{"from": "gateway", "to": "redis", "reason": "REDIS_URL configured"}],
    "pools": [
      {
        "id": "nvidia-h100-80gb",
        "label": "nvidia-h100-80gb",
        "duty": demo_duty_series(16, {"demo-run-1": (8, 0), "demo-run-2": (4, 3), "other": (1, 1)}, seed=1),
        "nodes": [
          {
            "name": "demo-h100-node-1",
            "ready": True,
            "instance_type": "a3-highgpu-8g",
            "gpu_capacity": 8,
            "gpu_allocatable": 8,
            "pods": ["open-rl-trainer-demo-run-1", "open-rl-trainer-demo-run-2"],
          },
          {
            "name": "demo-h100-node-2",
            "ready": True,
            "instance_type": "a3-highgpu-8g",
            "gpu_capacity": 8,
            "gpu_allocatable": 8,
            "pods": ["open-rl-sampler-demo-run-1"],
          },
        ],
      },
      {
        "id": "nvidia-l4",
        "label": "nvidia-l4",
        "duty": demo_duty_series(4, {"demo-run-2": (2, 1), "other": (1, 1)}, seed=2),
        "nodes": [
          {
            "name": "demo-l4-node-1",
            "ready": True,
            "instance_type": "g2-standard-24",
            "gpu_capacity": 2,
            "gpu_allocatable": 2,
            "pods": ["open-rl-sampler-demo-run-2"],
          },
          {
            "name": "demo-l4-node-2",
            "ready": False,
            "instance_type": "g2-standard-24",
            "gpu_capacity": 2,
            "gpu_allocatable": 0,
            "pods": [],
          },
        ],
      },
      {
        "id": "cpu",
        "label": "cpu",
        "nodes": [
          {
            "name": "demo-cpu-node-1",
            "ready": True,
            "instance_type": "n2-standard-8",
            "gpu_capacity": 0,
            "gpu_allocatable": 0,
            "pods": ["open-rl-gateway-7f9c4", "demo-redis-0"],
          }
        ],
      },
    ],
    "pods": [
      {
        "name": "open-rl-gateway-7f9c4",
        "phase": "Running",
        "node": "demo-cpu-node-1",
        "app": "open-rl-gateway",
        "ready": "1/1",
        "restarts": 0,
        "created_at": "2026-07-27T09:12:00+00:00",
        "problem": None,
        "containers": [{"name": "gateway", "image": "gcr.io/demo/open-rl-server:demo", "ready": True, "state": "running"}],
      },
      {
        "name": "demo-redis-0",
        "phase": "Running",
        "node": "demo-cpu-node-1",
        "app": "redis",
        "ready": "1/1",
        "restarts": 0,
        "created_at": "2026-07-27T09:10:00+00:00",
        "problem": None,
        "containers": [{"name": "redis", "image": "redis:7", "ready": True, "state": "running"}],
      },
      {
        "name": "open-rl-trainer-demo-run-1",
        "phase": "Running",
        "node": "demo-h100-node-1",
        "app": "open-rl-trainer-worker",
        "ready": "1/1",
        "restarts": 0,
        "created_at": "2026-07-29T06:02:00+00:00",
        "problem": None,
        "containers": [{"name": "trainer", "image": "gcr.io/demo/open-rl-server:demo", "ready": True, "state": "running"}],
      },
      {
        "name": "open-rl-trainer-demo-run-2",
        "phase": "Pending",
        "node": "demo-h100-node-1",
        "app": "open-rl-trainer-worker",
        "ready": "0/1",
        "restarts": 0,
        "created_at": "2026-07-29T07:41:00+00:00",
        "problem": "Unschedulable: waiting for a free GPU claim",
        "containers": [{"name": "trainer", "image": "gcr.io/demo/open-rl-server:demo", "ready": False, "state": "waiting"}],
      },
      {
        "name": "open-rl-sampler-demo-run-1",
        "phase": "Running",
        "node": "demo-h100-node-2",
        "app": "open-rl-sampler-worker",
        "ready": "1/1",
        "restarts": 2,
        "created_at": "2026-07-29T06:03:00+00:00",
        "problem": None,
        "containers": [{"name": "sampler", "image": "gcr.io/demo/open-rl-server:demo", "ready": True, "state": "running"}],
      },
      {
        "name": "open-rl-sampler-demo-run-2",
        "phase": "Failed",
        "node": "demo-l4-node-1",
        "app": "open-rl-sampler-worker",
        "ready": "0/1",
        "restarts": 4,
        "created_at": "2026-07-28T22:17:00+00:00",
        "reason": "Error",
        "message": "sampler repeatedly exceeded its GPU memory limit",
        "problem": "CrashLoopBackOff: CUDA out of memory",
        "containers": [
          {
            "name": "sampler",
            "kind": "app",
            "image": "gcr.io/demo/open-rl-server:demo",
            "ready": False,
            "state": "waiting",
            "reason": "CrashLoopBackOff",
            "message": "back-off restarting failed container",
            "exit_code": None,
            "restart_count": 4,
            "last_termination": {"reason": "OOMKilled", "message": None, "exit_code": 137, "signal": 0},
          }
        ],
        "conditions": [{"type": "Ready", "status": "False", "reason": "ContainersNotReady", "message": "sampler is not ready"}],
        "events": [
          {
            "reason": "BackOff",
            "message": "Back-off restarting failed container sampler",
            "type": "Warning",
            "count": 4,
            "source": "kubelet",
            "last_seen_at": "2026-07-29T08:02:00+00:00",
          }
        ],
      },
    ],
  }
  node_usage = {
    "demo-h100-node-1": (19.4, 148 * 2**30),
    "demo-h100-node-2": (8.7, 96 * 2**30),
    "demo-l4-node-1": (5.2, 34 * 2**30),
    "demo-l4-node-2": (0.8, 9 * 2**30),
    "demo-cpu-node-1": (0.43, 3 * 2**30),
  }
  for pool in snapshot["pools"]:
    for node in pool["nodes"]:
      cpu, memory = node_usage[node["name"]]
      node["usage"] = {"cpu_cores": cpu, "memory_bytes": memory}
  pod_usage = {
    "open-rl-gateway-7f9c4": (0.35, 450 * 2**20),
    "demo-redis-0": (0.08, 80 * 2**20),
    "open-rl-trainer-demo-run-1": (6.4, 58 * 2**30),
    "open-rl-sampler-demo-run-1": (3.1, 46 * 2**30),
    "open-rl-sampler-demo-run-2": (0.2, 2 * 2**30),
  }
  for pod in snapshot["pods"]:
    if pod["name"] in pod_usage:
      cpu, memory = pod_usage[pod["name"]]
      pod["usage"] = {"cpu_cores": cpu, "memory_bytes": memory}
    else:
      pod["usage"] = None
  return snapshot


def demo_runs() -> dict:
  return {
    "demo": True,
    "notice": DEMO_NOTICE,
    "runs": [
      {
        "run_id": "demo-run-1",
        "name": "math-rl-qwen3-8b",
        "base_model": "Qwen/Qwen3-8B",
        "created_at": "2026-07-29T06:02:00+00:00",
        "wandb_url": "https://wandb.ai/example/open-rl/runs/demo-run-1",
        "stoppable": True,
        "sources": ["worker", "queue"],
        "pods": ["open-rl-trainer-demo-run-1", "open-rl-sampler-demo-run-1"],
        "queue_depth": 5,
        "queue_oldest_at": "2026-07-29T08:01:42+00:00",
        "queue_oldest_seconds": 18,
        "worker_alive": True,
        "telemetry": {
          "requests_completed": 42,
          "requests_failed": 0,
          "failure_rate": 0.0,
          "operation_counts": {"create_model": 1, "forward_backward": 20, "optim_step": 20, "save_weights_for_sampler": 1},
          "last_operation": "forward_backward",
          "last_outcome": "ok",
          "last_latency_seconds": 3.42,
          "mean_latency_seconds": 2.18,
          "max_latency_seconds": 7.91,
          "active_request": {
            "request_id": "demo-request-active",
            "operation": "forward_backward",
            "started_at": 1785333775.0,
            "queue_wait_seconds": 0.18,
            "age_seconds": 4.2,
          },
          "latest_metrics": {"loss:mean": 0.7981, "grad_norm:mean": 0.42},
          "metric_series": {
            "loss:mean": [
              {"at": 1785333650.0, "value": 0.9124, "operation": "forward_backward"},
              {"at": 1785333710.0, "value": 0.8312, "operation": "forward_backward"},
              {"at": 1785333770.0, "value": 0.7981, "operation": "forward_backward"},
            ],
            "grad_norm:mean": [
              {"at": 1785333680.0, "value": 0.48, "operation": "optim_step"},
              {"at": 1785333740.0, "value": 0.42, "operation": "optim_step"},
            ],
          },
        },
        "state": {
          "phase": "running",
          "status": "ok",
          "reason": "2 running pods",
          "pod_phase_counts": {"Running": 2},
          "workload_phase_counts": {"Running": 1},
        },
      },
      {
        "run_id": "demo-run-2",
        "name": "sft-gemma-warmup",
        "base_model": "google/gemma-3-4b-it",
        "created_at": "2026-07-29T07:41:00+00:00",
        "wandb_url": None,
        "stoppable": True,
        "sources": ["worker", "queue"],
        "pods": ["open-rl-trainer-demo-run-2", "open-rl-sampler-demo-run-2"],
        "queue_depth": 2,
        "queue_oldest_at": "2026-07-29T07:59:15+00:00",
        "queue_oldest_seconds": 75,
        "worker_alive": True,
        "telemetry": {
          "requests_completed": 7,
          "requests_failed": 2,
          "failure_rate": 2 / 7,
          "operation_counts": {"create_model": 1, "forward_backward": 4, "optim_step": 2},
          "last_operation": "forward_backward",
          "last_outcome": "error",
          "last_error": "CUDA out of memory while allocating the training batch",
          "last_error_at": 1785333770.0,
          "last_latency_seconds": 11.7,
          "mean_latency_seconds": 5.2,
          "max_latency_seconds": 11.7,
          "latest_metrics": {"loss:mean": 1.182},
          "metric_series": {"loss:mean": [{"at": 1785333710.0, "value": 1.182, "operation": "forward_backward"}]},
        },
        "state": {
          "phase": "failed",
          "status": "error",
          "reason": "CrashLoopBackOff: CUDA out of memory",
          "pod_phase_counts": {"Pending": 1, "Failed": 1},
          "workload_phase_counts": {"Pending": 1},
        },
      },
      {
        "run_id": "demo-run-3",
        "name": "run-9f31ab02",
        "base_model": "Qwen/Qwen3-8B",
        "created_at": "2026-07-26T18:20:00+00:00",
        "wandb_url": "https://wandb.ai/example/open-rl/runs/demo-run-3",
        "stoppable": False,
        "sources": ["checkpoint"],
        "pods": [],
        "queue_depth": 0,
        "queue_oldest_at": None,
        "queue_oldest_seconds": None,
        "worker_alive": None,
        "telemetry": {},
        "state": {
          "phase": "saved",
          "status": "off",
          "reason": "saved artifacts are present; no active worker is visible",
          "pod_phase_counts": {},
          "workload_phase_counts": {},
        },
      },
    ],
  }


def demo_health() -> dict:
  return {
    "demo": True,
    "notice": DEMO_NOTICE,
    "checks": [
      {"id": "gateway", "group": "Gateway", "label": "Gateway process", "status": "ok", "detail": "distributed mode, FFT enabled"},
      {"id": "storage.redis", "group": "Storage", "label": "Redis", "status": "ok", "detail": "PING 0.8 ms — redis://demo-redis:6379"},
      {
        "id": "storage.shared",
        "group": "Storage",
        "label": "Shared filesystem",
        "status": "ok",
        "detail": "/mnt/shared/open-rl writable, 412 GiB free",
      },
      {"id": "kubernetes", "group": "Kubernetes", "label": "API server", "status": "ok", "detail": "6 pods visible in namespace open-rl-demo"},
      {"id": "scheduler", "group": "Scheduler", "label": "Placement API", "status": "ok", "detail": "2 workloads, 1 claim ledger, 1 seat"},
      {
        "id": "visibility.trace",
        "group": "Visibility",
        "label": "Trace export",
        "status": "off",
        "detail": "ENABLE_GCP_TRACE=0 — tracing not configured",
      },
      {"id": "visibility.events", "group": "Visibility", "label": "Pod events", "status": "ok", "detail": "4 recent events visible"},
      {
        "id": "visibility.metrics",
        "group": "Visibility",
        "label": "Resource metrics",
        "status": "ok",
        "detail": "usage visible for 5 pods and 5 nodes",
      },
      {
        "id": "visibility.sampler",
        "group": "Visibility",
        "label": "vLLM sampler",
        "status": "error",
        "detail": "open-rl-sampler-demo-run-2 is failing",
      },
    ],
    "stats": [
      {
        "id": "runs.active",
        "label": "Active runs",
        "value": "2",
        "value_number": 2,
        "unit": "runs",
        "detail": "live worker or queued work",
        "context": {},
        "status": "ok",
      },
      {
        "id": "queue.requests",
        "label": "Queued requests",
        "value": "7",
        "value_number": 7,
        "unit": "requests",
        "detail": "across 2 queues",
        "context": {"queue_count": 2, "oldest_model_id": "demo-run-2", "oldest_age_seconds": 75},
        "status": "ok",
      },
      {
        "id": "queue.request_age",
        "label": "Oldest request wait",
        "value": "1m 15s",
        "value_number": 75,
        "unit": "seconds",
        "detail": "demo-run-2",
        "context": {"model_id": "demo-run-2", "warn_after_seconds": 300},
        "status": "ok",
      },
      {
        "id": "queue.launch",
        "label": "Launches pending",
        "value": "0",
        "value_number": 0,
        "unit": "runs",
        "detail": "worker launch queue",
        "context": {"oldest_age_seconds": 0},
        "status": "ok",
      },
      {
        "id": "queue.launch_age",
        "label": "Oldest launch wait",
        "value": "0s",
        "value_number": 0,
        "unit": "seconds",
        "detail": "no pending launches",
        "context": {"warn_after_seconds": 60},
        "status": "ok",
      },
      {
        "id": "redis.memory",
        "label": "Redis memory",
        "value": "48.2 MiB",
        "value_number": 50541363,
        "unit": "bytes",
        "detail": "peak 61.0 MiB · no maxmemory limit",
        "context": {"peak_bytes": 63963136, "limit_bytes": None, "utilization": None},
        "status": "ok",
      },
      {
        "id": "redis.clients",
        "label": "Redis clients",
        "value": "9",
        "value_number": 9,
        "unit": "clients",
        "detail": "connected",
        "context": {},
        "status": "ok",
      },
      {
        "id": "gateway.rss",
        "label": "Gateway memory",
        "value": "213.4 MiB",
        "value_number": 223766118,
        "unit": "bytes",
        "detail": "resident set size",
        "context": {},
        "status": "ok",
      },
      {
        "id": "storage.disk",
        "label": "Disk free",
        "value": "412.0 GiB",
        "value_number": 442381631488,
        "unit": "bytes",
        "detail": "of 1.0 TiB at /mnt/shared/open-rl",
        "context": {"total_bytes": 1099511627776, "free_ratio": 0.402, "path": "/mnt/shared/open-rl"},
        "status": "ok",
      },
      {
        "id": "pods.running",
        "label": "Pods running",
        "value": "4",
        "value_number": 4,
        "unit": "pods",
        "detail": "1 failed · 1 pending",
        "context": {"phase_counts": {"Running": 4, "Failed": 1, "Pending": 1}},
        "status": "warn",
      },
      {
        "id": "cluster.cpu",
        "label": "Cluster CPU",
        "value": "34.53 cores",
        "value_number": 34.53,
        "unit": "cores",
        "detail": "19% of 184.00 allocatable",
        "context": {"allocatable_cores": 184.0, "utilization": 0.1877, "measured_nodes": 5},
        "status": "ok",
      },
      {
        "id": "cluster.memory",
        "label": "Cluster memory",
        "value": "290.0 GiB",
        "value_number": 311385128960,
        "unit": "bytes",
        "detail": "22% of 1.3 TiB allocatable",
        "context": {"allocatable_bytes": 1429365116108, "utilization": 0.2178, "measured_nodes": 5},
        "status": "ok",
      },
      {
        "id": "gpus.claimed",
        "label": "GPUs claimed",
        "value": "13/20",
        "value_number": 13,
        "unit": "devices",
        "detail": "across all pools",
        "context": {"capacity_devices": 20, "allocation_ratio": 0.65, "overcommitted": False},
        "status": "ok",
      },
      {
        "id": "scheduler.workloads",
        "label": "Scheduler workloads",
        "value": "2",
        "value_number": 2,
        "unit": "workloads",
        "detail": "1 waiting · 0 failed",
        "context": {"phase_counts": {"Running": 1, "Pending": 1}},
        "status": "ok",
      },
      {
        "id": "scheduler.seats",
        "label": "Claim ledger seats",
        "value": "1",
        "value_number": 1,
        "unit": "seats",
        "detail": "across 1 ledger · 0 shared",
        "context": {"ledgers": 1, "shared_ledgers": 0},
        "status": "ok",
      },
    ],
    "queues": [
      {"model_id": "demo-run-1", "depth": 5, "oldest_enqueued_at": "2026-07-29T08:01:42+00:00", "oldest_age_seconds": 18},
      {"model_id": "demo-run-2", "depth": 2, "oldest_enqueued_at": "2026-07-29T07:59:15+00:00", "oldest_age_seconds": 75},
    ],
  }


def demo_problems() -> dict:
  cluster = demo_cluster()
  return {
    "demo": True,
    "notice": DEMO_NOTICE,
    "problems": derive_problems(
      demo_health()["checks"],
      {
        "namespace": cluster["kubernetes"]["namespace"],
        "pods": cluster["pods"],
        "nodes": [node for pool in cluster["pools"] for node in pool["nodes"]],
        "scheduler": cluster["scheduler"],
      },
      demo_health()["stats"],
      demo_runs()["runs"],
    ),
  }


def demo_run_detail(run_id: str, log_tail: int = 0) -> dict | None:
  run = next((r for r in demo_runs()["runs"] if r["run_id"] == run_id), None)
  if run is None:
    return None
  cluster = demo_cluster()
  pods = [pod for pod in cluster["pods"] if run_id in pod["name"]]
  gpu_claims = {}
  for pool in cluster["pools"]:
    duty = pool.get("duty")
    if duty and duty["series"] and duty["series"][-1][1].get(run_id):
      gpu_claims[pool["id"]] = duty["series"][-1][1][run_id]
  queue_depth = run["queue_depth"]
  scheduler = cluster["scheduler"]
  workloads = [workload for workload in scheduler["workloads"] if workload["model_id"] == run_id]
  claims = {workload["claim_name"] for workload in workloads if workload["claim_name"]}
  ledgers = [ledger for ledger in scheduler["ledgers"] if ledger["claim_name"] in claims]
  k8s = {
    "available": True,
    "namespace": cluster["kubernetes"]["namespace"],
    "error": None,
    "pods": cluster["pods"],
    "nodes": [],
    "scheduler": scheduler,
  }
  diagnostics = run_diagnostics(run_id, run["state"], pods, queue_depth, k8s, run.get("queue_oldest_seconds"), run.get("telemetry"))
  diagnostics.extend(scheduler_run_diagnostics(workloads, ledgers, k8s))
  detail = {
    **run,
    "demo": True,
    "notice": DEMO_NOTICE,
    "pods": pods,
    "workloads": workloads,
    "claim_ledgers": ledgers,
    "queue_depth": queue_depth,
    "gpu_claims": gpu_claims,
    "gpu_devices": sum(gpu_claims.values()),
    "scheduled_devices": sum(workload["device_count"] for workload in workloads),
    "diagnostics": diagnostics,
  }
  if log_tail:
    detail["logs"] = {pod["name"]: demo_pod_logs(pod["name"])["text"] for pod in pods}
  return detail


def demo_pod_logs(pod: str) -> dict:
  lines = [
    "[demo] fictional log output — this pod does not exist",
    f"[demo] {pod} starting",
    "[demo] loading base model weights (shard 1/4)",
    "[demo] loading base model weights (4/4) done in 41.2s",
    "[demo] worker ready, polling queue open_rl:queue:demo-run-1",
    "[demo] forward_backward batch=32 seq_len=4096 loss=0.8312",
    "[demo] optim_step lr=1e-5 grad_norm=0.42",
    "[demo] forward_backward batch=32 seq_len=4096 loss=0.7981",
  ]
  return {"demo": True, "notice": DEMO_NOTICE, "pod": pod, "container": "demo", "text": "\n".join(lines)}
