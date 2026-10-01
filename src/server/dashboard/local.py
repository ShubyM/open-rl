"""The dashboard's sources on a single host without Kubernetes.

Inventory and hardware come from the same poll: `nvidia-smi` for GPUs and the
processes holding them, `psutil` for OpenRL worker processes (found by the
OPEN_RL_PROCESS_ROLE the launcher stamps into their environment). read()
returns the shape cluster.read() does, so runs, placements and history are
joined by the same code on a laptop, a GPU VM, and GKE.
"""

import collections
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import psutil

INTERVAL_SECONDS = 2.0
RETENTION_SECONDS = 3600
GPU_QUERY = ("nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used", "--format=csv,noheader,nounits")
APPS_QUERY = ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits")


def host() -> str:
  return os.getenv("NODE_NAME") or socket.gethostname()


def device_id(index: str) -> str:
  return f"local/{host()}/gpu-{index}"


def csv_rows(command: tuple[str, ...]) -> list[list[str]]:
  out = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True).stdout
  return [[cell.strip() for cell in line.split(",")] for line in out.splitlines() if line.strip()]


def number(text: str) -> float | None:
  try:
    return float(text)
  except ValueError:
    return None  # "[N/A]" on devices that do not report a field


class Sampler:
  """Polls the host every INTERVAL_SECONDS and keeps RETENTION_SECONDS of series."""

  def __init__(self) -> None:
    self.name = "local"
    self.lock = threading.Lock()
    self.thread: threading.Thread | None = None
    self.gpus: list[dict] = []
    self.gpu_error: str | None = None
    self.workers: dict[str, dict] = {}
    self.processes: dict[int, psutil.Process] = {}
    size = int(RETENTION_SECONDS / INTERVAL_SECONDS)
    self.gpu_series: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=size))
    self.worker_series: dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=size))

  def start(self) -> None:
    with self.lock:
      if self.thread is not None:
        return
      self.thread = threading.Thread(target=self.loop, name="openrl-local-sampler", daemon=True)
      self.thread.start()
    self.poll()

  def loop(self) -> None:
    while True:
      time.sleep(INTERVAL_SECONDS)
      try:
        self.poll()
      except Exception:
        pass  # the next poll tries again; a missed one is a gap in the chart

  def poll(self) -> None:
    now = time.time()
    try:
      gpus = [{"index": r[0], "uuid": r[1], "name": r[2], "utilization": number(r[3]), "memory_mib": number(r[4])} for r in csv_rows(GPU_QUERY)]
      apps = csv_rows(APPS_QUERY)
      gpu_error = None
    except (OSError, subprocess.SubprocessError) as exc:
      gpus, apps, gpu_error = [], [], f"nvidia-smi unavailable: {type(exc).__name__}"
    workers = self.scan_workers(now, {int(pid): uuid for uuid, pid in apps if pid.isdigit()})
    with self.lock:
      self.gpus, self.gpu_error, self.workers = gpus, gpu_error, workers
      for gpu in gpus:
        self.gpu_series[gpu["uuid"]].append((now, gpu["utilization"], gpu["memory_mib"]))
      for pod, worker in workers.items():
        self.worker_series[pod].append((now, worker["cpu_cores"], worker["memory_bytes"]))

  def scan_workers(self, now: float, gpu_of_pid: dict[int, str]) -> dict[str, dict]:
    """OpenRL workers on this host, keyed by a pod-like name. A worker's child
    processes (vLLM's engine core) inherit its environment and count as it."""
    groups: dict[str, list[tuple[psutil.Process, dict]]] = {}
    live = set()
    for proc in psutil.process_iter(["pid"]):
      try:
        env = proc.environ()
      except (psutil.Error, OSError):
        continue
      role = env.get("OPEN_RL_PROCESS_ROLE")
      if not role:
        continue
      proc = self.processes.setdefault(proc.pid, proc)  # cpu_percent() measures since the previous call on the same object
      live.add(proc.pid)
      job = env.get("OPEN_RL_WORKLOAD_ID") or env.get("OPEN_RL_TIME_SLICE_JOB_ID") or f"{role}-{env.get('OPEN_RL_RUNTIME_ID', proc.pid)}"
      groups.setdefault(job, []).append((proc, env))
    for pid in set(self.processes) - live:
      del self.processes[pid]
    workers = {}
    for job, members in groups.items():
      try:
        root, env = min(members, key=lambda m: m[0].create_time())
        cpu = sum(p.cpu_percent(None) for p, _ in members) / 100
        rss = sum(p.memory_info().rss for p, _ in members)
        started = root.create_time()
      except psutil.Error:
        continue
      pod = f"{job}-{root.pid}"
      workers[pod] = {
        "pod": pod,
        "pid": root.pid,
        "job": job,
        "uid": f"{job}:{root.pid}:{int(started)}",
        "role": env["OPEN_RL_PROCESS_ROLE"],
        "runtime_id": env.get("OPEN_RL_RUNTIME_ID"),
        "training_kind": "lora" if env.get("OPEN_RL_FINE_TUNING_TYPE") == "lora" else "fft",
        "started_at": started,
        "gpu_uuids": sorted({gpu_of_pid[p.pid] for p, _ in members if p.pid in gpu_of_pid}),
        "cpu_cores": cpu,
        "memory_bytes": rss,
      }
    return workers

  # ---- the Hardware interface (see hardware.py) ----

  async def devices(self, uuids: list[str], start: float, end: float) -> dict[str, dict]:
    with self.lock:
      rows = {uuid: [s for s in self.gpu_series.get(uuid, ()) if start <= s[0] <= end] for uuid in uuids}
    return {
      uuid: {"utilization": [[t, u] for t, u, _ in r if u is not None], "memory_mib": [[t, m] for t, _, m in r if m is not None]}
      for uuid, r in rows.items()
    }

  async def workers_series(self, pods: list[str], start: float, end: float) -> dict[str, dict]:
    with self.lock:
      rows = {pod: [s for s in self.worker_series.get(pod, ()) if start <= s[0] <= end] for pod in pods}
    return {pod: {"cpu_cores": [[t, c] for t, c, _ in r], "memory_bytes": [[t, m] for t, _, m in r]} for pod, r in rows.items()}


sampler = Sampler()


def iso(ts: float) -> str:
  return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(ts))


def read() -> dict:
  """This host in cluster.read()'s shape: one node, its GPUs, and each worker as a placed workload and pod."""
  sampler.start()
  name = host()
  with sampler.lock:
    gpus, gpu_error, workers = list(sampler.gpus), sampler.gpu_error, dict(sampler.workers)
  by_uuid = {gpu["uuid"]: device_id(gpu["index"]) for gpu in gpus}
  devices = [{"id": device_id(gpu["index"]), "name": f"gpu-{gpu['index']}", "uuid": gpu["uuid"]} for gpu in gpus]
  workloads, pods, claims = [], [], {}
  for w in workers.values():
    claims[w["uid"]] = [by_uuid[uuid] for uuid in w["gpu_uuids"] if uuid in by_uuid]
    workloads.append(
      {
        "name": w["job"],
        "uid": w["uid"],
        "created_at": iso(w["started_at"]),
        "role": w["role"],
        "model_id": w["runtime_id"],
        "owner_id": None,
        "training_kind": w["training_kind"],
        "exclusive": False,
        "requested_memory": None,
        "phase": "Running",
        "reason": None,
        "claim_name": w["uid"],
        "pod_name": w["pod"],
        "node_name": name,
        "device_count": len(claims[w["uid"]]),
        "placed_reason": None,
        "placed_message": None,
      }
    )
    pods.append(
      {
        "name": w["pod"],
        "uid": w["uid"],
        "phase": "Running",
        "node": name,
        "worker": w["job"],
        "owner_uids": [w["uid"]],
        "restarts": 0,
        "created_at": iso(w["started_at"]),
        "problem": None,
        "containers": [],
        "events": [],
      }
    )
  node = {"name": name, "ready": True, "accelerator": gpus[0]["name"] if gpus else None, "gpu_capacity": len(gpus), "devices": devices}
  return {
    "available": True,
    "namespace": "local",
    "error": None,
    "pods": pods,
    "nodes": [node],
    "nodes_error": gpu_error,
    "events_error": None,
    "scheduler": {"installed": False, "available": True, "error": None, "workloads": workloads, "ledgers": []},
    "devices": {"available": not gpu_error, "error": gpu_error, "nodes": {name: devices}, "claims": claims},
  }


def pod_logs(pod: str, tail: int) -> dict:
  """The tail of the log file LocalWorkerManager writes for a worker."""
  with sampler.lock:
    worker = sampler.workers.get(pod)
  if worker is None:
    raise LookupError(pod)
  path = Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp")) / f"{worker['role']}_{(worker['runtime_id'] or '').replace('/', '_')}.log"
  with open(path, "rb") as f:
    f.seek(0, os.SEEK_END)
    f.seek(max(0, f.tell() - 128 * 1024))
    lines = f.read().decode(errors="replace").splitlines()[-tail:]
  return {"pod": pod, "container": None, "previous": False, "text": "\n".join(lines)}
