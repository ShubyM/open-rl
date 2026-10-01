"""The two places the dashboard reads platform data from, behind one interface.

Both backends answer the same four calls:
  inventory()                       -> nodes, GPUs, pods, scheduler workloads and claims (kubernetes.read()'s shape)
  hardware()                        -> (source with devices()/workers_series(), None) or (None, why there is none)
  pod_logs(pod, container, tail, previous)
  describe()                        -> where each kind of data comes from, for pages and agents

current() picks one per call: OPEN_RL_TELEMETRY_BACKEND=kubernetes|local forces
it, otherwise Cluster whenever Kubernetes credentials load.
"""

import os

from server.telemetry import gke, kubernetes, local
from server.telemetry.prometheus import PromQL


class Cluster:
  name = "kubernetes"

  def inventory(self) -> dict:
    return kubernetes.read()

  def hardware(self) -> tuple[PromQL | None, str | None]:
    config = gke.configuration()
    url = os.getenv("OPEN_RL_PROMETHEUS_URL", "").rstrip("/")
    if url:
      return PromQL("prometheus", url, None, config["cluster"] or None, kubernetes.namespace()), None
    if config["configured"]:
      endpoint = f"https://monitoring.googleapis.com/v1/projects/{config['project']}/location/global/prometheus"
      return PromQL("gke", endpoint, gke.access_token, config["cluster"], kubernetes.namespace()), None
    return None, "No GPU metrics source: not on GKE and OPEN_RL_PROMETHEUS_URL is unset"

  def pod_logs(self, pod: str, container: str | None, tail: int, previous: bool) -> dict:
    return kubernetes.pod_logs(pod, container, tail, previous)

  def describe(self) -> dict:
    source, reason = self.hardware()
    config = gke.configuration()
    return {
      "mode": self.name,
      "inventory": "kubernetes",
      "hardware": source.name if source else None,
      "hardware_error": reason,
      "logs": "cloud_logging" if config["configured"] else "kubelet",
      "gke": config,
    }


class Host:
  name = "local"

  def inventory(self) -> dict:
    return local.read()

  def hardware(self) -> tuple[local.Sampler, None]:
    local.sampler.start()
    return local.sampler, None

  def pod_logs(self, pod: str, container: str | None, tail: int, previous: bool) -> dict:
    return local.pod_logs(pod, tail)

  def describe(self) -> dict:
    return {"mode": self.name, "inventory": "local", "hardware": "local", "hardware_error": None, "logs": "local_files", "gke": gke.configuration()}


CLUSTER, HOST = Cluster(), Host()


def current() -> Cluster | Host:
  forced = os.getenv("OPEN_RL_TELEMETRY_BACKEND", "auto").strip().lower()
  if forced == "local":
    return HOST
  if forced == "kubernetes":
    return CLUSTER
  core, _, _ = kubernetes.clients()
  return CLUSTER if core is not None else HOST
