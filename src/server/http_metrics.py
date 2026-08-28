"""Bounded, low-cardinality HTTP telemetry for local operations diagnostics."""

import os
import threading
import time
from collections import deque


def percentile(values: list[float], fraction: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  index = max(0, min(len(ordered) - 1, int((len(ordered) * fraction) + 0.999999) - 1))
  return ordered[index]


class HTTPMetrics:
  def __init__(self, max_samples: int = 5000) -> None:
    if max_samples <= 0:
      raise ValueError("max_samples must be positive")
    self._lock = threading.Lock()
    self._samples: deque[dict] = deque(maxlen=max_samples)
    self._in_flight = 0
    self._dropped_samples = 0
    self._last_overflow_at: float | None = None

  def clear(self) -> None:
    with self._lock:
      self._samples.clear()
      self._in_flight = 0
      self._dropped_samples = 0
      self._last_overflow_at = None

  def begin(self) -> float:
    with self._lock:
      self._in_flight += 1
    return time.perf_counter()

  def finish(self, started: float, method: str, route: str, status: int, group: str) -> None:
    self._record(method, route, status, time.perf_counter() - started, group, completed_at=time.time(), finished=True)

  def record(
    self,
    method: str,
    route: str,
    status: int,
    latency_seconds: float,
    group: str = "application",
    *,
    completed_at: float | None = None,
  ) -> None:
    self._record(method, route, status, latency_seconds, group, completed_at=completed_at, finished=False)

  def _record(
    self,
    method: str,
    route: str,
    status: int,
    latency_seconds: float,
    group: str,
    *,
    completed_at: float | None,
    finished: bool,
  ) -> None:
    sample = {
      "at": completed_at if completed_at is not None else time.time(),
      "method": method,
      "route": route,
      "status": int(status),
      "latency_seconds": max(0.0, float(latency_seconds)),
      "group": group,
    }
    with self._lock:
      if len(self._samples) == self._samples.maxlen:
        self._dropped_samples += 1
        self._last_overflow_at = sample["at"]
      self._samples.append(sample)
      if finished:
        self._in_flight = max(0, self._in_flight - 1)

  def snapshot(self, *, now: float | None = None, window_seconds: float | None = None) -> dict:
    now = now if now is not None else time.time()
    if window_seconds is None:
      try:
        window_seconds = max(1.0, float(os.getenv("OPEN_RL_HTTP_WINDOW_SECONDS", "300")))
      except ValueError:
        window_seconds = 300.0
    cutoff = now - window_seconds
    with self._lock:
      while self._samples and self._samples[0]["at"] < cutoff:
        self._samples.popleft()
      samples = [dict(sample) for sample in self._samples]
      in_flight = self._in_flight
      dropped_samples = self._dropped_samples
      window_truncated = self._last_overflow_at is not None and self._last_overflow_at >= cutoff

    groups = {
      name: self._summary([sample for sample in samples if sample["group"] == name], window_seconds)
      for name in ("application", "background", "diagnostic")
    }
    return {
      "window_seconds": window_seconds,
      "in_flight": in_flight,
      "sample_capacity": self._samples.maxlen,
      "sample_count": len(samples),
      "dropped_samples": dropped_samples,
      "window_truncated": window_truncated,
      "groups": groups,
      "routes": self._routes(samples),
      "recent_server_errors": [self._public_sample(sample) for sample in reversed(samples) if sample["status"] >= 500][:10],
    }

  @staticmethod
  def _public_sample(sample: dict) -> dict:
    return {
      "at": sample["at"],
      "method": sample["method"],
      "route": sample["route"],
      "status": sample["status"],
      "latency_seconds": sample["latency_seconds"],
      "group": sample["group"],
    }

  @classmethod
  def _summary(cls, samples: list[dict], window_seconds: float) -> dict:
    latencies = [sample["latency_seconds"] for sample in samples]
    server_errors = sum(sample["status"] >= 500 for sample in samples)
    client_errors = sum(400 <= sample["status"] < 500 for sample in samples)
    count = len(samples)
    return {
      "requests": count,
      "requests_per_second": count / window_seconds,
      "in_window_server_errors": server_errors,
      "in_window_client_errors": client_errors,
      "server_error_rate": server_errors / count if count else 0.0,
      "p50_latency_seconds": cls._rounded(percentile(latencies, 0.50)),
      "p95_latency_seconds": cls._rounded(percentile(latencies, 0.95)),
      "max_latency_seconds": cls._rounded(max(latencies) if latencies else None),
    }

  @classmethod
  def _routes(cls, samples: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for sample in samples:
      grouped.setdefault((sample["group"], sample["method"], sample["route"]), []).append(sample)
    routes = []
    for (group, method, route), observed in grouped.items():
      latencies = [sample["latency_seconds"] for sample in observed]
      routes.append(
        {
          "group": group,
          "method": method,
          "route": route,
          "requests": len(observed),
          "server_errors": sum(sample["status"] >= 500 for sample in observed),
          "client_errors": sum(400 <= sample["status"] < 500 for sample in observed),
          "p95_latency_seconds": cls._rounded(percentile(latencies, 0.95)),
          "max_latency_seconds": cls._rounded(max(latencies)),
          "last_status": observed[-1]["status"],
          "last_seen_at": observed[-1]["at"],
        }
      )
    return sorted(routes, key=lambda item: (-item["server_errors"], -(item["p95_latency_seconds"] or 0), -item["requests"], item["route"]))

  @staticmethod
  def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


http_metrics = HTTPMetrics()
