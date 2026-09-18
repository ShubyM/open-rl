"""Phase spans and host/CUDA stream timings without extra GPU synchronization."""

import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class TrainingTimings:
  def __init__(self, device: torch.device):
    self.device = device
    self.metrics: dict[str, float] = {}
    self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

  @contextmanager
  def phase(self, name: str) -> Iterator[None]:
    with tracer.start_as_current_span(f"training.{name}"):
      start = time.perf_counter()
      if self.device.type == "cuda":
        stream = torch.cuda.current_stream(self.device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(stream)
      try:
        yield
      finally:
        key = f"time/{name}_host:sum"
        self.metrics[key] = self.metrics.get(key, 0.0) + time.perf_counter() - start
        if self.device.type == "cuda":
          end.record(stream)
          self.events.append((name, begin, end))

  def collect_gpu(self) -> None:
    """Call after the training loop's existing loss.item() synchronization."""
    for name, begin, end in self.events:
      key = f"time/{name}_gpu:sum"
      self.metrics[key] = self.metrics.get(key, 0.0) + begin.elapsed_time(end) / 1000.0
    self.events.clear()
