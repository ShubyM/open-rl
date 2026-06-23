import logging
import os
import shlex
import subprocess
import time

from .checkpoint import CheckpointRestorer
from .process_discovery import discover_workload_gpu_pids
from .workload import WorkloadRef

logger = logging.getLogger(__name__)

DEFAULT_GPUCR_PRELOAD = "/usr/local/lib/open-rl/gpu-cr/vGPU-NVIDIA.so"


class GpuCrCheckpointRestorer(CheckpointRestorer):
  """CheckpointRestorer backed by GPU-CR's cr_client binaries."""

  def __init__(
    self,
    cr_client_bin: str | None = None,
    multi_cr_client_bin: str | None = None,
  ):
    self.cr_client_bin = cr_client_bin or os.getenv("GPUCR_CLIENT_BIN", "cr_client")
    self.multi_cr_client_bin = multi_cr_client_bin or os.getenv("GPUCR_MULTI_CLIENT_BIN", "multi_cr_client")
    self.checkpointed_pids: dict[str, list[int]] = {}
    self.initialized_pid_sets: set[tuple[int, ...]] = set()

  def checkpoint(self, workload: WorkloadRef) -> bool:
    pids = self.discover_pids(workload)
    if not pids:
      self.checkpointed_pids.pop(workload.key, None)
      logger.info("gpu-cr checkpoint skipped for workload=%s: no GPU PIDs found", workload.key)
      return False

    start = time.perf_counter()
    logger.info("gpu-cr checkpoint workload=%s pids=%s", workload.key, pids)
    self.run_command(pids, "-c")
    self.checkpointed_pids[workload.key] = pids
    logger.info("gpu-cr checkpoint workload=%s took %.0f ms", workload.key, (time.perf_counter() - start) * 1000)
    return True

  def restore(self, workload: WorkloadRef) -> None:
    pids = self.checkpointed_pids.get(workload.key)
    if not pids:
      raise RuntimeError(f"no checkpointed PIDs found for workload {workload.key}")

    start = time.perf_counter()
    logger.info("gpu-cr restore workload=%s pids=%s", workload.key, pids)
    self.run_command(pids, "-r")
    self.checkpointed_pids.pop(workload.key, None)
    logger.info("gpu-cr restore workload=%s took %.0f ms", workload.key, (time.perf_counter() - start) * 1000)

  def run_command(self, pids: list[int], action: str) -> None:
    if len(pids) == 1:
      self.run_gpucr([self.cr_client_bin, action, "-p", str(pids[0])])
      return

    pid_arg = ",".join(str(pid) for pid in pids)
    pid_set = tuple(pids)
    if action == "-c" and pid_set not in self.initialized_pid_sets:
      self.run_gpucr([self.multi_cr_client_bin, "-i", "-p", pid_arg])
      self.initialized_pid_sets.add(pid_set)
    self.run_gpucr([self.multi_cr_client_bin, action, "-p", pid_arg])

  def run_gpucr(self, argv: list[str]) -> None:
    result = subprocess.run(argv, capture_output=True, check=False, text=True)
    if result.returncode != 0:
      stderr = result.stderr.strip()
      stdout = result.stdout.strip()
      detail = stderr or stdout or f"exit code {result.returncode}"
      rendered_argv = " ".join(shlex.quote(arg) for arg in argv)
      raise RuntimeError(f"{rendered_argv} failed: {detail}")

  def discover_pids(self, workload: WorkloadRef) -> list[int]:
    return discover_workload_gpu_pids(workload)


def gpucr_worker_env(existing_ld_preload: str | None = None) -> dict[str, str]:
  if os.getenv("OPEN_RL_SNAPSHOT_AGENT_BACKEND", "").lower() != "gpucr":
    return {}

  preload = os.getenv("GPUCR_PRELOAD", DEFAULT_GPUCR_PRELOAD)
  ld_preload = existing_ld_preload or ""
  if preload not in ld_preload.split(":"):
    ld_preload = preload if not ld_preload else f"{preload}:{ld_preload}"
  worker_env = {
    "LD_PRELOAD": ld_preload,
    "GPU_VENDOR": os.getenv("GPU_VENDOR", "NVIDIA"),
  }
  if export_file_path := os.getenv("EXPORT_FILE_PATH"):
    worker_env["EXPORT_FILE_PATH"] = export_file_path
  return worker_env
