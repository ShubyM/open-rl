from dataclasses import dataclass

DEFAULT_TIME_SLICE_GROUP = "shared-accelerator"
TRAINER_TIME_SLICE_GROUP = "trainers"
SAMPLER_TIME_SLICE_GROUP = "samplers"


def workload_job_id(role: str, model_id: str) -> str:
  return f"{role}-{model_id}"


@dataclass(frozen=True)
class WorkloadRef:
  """One process's identity to the time slicer.

  The group is the accelerator bundle the workload shares -- under the
  scheduler it is the ResourceClaim name, so turns are taken only between
  workloads on the same physical devices. The owner is the unit of fairness:
  turns rotate between owners, so an owner never gets extra turns for having
  more processes. Exactly one workload per group is resident at a time,
  whatever its owner. A workload that names no owner is an owner of one.
  """

  job_id: str
  group: str = DEFAULT_TIME_SLICE_GROUP
  owner: str = ""

  def __post_init__(self) -> None:
    if not self.job_id:
      raise ValueError("workload requires job_id")

  @property
  def key(self) -> str:
    return f"{self.group}:{self.job_id}"

  @property
  def owner_key(self) -> str:
    return f"{self.group}:{self.owner or self.job_id}"

  def as_payload(self) -> dict[str, str]:
    payload = {"job_id": self.job_id, "group": self.group}
    if self.owner:
      payload["owner"] = self.owner
    return payload
