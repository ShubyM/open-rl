from dataclasses import dataclass

DEFAULT_CLAIM = "shared-accelerator"
TRAINER_CLAIM = "trainers"
SAMPLER_CLAIM = "samplers"


def local_workload_name(role: str, model_id: str) -> str:
  return f"{role}-{model_id}"


# Old names, until the launchers stop using them.
DEFAULT_TIME_SLICE_GROUP = DEFAULT_CLAIM
TRAINER_TIME_SLICE_GROUP = TRAINER_CLAIM
SAMPLER_TIME_SLICE_GROUP = SAMPLER_CLAIM
workload_job_id = local_workload_name


@dataclass(frozen=True)
class WorkloadRef:
  """One process's identity to the time slicer. name is the Workload name
  (the timeslice.io/job-id pod label); claim is the devices it shares."""

  name: str
  claim: str = DEFAULT_CLAIM

  def __post_init__(self) -> None:
    if not self.name:
      raise ValueError("workload requires a name")

  def as_payload(self) -> dict[str, str]:
    return {"name": self.name, "claim": self.claim}
