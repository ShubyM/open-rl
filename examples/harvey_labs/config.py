"""One configuration shared by training, evaluation, and LAB environments."""

from pathlib import Path

import chz
from tinker_cookbook import model_info


@chz.chz
class RunConfig:
  model_name: str = "Qwen/Qwen3.5-9B"
  renderer_name: str | None = None
  base_url: str | None = None
  lab_root: Path = chz.field(default=Path(__file__).resolve().parent / "harvey-labs", munger=lambda _, path: Path(path).expanduser().resolve())
  log_path: str = "artifacts/harvey-labs"

  task: str | None = None  # A single training task; otherwise use the seeded split.
  train_tasks: int = 300
  eval_tasks: int = 50
  task_split_seed: int = 0
  batch_size: int = 1
  rollouts_per_example: int = 4
  eval_rollouts_per_task: int = 4

  max_steps: int = 40
  max_turns: int = 40
  max_tokens: int = 3072
  max_trajectory_tokens: int = 128 * 1024
  max_tool_result_tokens: int = 8 * 1024
  command_timeout: int = 60
  judge_model: str = "gemini-3.5-flash"
  judge_parallel: int = 0  # Auto: 16 for GLM, 1 otherwise.

  learning_rate: float = 3e-6
  lora_rank: int = 32
  save_every: int = 5
  eval_every: int = 20
  final_eval: bool = True
  stream_minibatches: bool = False
  num_substeps: int = 1
  kl_penalty_coef: float = 0.0
  kl_discount_factor: float = 0.0
  # Warm-start weights with a fresh optimizer and batch counter; not a resume.
  load_checkpoint_path: str | None = None

  @property
  def resolved_renderer(self) -> str:
    if self.renderer_name:
      return self.renderer_name
    if "gemma" in self.model_name.lower():
      return "gemma4"
    return model_info.get_recommended_renderer_name(self.model_name)
