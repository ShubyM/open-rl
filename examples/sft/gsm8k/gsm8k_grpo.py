import asyncio
import os
from pathlib import Path

import chz
import tinker_cookbook.utils.logtree as logtree
from tinker_cookbook.recipes.math_rl.train import CLIConfig
from tinker_cookbook.recipes.math_rl.train import get_dataset_builder
from tinker_cookbook.rl.train import Config as RLTrainConfig
from tinker_cookbook.rl.train import main as rl_train_main
from tinker_utils import LimitedDatasetBuilder

# Set default API key and backend URL defaults
os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")


@chz.chz
class Config(CLIConfig):
  # Default settings for the GSM8K time-slicing RL run
  env: str = "gsm8k"
  base_url: str = os.getenv("TINKER_BASE_URL", os.getenv("BASE_URL", "http://127.0.0.1:8000"))
  model_name: str = "Qwen/Qwen3-0.6B"
  lora_rank: int = 16
  max_tokens: int = 256
  temperature: float = 0.8
  learning_rate: float = 1e-5
  group_size: int = 4
  groups_per_batch: int = 2
  log_path: str = str(Path(__file__).resolve().parent / "artifacts" / "gsm8k_grpo")
  behavior_if_log_dir_exists: str = "delete"
  loss_fn: str = "importance_sampling"
  max_steps: int = 20
  eval_every: int = 10


async def cli_main(cli_config: Config):
  renderer_name = "qwen3"
  
  base_builder = get_dataset_builder(
      env=cli_config.env,
      batch_size=cli_config.groups_per_batch,
      model_name=cli_config.model_name,
      renderer_name=renderer_name,
      group_size=cli_config.group_size,
      seed=cli_config.seed,
  )
  
  dataset_builder = LimitedDatasetBuilder(
      base_builder,
      max_batches=None,
      max_eval_batches=1, # Limiting evaluation to 1 batch!
  )
  
  config = RLTrainConfig(
      learning_rate=cli_config.learning_rate,
      dataset_builder=dataset_builder,
      model_name=cli_config.model_name,
      renderer_name=renderer_name,
      lora_rank=cli_config.lora_rank,
      max_tokens=cli_config.max_tokens,
      temperature=cli_config.temperature,
      wandb_project=cli_config.wandb_project,
      wandb_name=cli_config.wandb_name or f"gsm8k-grpo",
      log_path=cli_config.log_path,
      base_url=cli_config.base_url,
      eval_every=cli_config.eval_every,
      save_every=cli_config.save_every,
      max_steps=cli_config.max_steps,
  )
  
  await rl_train_main(config)


if __name__ == "__main__":
  config = chz.entrypoint(Config)
  with logtree.scope_disable():
    asyncio.run(cli_main(config))
