import asyncio
import os
from pathlib import Path

import chz
import tinker_cookbook.utils.logtree as logtree
from tinker_cookbook.recipes.math_rl.train import CLIConfig, cli_main

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


if __name__ == "__main__":
  config = chz.entrypoint(Config)
  with logtree.scope_disable():
    asyncio.run(cli_main(config))
