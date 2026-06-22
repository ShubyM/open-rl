import asyncio
import os
from pathlib import Path
from typing import Any, cast

import chz
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import cli_utils
from tinker_cookbook.checkpoint_utils import resolve_renderer_name_from_checkpoint_or_default
from tinker_cookbook.eval import BenchmarkEvaluator
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.train import Config as TrainConfig
from tinker_cookbook.supervised.train import main as train
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

os.environ.setdefault("TINKER_API_KEY", "tml-dummy-key")


@chz.chz
class GSM8KDataset(SupervisedDatasetBuilder):
  model_name: str
  batch_size: int = 16
  max_length: int = 640
  seed: int = 0

  def __call__(self):
    tokenizer = get_tokenizer(self.model_name)
    dataset = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=self.seed)

    def make_datum(row: dict) -> tinker.Datum:
      prompt = tokenizer.encode(f"Question: {row['question']}\nAnswer:", add_special_tokens=False)
      completion = tokenizer.encode(" " + row["answer"].strip(), add_special_tokens=False) + [tokenizer.eos_token_id]
      tokens = (prompt + completion)[: self.max_length]
      weights = ([0] * len(prompt) + [1] * len(completion))[: self.max_length]
      return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs=cast(Any, {"target_tokens": tokens[1:], "weights": [float(w) for w in weights[1:]]}),
      )

    return SupervisedDatasetFromHFDataset(dataset, self.batch_size, map_fn=make_datum), None


@chz.chz
class Config:
  base_model: str = "Qwen/Qwen3-0.6B"
  base_url: str = os.getenv("TINKER_BASE_URL", os.getenv("BASE_URL", "http://127.0.0.1:9003"))
  log_path: str = str(Path(__file__).with_name("artifacts") / "gsm8k_sft")
  epochs: int = 1
  batch: int = 16
  lr: float = 2e-5
  rank: int = 32
  max_len: int = 640
  seed: int = 0
  max_steps: int | None = None
  save_every: int = 0
  behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "delete"
  eval_every: int = 10
  eval_examples: int = 100


def main(config: Config) -> None:
  cli_utils.check_log_dir(config.log_path, behavior_if_exists=config.behavior_if_log_dir_exists)

  renderer_name = resolve_renderer_name_from_checkpoint_or_default(
    model_name=config.base_model,
    explicit_renderer_name=None,
    load_checkpoint_path=None,
    base_url=config.base_url,
  )

  evaluator_builders = []
  if config.eval_every > 0:

    def build_eval():
      tokenizer = get_tokenizer(config.base_model)
      renderer = get_renderer(renderer_name, tokenizer)
      return BenchmarkEvaluator("gsm8k", renderer, max_examples=config.eval_examples)

    evaluator_builders.append(build_eval)

  asyncio.run(
    train(
      TrainConfig(
        log_path=config.log_path,
        model_name=config.base_model,
        dataset_builder=GSM8KDataset(
          model_name=config.base_model,
          batch_size=config.batch,
          max_length=config.max_len,
          seed=config.seed,
        ),
        learning_rate=config.lr,
        lr_schedule="cosine",
        num_epochs=config.epochs,
        lora_rank=config.rank,
        base_url=config.base_url,
        save_every=config.save_every,
        eval_every=config.eval_every,
        infrequent_eval_every=0,
        max_steps=config.max_steps,
        evaluator_builders=evaluator_builders,
      )
    )
  )


if __name__ == "__main__":
  chz.nested_entrypoint(main, allow_hyphens=True)
