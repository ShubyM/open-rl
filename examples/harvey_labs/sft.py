#!/usr/bin/env python3
"""SFT a student on collected teacher traces, producing a snapshot that
train.py can warm-start RL from via load_checkpoint_path.

  uv --project examples run python examples/harvey_labs/sft.py \
    model_name=Qwen/Qwen3.5-9B  # traces default to the public HF dataset

Traces are message-level (collect_traces.py); they are re-rendered here with
the student renderer, so training tokens exactly match deployment rendering.
Filter hard: SFT on the teacher's failures teaches failure.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

import chz

RECIPE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RECIPE_DIR))

import tinker
from prompts import lab_renderer
from tinker.lib.public_interfaces import training_client as tinker_training_client
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.supervised.data import conversation_to_datum

tinker_training_client.MAX_CHUNK_BYTES_COUNT = 30_000_000


@chz.chz
class SftConfig:
  # Default pulls the public trace dataset; pass traces=<path>.jsonl[.gz] for
  # a local file or another hf:// URI.
  traces: str = "hf://ShubyM/harvey-lab-glm-traces/jsonl/raw.jsonl.gz"
  base_url: str = "http://127.0.0.1:9003"
  model_name: str = "Qwen/Qwen3.5-9B"
  renderer_name: str | None = "qwen3_5"
  lora_rank: int = 32
  learning_rate: float = 1e-4
  epochs: int = 2
  batch_size: int = 4
  max_length: int = 163840
  min_reward: float = 0.7
  best_per_task: int = 1
  seed: int = 0
  snapshot_name: str = "sft-traces"


def resolve_traces(traces: str) -> Path:
  """Local path, or hf://<user>/<dataset>/<path-in-repo> (e.g. the jsonl/
  mirrors collect_traces.py uploads with hf_repo=...)."""
  if not traces.startswith("hf://"):
    return Path(traces)
  from huggingface_hub import hf_hub_download

  parts = traces.removeprefix("hf://").split("/", 2)
  if len(parts) != 3:
    raise ValueError(f"Expected hf://<user>/<dataset>/<path-in-repo>, got {traces!r}")
  return Path(hf_hub_download("/".join(parts[:2]), parts[2], repo_type="dataset"))


def load_filtered(config: SftConfig) -> list[dict]:
  import gzip

  path = resolve_traces(config.traces)
  opener = gzip.open if path.suffix == ".gz" else open
  with opener(path, "rt", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]
  kept = [r for r in records if r["reward"] >= config.min_reward and r["stop_reason"] == "no_tool_call"]
  by_task: dict[str, list[dict]] = {}
  for r in kept:
    by_task.setdefault(r["task"], []).append(r)
  selected = []
  for task_records in by_task.values():
    task_records.sort(key=lambda r: -r["reward"])
    selected.extend(task_records[: config.best_per_task])
  print(
    f"[sft] {len(records)} traces -> {len(kept)} above reward {config.min_reward} "
    f"-> {len(selected)} after best_per_task={config.best_per_task} ({len(by_task)} tasks)"
  )
  return selected


def to_conversation(record: dict, renderer) -> list[dict]:
  raw = record["messages"]
  if raw[0]["role"] != "system" or raw[1]["role"] != "user":
    raise ValueError(f"Unexpected trace shape for {record['task']} r{record['rollout']}")
  conversation = renderer.create_conversation_prefix_with_tools(
    tools=record["tool_specs"],
    system_prompt=raw[0]["content"],
  ) + [{"role": "user", "content": raw[1]["content"]}]
  call_names: dict[str, str] = {}
  for m in raw[2:]:
    if m["role"] == "assistant":
      message: dict = {"role": "assistant", "content": m.get("content") or ""}
      if m.get("tool_calls"):
        calls = []
        for c in m["tool_calls"]:
          call_names[c["id"]] = c["function"]["name"]
          calls.append(
            ToolCall(
              id=c["id"],
              function=ToolCall.FunctionBody(name=c["function"]["name"], arguments=c["function"]["arguments"]),
            )
          )
        message["tool_calls"] = calls
      conversation.append(message)
    elif m["role"] == "tool":
      conversation.append(
        {
          "role": "tool",
          "content": m["content"],
          "tool_call_id": m["tool_call_id"],
          "name": call_names.get(m["tool_call_id"], ""),
        }
      )
    else:
      raise ValueError(f"Unexpected role {m['role']!r} in trace")
  return conversation


async def main(config: SftConfig) -> None:
  renderer = lab_renderer(config.model_name, config.renderer_name)
  records = load_filtered(config)
  if not records:
    raise ValueError("No traces survived filtering")
  datums = [
    conversation_to_datum(to_conversation(r, renderer), renderer, max_length=config.max_length, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
    for r in records
  ]

  service_client = tinker.ServiceClient(base_url=config.base_url)
  training_client = await service_client.create_lora_training_client_async(base_model=config.model_name, rank=config.lora_rank)
  adam = tinker.AdamParams(learning_rate=config.learning_rate)
  rng = random.Random(config.seed)
  step = 0
  for epoch in range(config.epochs):
    order = list(range(len(datums)))
    rng.shuffle(order)
    for start in range(0, len(order), config.batch_size):
      batch = [datums[i] for i in order[start : start + config.batch_size]]
      fb = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
      opt = await training_client.optim_step_async(adam)
      fb_result = await fb.result_async()
      await opt.result_async()
      step += 1
      print(f"[sft] epoch {epoch} step {step}: {len(batch)} examples metrics={fb_result.metrics}", flush=True)

  fut = await training_client.save_weights_for_sampler_async(name=config.snapshot_name)
  res = await fut.result_async()
  path = res.path
  print(f"\n[sft] done: {step} steps over {len(datums)} examples x {config.epochs} epochs")
  print(f"[sft] snapshot: {path}")
  print(f"[sft] warm-start RL with: load_checkpoint_path={path}")


if __name__ == "__main__":
  asyncio.run(main(chz.entrypoint(SftConfig, allow_hyphens=True)))
