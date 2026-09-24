"""CPU-only API fixture for SDK and pig-latin tests.

The trainer and sampler use separate models and the production request queues.
Sampling reads the exported adapter, so an optimizer step cannot change the
sampler until the client explicitly publishes weights. This is deliberately
test support, not another production training or sampling backend.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from server import api_server
from server.store import RequestStore
from server.training_requests_processor import call_backend


class CpuSampler:
  def __init__(self, base_model: str):
    self.base_model = base_model
    self.tokenizer = AutoTokenizer.from_pretrained(base_model)
    self.model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32, device_map="cpu")

  @torch.no_grad()
  def sample(self, request: dict[str, Any]) -> dict[str, Any]:
    if request.get("weights_path") or request.get("stop"):
      raise NotImplementedError("The CPU test fixture supports base/LoRA sampling without custom stop sequences")
    adapter_path = request.get("lora_path")
    if adapter_path:
      # LoRA exports currently share a mutable directory. Reload the small
      # adapter for each request, keeping only one independent base model.
      if isinstance(self.model, PeftModel):
        self.model.delete_adapter("sample")
        self.model.load_adapter(adapter_path, adapter_name="sample")
      else:
        self.model = PeftModel.from_pretrained(self.model, adapter_path, adapter_name="sample")
      self.model.set_adapter("sample")
    else:
      if request.get("lora_id") not in (None, self.base_model):
        raise ValueError("Publish adapter weights before sampling")
      if isinstance(self.model, PeftModel):
        self.model = self.model.unload()
    self.model.eval()

    tokens = request["prompt_token_ids"]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    prompt_scores = None
    if request.get("include_prompt_logprobs"):
      logits = self.model(input_ids, attention_mask=torch.ones_like(input_ids)).logits[0, :-1]
      scores = logits.log_softmax(dim=-1).gather(-1, input_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
      prompt_scores = [None, *scores.tolist()]

    temperature = request.get("temperature", 1.0)
    do_sample = temperature > 0
    input_ids = input_ids.repeat(request.get("num_samples", 1), 1)
    output = self.model.generate(
      input_ids,
      attention_mask=torch.ones_like(input_ids),
      max_new_tokens=request["max_tokens"],
      pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
      do_sample=do_sample,
      temperature=temperature if do_sample else None,
      top_p=request.get("top_p", 1.0) if do_sample else None,
      top_k=max(0, request.get("top_k", -1)) if do_sample else None,
      output_scores=True,
      return_dict_in_generate=True,
    )
    eos = self.model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    sequences = []
    for index, sequence in enumerate(output.sequences):
      generated = sequence[len(tokens) :].tolist()
      logprobs = []
      for step, token in enumerate(generated):
        logprobs.append(output.scores[step][index].log_softmax(dim=-1)[token].item())
        if token in eos_ids:
          generated = generated[: step + 1]
          break
      sequences.append({"tokens": generated, "logprobs": logprobs, "stop_reason": "stop" if generated[-1] in eos_ids else "length"})
    result = {"type": "sample", "sequences": sequences}
    if prompt_scores is not None:
      result["prompt_logprobs"] = prompt_scores
    return result


async def run_sampler(sampler: CpuSampler, store: RequestStore) -> None:
  while True:
    requests = await store.get_sampling_requests_for_model(sampler.base_model)
    for request in requests:
      try:
        result = await call_backend(sampler.sample, request)
      except Exception as exc:
        result = {"type": "RequestFailedResponse", "error_message": str(exc)}
      await store.set_future(request["request_id"], result)
    if not requests:
      await asyncio.sleep(0.01)


@asynccontextmanager
async def lifespan(app):
  async with api_server.lifespan(app):
    sampler = await call_backend(CpuSampler, os.environ["BASE_MODEL"])
    task = asyncio.create_task(run_sampler(sampler, api_server.store))
    try:
      yield
    finally:
      task.cancel()
      await asyncio.gather(task, return_exceptions=True)


def create_app():
  app = api_server.app
  app.router.lifespan_context = lifespan
  return app
