"""The CPU CI sampler consumes publications, independently of the trainer."""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from tests.cpu_sampler import CpuSampler
from training.lora_trainer_worker import LoraTrainingWorker
from training.types import LoraConfig


class CpuSamplerTest(unittest.TestCase):
  def test_sampling_reads_exported_adapter_and_changes_only_after_publication(self):
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": directory}):
      torch.manual_seed(1)
      model = LlamaForCausalLM(
        LlamaConfig(
          vocab_size=16,
          hidden_size=8,
          intermediate_size=16,
          num_hidden_layers=1,
          num_attention_heads=2,
          num_key_value_heads=2,
          max_position_embeddings=32,
          pad_token_id=0,
          eos_token_id=15,
        )
      )
      tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<pad>": 0, **{str(i): i for i in range(1, 16)}}, unk_token="<pad>")),
        pad_token="<pad>",
        eos_token="15",
      )
      base_path = os.path.join(directory, "base")
      model.save_pretrained(base_path)
      tokenizer.save_pretrained(base_path)
      trainer = LoraTrainingWorker(base_model=model, tokenizer=tokenizer, device="cpu", base_model_name=base_path)
      trainer.create_adapter("adapter", LoraConfig(rank=2, lora_alpha=2, lora_dropout=0.0, train_unembed=True))
      trainer.save_for_sampler("adapter", "first", None)
      sampler = CpuSampler(base_path)
      request = {
        "prompt_token_ids": [1, 2, 3],
        "max_tokens": 2,
        "num_samples": 1,
        "temperature": 0.0,
        "lora_id": "adapter",
        "lora_path": os.path.join(directory, "peft", "adapter", "adapter"),
        "include_prompt_logprobs": True,
      }
      first = sampler.sample(request)
      self.assertEqual(first["type"], "sample")
      self.assertTrue(first["sequences"][0]["tokens"])
      self.assertEqual(len(first["sequences"][0]["tokens"]), len(first["sequences"][0]["logprobs"]))

      with torch.no_grad():
        for param in trainer.adapters["adapter"].params:
          param.add_(torch.randn_like(param) * 0.4)
      self.assertEqual(sampler.sample(request), first)

      trainer.save_for_sampler("adapter", "second", None)
      published = sampler.sample(request)
      self.assertNotEqual(published["prompt_logprobs"], first["prompt_logprobs"])
      trainer.peft_model.eval()
      with torch.no_grad():
        input_ids = torch.tensor([request["prompt_token_ids"]])
        scores = trainer.peft_model(input_ids).logits[0, :-1].log_softmax(dim=-1)
        expected = scores.gather(-1, input_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
      torch.testing.assert_close(torch.tensor(published["prompt_logprobs"][1:]), expected)


if __name__ == "__main__":
  unittest.main()
