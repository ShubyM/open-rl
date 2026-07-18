"""One-process smoke test for the real LoRA training path."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import torch

from server.store import get_model_revision
from server.training_requests_processor import InProcessLoraBackend


def make_tiny_model(path: Path) -> str:
  """Write a complete causal LM and tokenizer without a network download."""
  from tokenizers import Tokenizer
  from tokenizers.models import WordLevel
  from tokenizers.pre_tokenizers import Whitespace
  from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

  vocab = {str(token): token for token in range(32)}
  vocab.update({"[UNK]": 32, "[PAD]": 33, "[BOS]": 34, "[EOS]": 35})
  tokenizer_model = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
  tokenizer_model.pre_tokenizer = Whitespace()
  tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_model,
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[BOS]",
    eos_token="[EOS]",
  )
  tokenizer.save_pretrained(path)
  config = LlamaConfig(
    vocab_size=len(vocab),
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=32,
    pad_token_id=33,
    bos_token_id=34,
    eos_token_id=35,
  )
  LlamaForCausalLM(config).save_pretrained(path)
  return str(path)


def test_real_training_step_needs_no_server(tmp_path: Path, monkeypatch) -> None:
  import training.lora_trainer_worker as lora_worker

  monkeypatch.setenv("OPEN_RL_TMP_DIR", str(tmp_path / "runtime"))
  monkeypatch.setenv("OPEN_RL_FUSED_LOGPROB", "0")
  monkeypatch.setattr(lora_worker, "ENABLE_GRADIENT_CHECKPOINTING", False)
  model_path = make_tiny_model(tmp_path / "model")

  async def train_once() -> None:
    backend = InProcessLoraBackend()
    model_id = "tiny-model"
    created = await backend.request(
      "create_model",
      {
        "base_model": model_path,
        "lora_config": {
          "rank": 2,
          "seed": 1,
          "train_attn": True,
          "train_mlp": False,
          "train_unembed": False,
        },
      },
      model_id,
    )
    assert created["type"] == "model_created"

    params = backend.worker.adapter_states[model_id]["trainable_params"]
    before = [param.detach().clone() for param in params]
    forward = await backend.request(
      "forward_backward",
      {
        "data": [
          {
            "model_input": {"chunks": [{"tokens": [1, 2]}]},
            "loss_fn_inputs": {
              "target_tokens": {"data": [2]},
              "weights": {"data": [1.0]},
            },
          }
        ],
        "loss_fn": "cross_entropy",
      },
      model_id,
    )
    assert forward["type"] == "forward_backward_completed"
    assert math.isfinite(forward["metrics"]["loss:mean"])

    optimized = await backend.request(
      "optim_step",
      {
        "adam_params": {
          "learning_rate": 0.1,
          "beta1": 0.0,
          "beta2": 0.0,
          "eps": 1e-8,
        }
      },
      model_id,
    )
    assert optimized["type"] == "optim_step_completed"
    assert optimized["revision"] == 1
    assert await get_model_revision(backend.store, model_id) == 1
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, params, strict=True))

  asyncio.run(train_once())
