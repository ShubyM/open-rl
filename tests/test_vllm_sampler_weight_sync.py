from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from multiprocessing import shared_memory
from pathlib import Path
from unittest.mock import patch

import torch
from pydantic import ValidationError
from safetensors.torch import load_file, save

from tests._server_fixture import SERVER_DIR


def _load_vllm_sampler_module():
  spec = importlib.util.spec_from_file_location("vllm_sampler_under_test", Path(SERVER_DIR) / "vllm_sampler.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _load_state_delta_module():
  spec = importlib.util.spec_from_file_location("state_delta_for_sampler_test", Path(SERVER_DIR) / "state_delta.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class _RequestStub:
  def __init__(self, data):
    self.data = data

  async def json(self):
    return self.data


class _FakeOutput:
  token_ids = [7]
  logprobs = None
  finish_reason = "length"


class _FakeRequestOutput:
  outputs = [_FakeOutput()]
  prompt_logprobs = None


class _FakeEngine:
  def __init__(self):
    self.lora_request = None

  async def generate(self, prompt, sampling_params, request_id, lora_request):
    self.lora_request = lora_request
    yield _FakeRequestOutput()


vllm_sampler = _load_vllm_sampler_module()


class TestVLLMSamplerWeightSync(unittest.TestCase):
  def test_lora_tensor_sync_request_rejects_missing_required_fields(self) -> None:
    with self.assertRaises(ValidationError):
      vllm_sampler.LoraTensorSyncRequest.model_validate({"adapter_name": "sampler-a"})

  def test_sync_lora_tensors_reads_shared_memory_payload(self) -> None:
    state_delta = _load_state_delta_module()
    tensor_name = "base_model.model.layers.0.self_attn.q_proj.lora_A.adapter-a.weight"
    tensor = torch.zeros(2, 4)
    manifest = state_delta.build_lora_delta_manifest(
      run_id="adapter-a",
      version=2,
      apply_target="vllm_lora",
      adapter_config={"peft_type": "LORA"},
      tensors=[(tensor_name, tensor)],
      created_at=1.0,
    )
    payload_bytes = save({manifest.tensors[0].normalized_name: tensor})
    shm = shared_memory.SharedMemory(create=True, size=len(payload_bytes))
    shm.buf[: len(payload_bytes)] = payload_bytes
    try:
      with tempfile.TemporaryDirectory() as tmp_dir:
        request = _RequestStub(
          {
            "run_id": "adapter-a",
            "version": 2,
            "adapter_name": "sampler-a",
            "adapter_config": {"peft_type": "LORA"},
            "manifest": manifest.to_dict(),
            "transport_receipt": {
              "transport": "vllm_lora_tensor_shm",
              "delta_id": manifest.delta_id,
              "version": manifest.version,
              "locations": {manifest.tensors[0].storage_key: f"shm://{shm.name}"},
              "expires_at": None,
            },
            "tensors_safetensors_shm": {"name": shm.name, "size": len(payload_bytes)},
          }
        )
        with patch.dict(os.environ, {"OPEN_RL_TMP_DIR": tmp_dir}):
          response = asyncio.run(vllm_sampler.sync_lora_tensors(request))

        self.assertEqual(response["type"], "sync_lora_tensors")
        self.assertEqual(response["delta_id"], manifest.delta_id)
        adapter_path = Path(response["adapter_path"])
        self.assertTrue((adapter_path / "adapter_model.safetensors").exists())
        self.assertTrue((adapter_path / "adapter_config.json").exists())
        self.assertEqual(load_file(adapter_path / "adapter_model.safetensors").keys(), {"base_model.model.layers.0.self_attn.q_proj.lora_A.weight"})
        with (adapter_path / "adapter_config.json").open() as f:
          self.assertEqual(json.load(f), {"peft_type": "LORA"})
    finally:
      shm.close()
      shm.unlink()

  def test_generate_uses_synced_lora_adapter_without_request_path(self) -> None:
    fake_engine = _FakeEngine()
    vllm_sampler.engine = fake_engine
    vllm_sampler.synced_lora_adapters["adapter-a"] = {"adapter_path": "/tmp/adapter-a-v2", "version": 2}
    try:
      response = asyncio.run(vllm_sampler.generate(_RequestStub({"request_id": "req-1", "prompt_token_ids": [1], "lora_id": "adapter-a"})))
    finally:
      vllm_sampler.engine = None
      vllm_sampler.synced_lora_adapters.clear()

    self.assertEqual(response["sequences"], [{"tokens": [7], "logprobs": [], "stop_reason": "length"}])
    self.assertEqual(fake_engine.lora_request.lora_name, "adapter-a@2")
    self.assertEqual(fake_engine.lora_request.lora_path, "/tmp/adapter-a-v2")


if __name__ == "__main__":
  unittest.main()
