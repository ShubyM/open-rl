from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from tests._server_fixture import SERVER_DIR


def _load_weight_sync_module():
  spec = importlib.util.spec_from_file_location("weight_sync_under_test", Path(SERVER_DIR) / "weight_sync.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


weight_sync = _load_weight_sync_module()


def _load_state_delta_module():
  spec = importlib.util.spec_from_file_location("state_delta_for_weight_sync_test", Path(SERVER_DIR) / "state_delta.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


state_delta = _load_state_delta_module()


class _ModelStub:
  def state_dict(self):
    return {
      "base_model.model.layers.0.self_attn.q_proj.lora_A.adapter-a.weight": torch.zeros(2, 4),
      "base_model.model.layers.0.self_attn.q_proj.lora_B.adapter-a.weight": torch.ones(4, 2),
      "base_model.model.layers.0.self_attn.q_proj.lora_A.adapter-b.weight": torch.full((2, 4), 2.0),
      "base_model.model.score.modules_to_save.adapter-a.weight": torch.full((4, 4), 4.0),
      "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.full((4, 4), 3.0),
    }


class _PeftSaveStub(_ModelStub):
  def set_adapter(self, adapter_id):
    self.adapter_id = adapter_id

  def save_pretrained(self, state_path, selected_adapters):
    adapter_dir = Path(state_path)
    with (adapter_dir / "adapter_config.json").open("w") as f:
      json.dump({"peft_type": "LORA"}, f)


class _SenderStub:
  def __init__(self):
    self.sent = None

  def update_weights(self, weights):
    self.sent = weights


class _ResponseStub:
  def raise_for_status(self):
    return None

  def json(self):
    return {"type": "sync_lora_tensors"}


class _HttpClientStub:
  posted_url = None
  posted_json = None

  def __init__(self, *args, **kwargs):
    pass

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    return False

  def post(self, url, json):
    self.__class__.posted_url = url
    self.__class__.posted_json = json
    return _ResponseStub()


class TestWeightSync(unittest.TestCase):
  def test_lora_selector_only_returns_requested_adapter_tensors(self) -> None:
    tensors = weight_sync.LoraTensorSelector().select(_ModelStub(), "adapter-a")

    self.assertEqual(
      [tensor.name for tensor in tensors],
      [
        "base_model.model.layers.0.self_attn.q_proj.lora_A.adapter-a.weight",
        "base_model.model.layers.0.self_attn.q_proj.lora_B.adapter-a.weight",
      ],
    )
    self.assertEqual([tensor.role for tensor in tensors], ["lora", "lora"])
    self.assertEqual(tensors[0].shape, (2, 4))
    self.assertEqual(tensors[0].dtype, "float32")

  def test_bridge_publishes_versioned_file_delta(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      bridge = weight_sync.WeightSyncBridge(
        transfer_engine=weight_sync.FileAdapterTransferEngine(tmp_dir),
      )

      state = bridge.publish_for_inference(
        run_id="adapter-a",
        version=None,
        model=_ModelStub(),
        adapter_name="tinker://adapter-a/sampler_weights/current",
        durable_ref="/tmp/open-rl/peft/adapter-a/adapter-a",
        base_model_id="Qwen/Qwen3-0.6B",
      )

      self.assertEqual(state.model_id, "adapter-a")
      self.assertEqual(state.version, 1)
      self.assertEqual(state.base_model, "Qwen/Qwen3-0.6B")
      self.assertEqual(state.state_id, "tinker://adapter-a/sampler_weights/current")
      self.assertEqual(state.adapter_ref, "/tmp/open-rl/peft/adapter-a/adapter-a")
      self.assertEqual(state.tensor_count, 2)
      self.assertEqual(state.transport, "file_adapter_reload")
      self.assertIsNotNone(state.delta_ref)

      manifest_path = Path(state.delta_ref) / "manifest.json"
      with open(manifest_path) as f:
        manifest = json.load(f)
      self.assertEqual(manifest["apply_target"], "vllm_lora")
      self.assertEqual(manifest["run_id"], "adapter-a")
      self.assertTrue(all(entry["checksum"] for entry in manifest["tensors"]))
      self.assertEqual(len(manifest["tensors"]), 2)
      tensor_path = Path(state.delta_ref) / "tensors.safetensors"
      self.assertEqual(
        set(load_file(tensor_path)),
        {
          "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
          "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
        },
      )

  def test_vllm_payload_normalizes_peft_adapter_names(self) -> None:
    self.assertEqual(
      state_delta.normalize_lora_tensor_name("base_model.model.layers.0.self_attn.q_proj.lora_A.adapter-a.weight", "adapter-a"),
      "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
    )

  def test_vllm_payload_can_cast_adapter_tensors(self) -> None:
    tensors = weight_sync.LoraTensorSelector().select(_ModelStub(), "adapter-a")
    manifest = weight_sync.build_vllm_lora_delta_manifest("adapter-a", 1, tensors, None)
    from delta_store import tensor_bytes
    from safetensors.torch import load

    payload = load(tensor_bytes(weight_sync.tensors_for_manifest(tensors, manifest), dtype=torch.float16))

    self.assertEqual(payload["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"].dtype, torch.float16)

  def test_from_env_defaults_to_versioned_vllm_adapter_reload_when_vllm_url_is_set(self) -> None:
    with patch.dict(os.environ, {"VLLM_URL": "http://vllm-service:8001"}, clear=True):
      bridge = weight_sync.WeightSyncBridge.from_env()

    self.assertEqual(bridge.transfer_engine.name, "vllm_lora_adapter_reload")

  def test_from_env_enables_vllm_shared_memory_transport_when_explicitly_set(self) -> None:
    with patch.dict(os.environ, {"VLLM_URL": "http://vllm-service:8001", "OPEN_RL_WEIGHT_SYNC_TRANSPORT": "vllm_lora_tensors_shm"}, clear=True):
      bridge = weight_sync.WeightSyncBridge.from_env()

    self.assertEqual(bridge.transfer_engine.name, "vllm_lora_tensor_shm")

  def test_shared_memory_transfer_posts_control_reference_not_tensor_body(self) -> None:
    tensors = weight_sync.LoraTensorSelector().select(_ModelStub(), "adapter-a")
    with tempfile.TemporaryDirectory() as tmp_dir:
      adapter_dir = Path(tmp_dir)
      with (adapter_dir / "adapter_config.json").open("w") as f:
        json.dump({"peft_type": "LORA"}, f)

      with patch.object(weight_sync.httpx, "Client", _HttpClientStub):
        engine = weight_sync.VLLMLoraTensorSharedMemoryTransferEngine("http://vllm-service:8001")
        state = engine.publish(
          run_id="adapter-a",
          version=7,
          adapter_name="tinker://adapter-a/sampler_weights/current",
          tensors=tensors,
          durable_ref=str(adapter_dir),
          base_model_id="Qwen/Qwen3-0.6B",
        )

    self.assertEqual(state.transport, "vllm_lora_tensor_shm")
    self.assertEqual(_HttpClientStub.posted_json["base_model_id"], "Qwen/Qwen3-0.6B")
    self.assertNotIn("tensors_safetensors_b64", _HttpClientStub.posted_json)
    self.assertIn("tensors_safetensors_shm", _HttpClientStub.posted_json)
    self.assertIn("manifest", _HttpClientStub.posted_json)
    self.assertIn("transport_receipt", _HttpClientStub.posted_json)
    self.assertEqual(_HttpClientStub.posted_json["manifest"]["apply_target"], "vllm_lora")
    self.assertEqual(_HttpClientStub.posted_json["transport_receipt"]["delta_id"], _HttpClientStub.posted_json["manifest"]["delta_id"])
    self.assertGreater(_HttpClientStub.posted_json["tensors_safetensors_shm"]["size"], 0)

  def test_vllm_transfer_routes_sync_to_base_model_worker(self) -> None:
    tensors = weight_sync.LoraTensorSelector().select(_ModelStub(), "adapter-a")
    routes = {
      "Qwen/Qwen3-0.6B": "http://qwen-vllm:8001",
      "google/gemma-3-1b-it": "http://gemma-vllm:8001",
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
      adapter_dir = Path(tmp_dir)
      with (adapter_dir / "adapter_config.json").open("w") as f:
        json.dump({"peft_type": "LORA"}, f)

      with patch.object(weight_sync.httpx, "Client", _HttpClientStub), patch.dict(os.environ, {"OPEN_RL_VLLM_ROUTES": json.dumps(routes)}):
        engine = weight_sync.VLLMAdapterTransferEngine("http://default-vllm:8001")
        engine.publish(
          run_id="adapter-a",
          version=7,
          adapter_name="tinker://adapter-a/sampler_weights/current",
          tensors=tensors,
          durable_ref=str(adapter_dir),
          base_model_id="Qwen/Qwen3-0.6B",
        )

    self.assertEqual(_HttpClientStub.posted_url, "http://qwen-vllm:8001/sync_lora_adapter")
    self.assertEqual(_HttpClientStub.posted_json["base_model_id"], "Qwen/Qwen3-0.6B")

  def test_tensor_dtype_cast_is_reflected_in_manifest(self) -> None:
    tensors = weight_sync.LoraTensorSelector().select(_ModelStub(), "adapter-a")
    with tempfile.TemporaryDirectory() as tmp_dir:
      adapter_dir = Path(tmp_dir)
      with (adapter_dir / "adapter_config.json").open("w") as f:
        json.dump({"peft_type": "LORA"}, f)

      with patch.object(weight_sync.httpx, "Client", _HttpClientStub), patch.dict(os.environ, {"OPEN_RL_WEIGHT_SYNC_CHECKSUM": "1"}):
        engine = weight_sync.VLLMLoraTensorHttpTransferEngine("http://vllm-service:8001", tensor_dtype=torch.float16)
        engine.publish(
          run_id="adapter-a",
          version=8,
          adapter_name="tinker://adapter-a/sampler_weights/current",
          tensors=tensors,
          durable_ref=str(adapter_dir),
          base_model_id="Qwen/Qwen3-0.6B",
        )

    entries = _HttpClientStub.posted_json["manifest"]["tensors"]
    self.assertEqual(_HttpClientStub.posted_json["base_model_id"], "Qwen/Qwen3-0.6B")
    self.assertEqual({entry["dtype"] for entry in entries}, {"float16"})
    self.assertTrue(all(entry["checksum"] for entry in entries))

  def test_hot_sync_fallback_notifies_vllm_adapter_endpoint(self) -> None:
    class _FailThenCaptureClient:
      calls = []

      def __init__(self, *args, **kwargs):
        pass

      def __enter__(self):
        return self

      def __exit__(self, exc_type, exc, tb):
        return False

      def post(self, url, json):
        self.__class__.calls.append((url, json))
        if url.endswith("/sync_lora_tensors"):
          raise RuntimeError("boom")
        return _ResponseStub()

    tensors = weight_sync.LoraTensorSelector().select(_ModelStub(), "adapter-a")
    with tempfile.TemporaryDirectory() as tmp_dir:
      adapter_dir = Path(tmp_dir) / "adapter"
      adapter_dir.mkdir()
      with (adapter_dir / "adapter_config.json").open("w") as f:
        json.dump({"peft_type": "LORA"}, f)

      with patch.object(weight_sync.httpx, "Client", _FailThenCaptureClient):
        engine = weight_sync.VLLMLoraTensorHttpTransferEngine("http://vllm-service:8001")
        state = engine.publish(
          run_id="adapter-a",
          version=9,
          adapter_name="tinker://adapter-a/sampler_weights/current",
          tensors=tensors,
          durable_ref=str(adapter_dir),
          base_model_id="Qwen/Qwen3-0.6B",
        )

    self.assertEqual(state.transport, "file_adapter_reload")
    self.assertEqual(_FailThenCaptureClient.calls[-1][0], "http://vllm-service:8001/sync_lora_adapter")
    self.assertEqual(_FailThenCaptureClient.calls[-1][1]["version"], 9)
    self.assertEqual(_FailThenCaptureClient.calls[-1][1]["adapter_path"], str(adapter_dir))
    self.assertEqual(_FailThenCaptureClient.calls[-1][1]["base_model_id"], "Qwen/Qwen3-0.6B")

  def test_torchrl_transfer_engine_sends_exact_selected_tensors(self) -> None:
    tensors = weight_sync.LoraTensorSelector().select(_ModelStub(), "adapter-a")
    sender = _SenderStub()
    engine = weight_sync.TorchRLVLLMTransferEngine(sender)

    metadata = engine.model_metadata_for(tensors)
    state = engine.publish(
      run_id="adapter-a",
      version=3,
      adapter_name="tinker://adapter-a/sampler_weights/current",
      tensors=tensors,
      durable_ref=None,
    )

    self.assertEqual(list(sender.sent), [tensor.name for tensor in tensors])
    self.assertEqual(metadata[tensors[0].name], (torch.float32, torch.Size([2, 4])))
    self.assertEqual(state.transport, "torchrl_vllm_nccl")
    self.assertEqual(state.runtime_backend, "vllm")
    self.assertEqual(state.tensor_count, 2)

  def test_checkpoint_delta_sidecar_uses_file_delta_store(self) -> None:
    from trainer import TrainerEngine

    engine = TrainerEngine()
    engine.peft_model = _PeftSaveStub()
    with tempfile.TemporaryDirectory() as tmp_dir:
      Path(tmp_dir, "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
      delta_ref = engine.write_checkpoint_delta("adapter-a", tmp_dir)

      self.assertTrue(Path(delta_ref, "manifest.json").exists())
      self.assertTrue(Path(delta_ref, "tensors.safetensors").exists())


if __name__ == "__main__":
  unittest.main()
