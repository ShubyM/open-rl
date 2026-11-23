"""Real zero-mock E2E test: create_app + TrainingRequestsProcessor + 2-layer CPU tiny-Llama (FFT & LoRA)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from server.api import wire
from server.api_server import create_app
from server.proto import tinker_public_pb2 as pb
from server.store import InMemoryStateStore, InMemoryStore
from server.training_requests_processor import TrainingRequestsProcessor
from training.fft_trainer_worker import FFTTrainingWorker
from training.lora_trainer_worker import LoraTrainingWorker


def _save_tiny_llama(directory: str) -> None:
  torch.manual_seed(0)
  tok = Tokenizer(models.WordLevel({str(i): i for i in range(64)}, unk_token="0"))
  tok.pre_tokenizer = pre_tokenizers.Whitespace()
  PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="0", pad_token="0").save_pretrained(directory)
  cfg = LlamaConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=64,
  )
  LlamaForCausalLM(cfg).save_pretrained(directory)


def _fb_json(model_id: str, *, forward_only: bool = False) -> dict:
  rows = [[3, 5, 8, 13, 21, 34], [2, 7, 1, 8]]
  data = [
    {
      "model_input": {"chunks": [{"tokens": row[:-1]}]},
      "loss_fn_inputs": {
        "target_tokens": {"data": row[1:], "dtype": "int64", "shape": [len(row) - 1]},
        "weights": {"data": [1.0] * (len(row) - 1), "dtype": "float32", "shape": [len(row) - 1]},
      },
    }
    for row in rows
  ]
  return {
    "model_id": model_id,
    "seq_id": 1,
    "forward_only": forward_only,
    "forward_backward_input": {"data": data, "loss_fn": "cross_entropy", "loss_fn_config": {}},
  }


class ApiServerEndToEndTest(unittest.IsolatedAsyncioTestCase):
  async def _run_full_lifecycle(self, *, fine_tuning_type: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      model_dir = os.path.join(tmp, "tiny-llama")
      _save_tiny_llama(model_dir)
      store, state = InMemoryStore(), InMemoryStateStore()
      worker = LoraTrainingWorker() if fine_tuning_type == "lora" else FFTTrainingWorker()
      worker.device = torch.device("cpu")
      if isinstance(worker, FFTTrainingWorker):
        worker.set_weight_sync_strategy("full")

      env = {"OPEN_RL_TMP_DIR": tmp, "OPEN_RL_ENABLE_FFT": "true", "SAMPLING_BACKEND": "torch"}
      with patch.dict(os.environ, env):
        app = create_app(store=store, state=state, tmp_dir=tmp)
        proc = TrainingRequestsProcessor(store, worker)
        loop_task = asyncio.create_task(proc.run())
        try:
          async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:

            async def retrieve(req_id: str, *, accept: str = wire.JSON) -> httpx.Response:
              for _ in range(200):
                resp = await client.post("/api/v1/retrieve_future", json={"request_id": req_id}, headers={"Accept": accept})
                if resp.status_code == 200:
                  return resp
                self.assertEqual(resp.status_code, 408, resp.text)
                await asyncio.sleep(0.01)
              self.fail(f"Timed out waiting for future {req_id}")

            # 1. create_session -> create_model
            sess = (await client.post("/api/v1/create_session", json={})).json()["session_id"]
            user_meta = {"openrl.fine_tuning_type": fine_tuning_type}
            if fine_tuning_type == "full":
              user_meta["openrl.weight_sync"] = "full"
              user_meta["openrl.fft_cpu_offload"] = "false"
            cm = await client.post(
              "/api/v1/create_model",
              json={"base_model": model_dir, "session_id": sess, "lora_config": {"rank": 2, "seed": 7}, "user_metadata": user_meta},
            )
            self.assertEqual(cm.status_code, 200, cm.text)
            model_id = cm.json()["request_id"]
            created = (await retrieve(model_id)).json()
            self.assertEqual(created["type"], "create_model")
            self.assertTrue(created["is_lora"])

            model_obj = worker.peft_model if isinstance(worker, LoraTrainingWorker) else worker.model
            before_weights = [p.detach().clone() for p in model_obj.parameters() if p.requires_grad]

            # 2. protobuf forward_backward -> retrieve_future (protobuf)
            fb_payload = _fb_json(model_id)
            fb_proto = wire.json_to_fb_proto(fb_payload)
            fb_pb_resp = await client.post("/api/v1/forward_backward", content=fb_proto, headers={"Content-Type": wire.PROTO})
            self.assertEqual(fb_pb_resp.status_code, 200, fb_pb_resp.text)
            fb_pb_out = await retrieve(fb_pb_resp.json()["request_id"], accept=wire.PROTO)
            self.assertEqual(fb_pb_out.headers["content-type"], wire.PROTO)
            parsed_fb = pb.ForwardBackwardOutput()
            parsed_fb.ParseFromString(fb_pb_out.content)
            self.assertEqual(parsed_fb.loss_fn_outputs[0].num_datums, 2)

            # 3. JSON forward_backward -> retrieve_future (JSON)
            fb_json_resp = await client.post("/api/v1/forward_backward", json=fb_payload)
            self.assertEqual(fb_json_resp.status_code, 200, fb_json_resp.text)
            fb_json_out = (await retrieve(fb_json_resp.json()["request_id"], accept=wire.JSON)).json()
            self.assertEqual(fb_json_out["type"], "forward_backward")
            self.assertEqual(len(fb_json_out["loss_fn_outputs"]), 2)
            self.assertAlmostEqual(fb_json_out["metrics"]["loss:sum"], parsed_fb.metrics["loss:sum"], places=4)

            # 4. /forward (forward_only=True) must not change accumulated gradients
            grads_before = [p.grad.detach().clone() for p in model_obj.parameters() if p.grad is not None]
            fwd_proto = wire.json_to_fb_proto(fb_payload, forward_only=True)
            fwd_resp = await client.post("/api/v1/forward", content=fwd_proto, headers={"Content-Type": wire.PROTO})
            self.assertEqual(fwd_resp.status_code, 200, fwd_resp.text)
            fwd_out = await retrieve(fwd_resp.json()["request_id"], accept=wire.PROTO)
            self.assertEqual(fwd_out.headers["content-type"], wire.PROTO)
            grads_after = [p.grad.detach().clone() for p in model_obj.parameters() if p.grad is not None]
            for a, b in zip(grads_before, grads_after, strict=True):
              torch.testing.assert_close(a, b)

            # 5. optim_step mutates trainable weights
            opt_body = {"model_id": model_id, "adam_params": {"learning_rate": 1e-2}}
            opt_req = (await client.post("/api/v1/optim_step", json=opt_body)).json()["request_id"]
            opt_out = (await retrieve(opt_req)).json()
            self.assertEqual(opt_out["type"], "optim_step")
            after_weights = [p.detach().clone() for p in model_obj.parameters() if p.requires_grad]
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_weights, after_weights, strict=True)))

            # 6. save_weights -> load_weights
            sv_req = (await client.post("/api/v1/save_weights", json={"model_id": model_id, "path": "step-1"})).json()["request_id"]
            sv_out = (await retrieve(sv_req)).json()
            self.assertEqual(sv_out["type"], "save_weights")
            self.assertTrue(sv_out["path"].startswith(f"tinker://{model_id}/weights/"), sv_out)

            ld_body = {"model_id": model_id, "path": sv_out["path"], "optimizer": True}
            ld_req = (await client.post("/api/v1/load_weights", json=ld_body)).json()["request_id"]
            ld_out = (await retrieve(ld_req)).json()
            self.assertEqual(ld_out["type"], "load_weights")

            # 7. asample -> retrieve_future (both protobuf and JSON)
            samp_body = {
              "model_id": model_id,
              "prompt": {"chunks": [{"tokens": [3, 5, 8]}]},
              "sampling_params": {"max_tokens": 2, "temperature": 0.0},
              "num_samples": 1,
              "prompt_logprobs": True,
            }
            s1 = (await client.post("/api/v1/asample", json=samp_body)).json()["request_id"]
            s1_pb = await retrieve(s1, accept=wire.PROTO)
            self.assertEqual(s1_pb.headers["content-type"], wire.PROTO)
            parsed_s = pb.SampleResponse()
            parsed_s.ParseFromString(s1_pb.content)
            self.assertEqual(len(parsed_s.sequences), 1)

            s2 = (await client.post("/api/v1/asample", json=samp_body)).json()["request_id"]
            s2_json = (await retrieve(s2, accept=wire.JSON)).json()
            self.assertEqual(s2_json["type"], "sample")
            self.assertEqual(len(s2_json["sequences"][0]["tokens"]), 2)
            self.assertIsNone(s2_json["prompt_logprobs"][0])
        finally:
          loop_task.cancel()
          await asyncio.gather(loop_task, return_exceptions=True)

  async def test_lora_end_to_end_http_and_protobuf(self) -> None:
    await self._run_full_lifecycle(fine_tuning_type="lora")

  async def test_fft_end_to_end_http_and_protobuf(self) -> None:
    await self._run_full_lifecycle(fine_tuning_type="full")

  def test_api_server_and_dispatcher_import_without_torch(self) -> None:
    code = """
import sys
sys.modules["torch"] = None
import server.api.wire
import server.api_server
import server.sampling_dispatcher
res = server.sampling_dispatcher.to_result(
  {"choices": [{"token_ids": [7, 8], "finish_reason": "stop", "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -0.2}]}}]},
  {"prompt_token_ids": [1], "include_prompt_logprobs": False},
)
assert server.api.wire.is_proto_result(res)
assert server.api.wire.sample_response_to_json(res["proto_bytes"])["sequences"][0]["tokens"] == [7, 8]
"""
    proc = subprocess.run([sys.executable, "-c", code], env=dict(os.environ), capture_output=True, text=True)
    self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
  unittest.main()
