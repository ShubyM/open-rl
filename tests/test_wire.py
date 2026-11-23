"""Lean tests for `server.api.wire`, `commands.ForwardBackward`, and binary store framing."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from server.api import wire
from server.proto import tinker_public_pb2 as pb
from server.store import InMemoryStore, decode_wire_dict, encode_wire_dict
from training import commands

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class WireTest(unittest.TestCase):
  def test_sdk_0_29_fixture_parse_cost_datums_and_json_roundtrip(self) -> None:
    raw = (FIXTURES / "fwdbwd_request_tinker_0.29.0.pb").read_bytes()
    expected = json.loads((FIXTURES / "fwdbwd_request_tinker_0.29.0.json").read_text())

    req = wire.parse(raw)
    self.assertEqual((req.model_id, req.seq_id), (expected["model_id"], expected["seq_id"]))
    self.assertEqual(wire.loss_config(req), expected["forward_backward_input"]["loss_fn_config"])
    self.assertEqual(wire.cost(req), sum(len(ch["tokens"]) for d in expected["forward_backward_input"]["data"] for ch in d["model_input"]["chunks"]))

    datums = wire.datums(raw)
    self.assertEqual(len(datums), len(expected["forward_backward_input"]["data"]))
    for datum, exp in zip(datums, expected["forward_backward_input"]["data"]):
      self.assertEqual(datum.tokens.dtype, np.int32)
      self.assertTrue(datum.tokens.flags.writeable)
      self.assertEqual(datum.tokens.tolist(), [t for ch in exp["model_input"]["chunks"] for t in ch["tokens"]])
      for k, exp_t in exp["loss_fn_inputs"].items():
        self.assertTrue(datum.inputs[k].flags.writeable)
        self.assertEqual(datum.inputs[k].tolist(), exp_t["data"])

    # JSON -> protobuf -> datums matches the real SDK binary request
    re_datums = wire.datums(wire.json_to_fb_proto(expected))
    for d1, d2 in zip(datums, re_datums):
      np.testing.assert_array_equal(d1.tokens, d2.tokens)
      self.assertEqual(set(d1.inputs), set(d2.inputs))
      for k in d1.inputs:
        np.testing.assert_array_equal(d1.inputs[k], d2.inputs[k])

  def test_parse_rejects_malformed_and_unsupported_requests(self) -> None:
    with self.assertRaises(wire.ProtoDecodeError):
      wire.parse(b"\xff\xff\xff not proto")

    def _req(setup) -> bytes:
      msg = pb.ForwardBackwardRequest(model_id="m", seq_id=1, loss_fn="cross_entropy")
      d = msg.data.add()
      d.model_input.add().encoded_text.tokens = np.array([1], dtype=np.int32).tobytes()
      setup(d)
      return msg.SerializeToString()

    def _bf16(d):
      t = d.loss_fn_inputs["w"]
      t.dtype, t.dense = pb.DTYPE_BFLOAT16, b"\x00\x00"

    def _sparse(d):
      t = d.loss_fn_inputs["w"]
      t.dtype, t.sparse_csr.values = pb.DTYPE_FLOAT32, np.array([1.0], dtype=np.float32).tobytes()

    def _image(d):
      d.model_input.add().image.data = b"png"

    def _2d(d):
      t = d.loss_fn_inputs["w"]
      t.dtype, t.dense = pb.DTYPE_FLOAT32, np.zeros(4, dtype=np.float32).tobytes()
      t.shape.extend([2, 2])

    for label, fn in (("bf16", _bf16), ("sparse", _sparse), ("image", _image), ("2d", _2d)):
      with self.subTest(label), self.assertRaises(wire.ProtoDecodeError):
        wire.parse(_req(fn))

  def test_output_and_sample_output_binary_and_json_edge_roundtrip(self) -> None:
    metrics = {"loss:mean": 1.5, "loss:sum": 3.0}
    raw_fb = wire.output([np.array([-0.5, -1.5], dtype=np.float32), np.array([-0.25, -0.75, -2.0], dtype=np.float32)], metrics)
    decoded_fb = wire.fb_output_to_json(raw_fb)
    self.assertEqual(decoded_fb["type"], "forward_backward")
    self.assertEqual(decoded_fb["metrics"], metrics)
    self.assertEqual(decoded_fb["loss_fn_outputs"][0]["logprobs"], {"data": [-0.5, -1.5], "dtype": "float32", "shape": [2]})
    self.assertEqual(decoded_fb["loss_fn_outputs"][1]["logprobs"], {"data": [-0.25, -0.75, -2.0], "dtype": "float32", "shape": [3]})

    raw_s = wire.sample_output(
      [
        wire.SampleOut(stop=True, tokens=np.array([9707, 11], dtype=np.int32), logprobs=np.array([-0.5, -1.0], dtype=np.float32)),
        wire.SampleOut(stop=False, tokens=[151645], logprobs=None),
      ],
      prompt_logprobs=[None, -3.0, -0.125],
      prompt_cache_hit_tokens=5,
    )
    decoded_s = wire.sample_response_to_json(raw_s)
    self.assertEqual(decoded_s["type"], "sample")
    self.assertEqual(decoded_s["sequences"][0], {"tokens": [9707, 11], "logprobs": [-0.5, -1.0], "stop_reason": "stop"})
    self.assertEqual(decoded_s["sequences"][1], {"tokens": [151645], "stop_reason": "length"})
    self.assertEqual(decoded_s["prompt_logprobs"], [None, -3.0, -0.125])
    self.assertEqual(decoded_s["prompt_cache_hit_tokens"], 5)

    # Empty NaN prompt_logprob encoded directly in protobuf
    msg = pb.SampleResponse.FromString(raw_s)
    self.assertTrue(math.isnan(np.frombuffer(msg.prompt_logprobs, np.float32)[0]))


class BinaryCommandAndStoreTest(unittest.IsolatedAsyncioTestCase):
  async def test_proto_bytes_passthrough_in_memory_and_utf8_redis_framing(self) -> None:
    raw = (FIXTURES / "fwdbwd_request_tinker_0.29.0.pb").read_bytes()
    cmd = commands.ForwardBackward(request_id="r1", model_id="m1", proto_bytes=raw, loss_fn="importance_sampling")
    wired = commands.wire(cmd)
    self.assertIs(wired["payload"]["proto_bytes"], raw)

    # InMemoryStore: zero-copy bytes passthrough
    mem = InMemoryStore()
    await mem.put_request(wired)
    [popped] = await mem.get_requests()
    self.assertIs(popped["payload"]["proto_bytes"], raw)
    self.assertEqual(commands.parse_command(popped), cmd)

    # UTF-8 Redis wire framing (base64 without mutating caller dict)
    encoded = encode_wire_dict(wired)
    self.assertIsInstance(encoded, str)
    self.assertIs(wired["payload"]["proto_bytes"], raw)
    self.assertEqual(commands.parse_command(decode_wire_dict(encoded)), cmd)

    fb_env = wire.proto_result("forward_backward", wire.output([np.array([-0.5], dtype=np.float32)], {"loss:mean": 0.5}), metrics={"loss:mean": 0.5})
    restored = decode_wire_dict(encode_wire_dict(fb_env))
    self.assertEqual(restored, fb_env)
    self.assertTrue(wire.is_proto_result(restored))


if __name__ == "__main__":
  unittest.main()
