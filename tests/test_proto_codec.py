"""Tests for the Tinker protobuf codec.

The request fixture was produced by the real tinker 0.29.0 SDK
(``forward_backward_request_to_proto``), so the decoder is checked against
bytes a client actually sends rather than against our own encoder. Response
encoders are checked structurally here and, when the tinker SDK is importable,
round-tripped through the SDK's own deserializers.
"""

from __future__ import annotations

import array
import json
import math
import unittest
from pathlib import Path

from server import proto_codec
from server.proto import tinker_public_pb2 as pb

FIXTURES = Path(__file__).resolve().parent / "fixtures"

try:  # The SDK is optional in the unit-test environment.
  from tinker.proto import response_conv as sdk_response_conv
except Exception:  # noqa: BLE001
  sdk_response_conv = None


class DecodeRequestTest(unittest.TestCase):
  def test_sdk_0_29_request_decodes_to_the_json_request_shape(self) -> None:
    body = (FIXTURES / "fwdbwd_request_tinker_0.29.0.pb").read_bytes()
    expected = json.loads((FIXTURES / "fwdbwd_request_tinker_0.29.0.json").read_text())
    self.assertEqual(proto_codec.decode_forward_backward(body), expected)

  def test_integer_tensors_stay_ints_and_floats_stay_floats(self) -> None:
    body = (FIXTURES / "fwdbwd_request_tinker_0.29.0.pb").read_bytes()
    datum = proto_codec.decode_forward_backward(body)["forward_backward_input"]["data"][1]
    self.assertTrue(all(isinstance(t, int) for t in datum["model_input"]["chunks"][0]["tokens"]))
    self.assertTrue(all(isinstance(t, int) for t in datum["loss_fn_inputs"]["target_tokens"]["data"]))
    self.assertTrue(all(isinstance(w, float) for w in datum["loss_fn_inputs"]["weights"]["data"]))
    self.assertEqual(datum["loss_fn_inputs"]["logprobs"]["data"][1], -1.25)

  def test_forward_only_and_legacy_float_config_are_carried(self) -> None:
    msg = pb.ForwardBackwardRequest(model_id="m", seq_id=3, loss_fn="cross_entropy", forward_only=True)
    msg.loss_fn_config["clip_range"] = 0.5
    decoded = proto_codec.decode_forward_backward(msg.SerializeToString())
    self.assertTrue(decoded["forward_only"])
    self.assertEqual(decoded["forward_backward_input"]["loss_fn_config"], {"clip_range": 0.5})
    self.assertEqual(decoded["forward_backward_input"]["data"], [])

  def test_int32_tensors_widen_to_int64(self) -> None:
    msg = pb.ForwardBackwardRequest(model_id="m", seq_id=1, loss_fn="cross_entropy")
    datum = msg.data.add()
    datum.model_input.add().encoded_text.tokens = array.array("i", [1, 2, 3]).tobytes()
    tensor = datum.loss_fn_inputs["target_tokens"]
    tensor.dtype = pb.DTYPE_INT32
    tensor.dense = array.array("i", [2, 3, 4]).tobytes()
    decoded = proto_codec.decode_forward_backward(msg.SerializeToString())
    self.assertEqual(
      decoded["forward_backward_input"]["data"][0]["loss_fn_inputs"]["target_tokens"], {"data": [2, 3, 4], "dtype": "int64", "shape": [3]}
    )

  def test_rejects_what_the_workers_cannot_run(self) -> None:
    def request_with(tensor_setup=None, chunk_setup=None) -> bytes:
      msg = pb.ForwardBackwardRequest(model_id="m", seq_id=1, loss_fn="cross_entropy")
      datum = msg.data.add()
      chunk = datum.model_input.add()
      if chunk_setup:
        chunk_setup(chunk)
      else:
        chunk.encoded_text.tokens = array.array("i", [1]).tobytes()
      tensor = datum.loss_fn_inputs["weights"]
      if tensor_setup:
        tensor_setup(tensor)
      else:
        tensor.dtype = pb.DTYPE_FLOAT32
        tensor.dense = array.array("f", [1.0]).tobytes()
      return msg.SerializeToString()

    def bf16(tensor):
      tensor.dtype = pb.DTYPE_BFLOAT16
      tensor.dense = b"\x00\x00"

    def sparse(tensor):
      tensor.dtype = pb.DTYPE_FLOAT32
      tensor.sparse_csr.values = array.array("f", [1.0]).tobytes()

    def image(chunk):
      chunk.image.data = b"png"

    for label, body in (
      ("bfloat16", request_with(tensor_setup=bf16)),
      ("sparse", request_with(tensor_setup=sparse)),
      ("image", request_with(chunk_setup=image)),
    ):
      with self.subTest(label), self.assertRaises(proto_codec.ProtoDecodeError):
        proto_codec.decode_forward_backward(body)

  def test_garbage_is_a_decode_error_not_a_crash(self) -> None:
    with self.assertRaises(proto_codec.ProtoDecodeError):
      proto_codec.decode_forward_backward(b"\xff\xff\xff not a protobuf")


SAMPLE_RESULT = {
  "type": "sample_completed",
  "sequences": [
    {"tokens": [9707, 11, 1879], "logprobs": [-0.5, -1.0, -0.25], "stop_reason": "stop"},
    {"tokens": [151645], "logprobs": [-2.0], "stop_reason": "length"},
    {"tokens": [1, 2], "stop_reason": "abort"},
  ],
  "prompt_logprobs": [None, -3.0, -0.125],
}

FWDBWD_RESULT = {
  "type": "forward_backward_completed",
  "loss_fn_output_type": "ArrayRecord",
  "metrics": {"loss:mean": 1.5, "loss:sum": 3.0},
  "loss_fn_outputs": [
    {"logprobs": {"data": [-0.5, -1.5], "dtype": "float32", "shape": [2]}},
    {"logprobs": {"data": [-0.25, -0.75, -2.0], "dtype": "float32", "shape": [3]}},
  ],
}


def _floats(raw: bytes) -> list[float]:
  arr = array.array("f")
  arr.frombytes(raw)
  return arr.tolist()


def _ints(code: str, raw: bytes) -> list[int]:
  arr = array.array(code)
  arr.frombytes(raw)
  return arr.tolist()


class EncodeSampleResponseTest(unittest.TestCase):
  def test_sequences_tokens_logprobs_and_stop_reasons(self) -> None:
    msg = pb.SampleResponse()
    msg.ParseFromString(proto_codec.encode_sample_response(SAMPLE_RESULT))
    self.assertEqual(len(msg.sequences), 3)
    self.assertEqual(_ints("i", msg.sequences[0].tokens), [9707, 11, 1879])
    self.assertEqual(_floats(msg.sequences[0].logprobs), [-0.5, -1.0, -0.25])
    self.assertEqual(msg.sequences[0].stop_reason, pb.STOP_REASON_STOP)
    self.assertEqual(msg.sequences[1].stop_reason, pb.STOP_REASON_LENGTH)
    # Unknown reasons collapse to LENGTH: the SDK raises on any other enum value.
    self.assertEqual(msg.sequences[2].stop_reason, pb.STOP_REASON_LENGTH)
    self.assertEqual(msg.sequences[2].logprobs, b"")

  def test_prompt_logprobs_keep_their_slots_with_nan_for_none(self) -> None:
    msg = pb.SampleResponse()
    msg.ParseFromString(proto_codec.encode_sample_response(SAMPLE_RESULT))
    values = _floats(msg.prompt_logprobs)
    self.assertEqual(len(values), 3)
    self.assertTrue(math.isnan(values[0]))
    self.assertEqual(values[1:], [-3.0, -0.125])

  @unittest.skipUnless(sdk_response_conv, "tinker SDK not installed")
  def test_sdk_deserializes_what_we_encode(self) -> None:
    response = sdk_response_conv.deserialize_sample_response(proto_codec.encode_sample_response(SAMPLE_RESULT))
    self.assertEqual([s.tokens for s in response.sequences], [[9707, 11, 1879], [151645], [1, 2]])
    self.assertEqual(response.sequences[0].logprobs, [-0.5, -1.0, -0.25])
    self.assertEqual([s.stop_reason for s in response.sequences], ["stop", "length", "length"])
    self.assertIsNone(response.sequences[2].logprobs)


class EncodeForwardBackwardOutputTest(unittest.TestCase):
  def test_batched_tensor_layout(self) -> None:
    msg = pb.ForwardBackwardOutput()
    msg.ParseFromString(proto_codec.encode_forward_backward_output(FWDBWD_RESULT))
    self.assertEqual(msg.loss_fn_output_type, "ArrayRecord")
    self.assertEqual(dict(msg.metrics), {"loss:mean": 1.5, "loss:sum": 3.0})
    self.assertEqual(len(msg.loss_fn_outputs), 1)
    record = msg.loss_fn_outputs[0]
    self.assertEqual(record.num_datums, 2)
    field = record.fields["logprobs"]
    self.assertEqual(field.dtype, pb.DTYPE_FLOAT32)
    self.assertEqual(list(field.trailing_shape), [])
    self.assertEqual(_floats(field.data), [-0.5, -1.5, -0.25, -0.75, -2.0])
    # Byte offsets, num_datums + 1 entries: datum 0 is bytes [0, 8), datum 1 is [8, 20).
    self.assertEqual(_ints("q", field.offsets), [0, 8, 20])

  def test_empty_outputs_still_carry_metrics(self) -> None:
    msg = pb.ForwardBackwardOutput()
    msg.ParseFromString(
      proto_codec.encode_forward_backward_output({"type": "forward_backward_completed", "metrics": {"loss:mean": 0.0}, "loss_fn_outputs": []})
    )
    self.assertEqual(len(msg.loss_fn_outputs), 0)
    self.assertEqual(dict(msg.metrics), {"loss:mean": 0.0})

  @unittest.skipUnless(sdk_response_conv, "tinker SDK not installed")
  def test_sdk_deserializes_what_we_encode(self) -> None:
    output = sdk_response_conv.deserialize_forward_backward_output(proto_codec.encode_forward_backward_output(FWDBWD_RESULT))
    self.assertEqual(output.loss_fn_output_type, "ArrayRecord")
    self.assertEqual(output.metrics, {"loss:mean": 1.5, "loss:sum": 3.0})
    self.assertEqual(len(output.loss_fn_outputs), 2)
    self.assertEqual(list(output.loss_fn_outputs[0]["logprobs"].data), [-0.5, -1.5])
    self.assertEqual(list(output.loss_fn_outputs[1]["logprobs"].data), [-0.25, -0.75, -2.0])
    self.assertEqual(output.loss_fn_outputs[1]["logprobs"].shape, [3])


class EncodeFutureResultTest(unittest.TestCase):
  def test_only_proto_capable_types_are_encoded(self) -> None:
    self.assertIsNotNone(proto_codec.encode_future_result(SAMPLE_RESULT))
    self.assertIsNotNone(proto_codec.encode_future_result(FWDBWD_RESULT))
    self.assertIsNone(proto_codec.encode_future_result({"type": "optim_step_completed", "metrics": {}}))
    self.assertIsNone(proto_codec.encode_future_result({"status": "pending"}))


if __name__ == "__main__":
  unittest.main()
