import unittest

from server.api import wire
from training import commands


class CommandWireFormatTest(unittest.TestCase):
  def test_commands_round_trip_through_the_wire(self) -> None:
    fb_proto = wire.json_to_fb_proto(
      {"model_id": "m", "forward_backward_input": {"data": [{"model_input": [1, 2], "loss_fn_inputs": {"target_tokens": [2, 3]}}]}}
    )
    for command in (
      commands.CreateModel(request_id="r", model_id="m", base_model="base", fine_tuning_type="full", full_config={"cpu_offload": False}),
      commands.CreateModelFromState(request_id="r", model_id="m", state_path="/ckpt", restore_optimizer=True),
      commands.ForwardBackward(request_id="r", model_id="m", proto_bytes=fb_proto, loss_fn="importance_sampling", forward_only=True),
      commands.OptimStep(request_id="r", model_id="m", adam_params={"learning_rate": 1e-4}),
      commands.Sample(request_id="r", model_id="m", prompt_tokens=[1, 2, 3], max_tokens=4),
      commands.SaveState(request_id="r", model_id="m", state_path="/ckpt", include_optimizer=True, kind="weights"),
      commands.LoadWeights(request_id="r", model_id="m", state_path="/ckpt"),
      commands.SaveWeightsForSampler(request_id="r", model_id="m", alias="final", sampling_session_id="tinker://m/sampler_weights/sampler-0"),
      commands.Shutdown(model_id="m"),
    ):
      with self.subTest(op=command.op):
        raw = commands.wire(command)
        self.assertEqual(raw["op"], command.op)
        self.assertEqual(commands.parse_command(raw), command)

  def test_unknown_op_is_rejected(self) -> None:
    with self.assertRaises(ValueError):
      commands.parse_command({"request_id": "r", "model_id": "m", "op": "frobnicate"})

  def test_legacy_nested_payload_preserves_optimizer_settings(self) -> None:
    raw = {"request_id": "r", "model_id": "m", "op": "optim_step", "payload": {"adam_params": {"learning_rate": 0.02}}}
    command = commands.parse_command(raw)
    self.assertEqual(command.adam_params, {"learning_rate": 0.02})
    self.assertEqual(commands.wire(command)["payload"]["adam_params"], {"learning_rate": 0.02})

  def test_enqueue_stamp_keeps_the_envelope_existing_workers_accept(self) -> None:
    trace_context = {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01", commands.ENQUEUED_AT_KEY: "1700000000.5"}
    raw = commands.wire(commands.OptimStep(request_id="r", model_id="m", trace_context=trace_context))
    self.assertEqual(set(raw), {"op", "request_id", "model_id", "trace_context", "payload"})
    self.assertEqual(commands.parse_command(raw).enqueued_at, 1700000000.5)

  def test_ambiguous_envelope_is_rejected(self) -> None:
    with self.assertRaises(ValueError):
      commands.parse_command({"request_id": "r", "model_id": "m", "op": "optim_step", "payload": {"model_id": "other"}})


if __name__ == "__main__":
  unittest.main()
