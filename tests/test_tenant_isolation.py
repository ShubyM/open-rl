"""Tenancy: several models on one worker stay isolated, and the create/delete
rules are enforced by the request processor.

Real tiny Llama on CPU, the real in-memory store and the real LoRA and FFT
workers. Each test drives typed commands through the processor the way the
API server enqueues them.
"""

import torch

from tests.test_training_loop import TinyLlamaCase, training_data
from training import commands


class TenantIsolationTest(TinyLlamaCase):
  async def test_accumulated_gradients_survive_a_tenant_switch(self) -> None:
    """Adapter a's pending gradients are still there after a forward_backward on
    adapter b, so a can step on them. Activating b freezes a's params, it never
    zeroes a's grads."""
    processor = self.lora_processor()
    await self.run_commands(processor, self.create_lora("a"), self.create_lora("b"))
    data = training_data()

    await self.run_commands(processor, commands.ForwardBackward(request_id="fb-a", model_id="a", data=data))
    a_params = processor.trainers["a"].params
    self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in a_params))
    grads_before = [None if p.grad is None else p.grad.detach().clone() for p in a_params]

    await self.run_commands(processor, commands.ForwardBackward(request_id="fb-b", model_id="b", data=data))
    for param, saved in zip(a_params, grads_before):
      if saved is None:
        self.assertIsNone(param.grad)
      else:
        self.assertTrue(torch.equal(param.grad, saved))

    stepped = await self.run_commands(processor, commands.OptimStep(request_id="step-a", model_id="a", adam_params={"learning_rate": 5e-2}))
    self.assertIn("grad_norm:mean", stepped["step-a"]["metrics"])

  async def test_duplicate_model_id_is_rejected_and_leaves_the_first_intact(self) -> None:
    processor = self.lora_processor()
    first = await self.run_commands(processor, self.create_lora("a"))
    self.assertEqual(first["create-a"]["type"], "model_created")

    dup = commands.CreateModel(request_id="dup", model_id="a", base_model=self.base_model, lora_config={"rank": 4, "seed": 1})
    result = await self.run_commands(processor, dup)
    self.assertEqual(result["dup"]["type"], "RequestFailedResponse")
    self.assertIn("already exists", result["dup"]["error_message"])
    self.assertEqual(list(processor.trainers), ["a"])

  async def test_lora_worker_rejects_a_full_create(self) -> None:
    processor = self.lora_processor()
    conflict = commands.CreateModel(
      request_id="c", model_id="x", base_model=self.base_model, fine_tuning_type="full", full_config={"seed": 1, "cpu_offload": False}
    )
    result = await self.run_commands(processor, conflict)
    self.assertEqual(result["c"]["type"], "RequestFailedResponse")
    self.assertIn("LoRA", result["c"]["error_message"])
    self.assertEqual(processor.trainers, {})

  async def test_fft_worker_rejects_a_lora_create(self) -> None:
    processor = self.fft_processor()
    result = await self.run_commands(processor, self.create_lora("x"))
    self.assertEqual(result["create-x"]["type"], "RequestFailedResponse")
    self.assertIn("full-parameter", result["create-x"]["error_message"])
    self.assertEqual(processor.trainers, {})

  async def test_single_model_worker_refuses_a_second_model(self) -> None:
    processor = self.fft_processor()
    created = await self.run_commands(processor, self.create_full("m"))
    self.assertEqual(created["create-m"]["type"], "model_created")

    result = await self.run_commands(processor, self.create_full("n"))
    self.assertEqual(result["create-n"]["type"], "RequestFailedResponse")
    self.assertIn("single-model", result["create-n"]["error_message"])
    self.assertEqual(list(processor.trainers), ["m"])

  async def test_deleting_an_adapter_frees_it_and_later_commands_fail(self) -> None:
    processor = self.lora_processor()
    await self.run_commands(processor, self.create_lora("a"))
    await self.run_commands(processor, commands.ForwardBackward(request_id="fb", model_id="a", data=training_data()))

    deleted = await self.run_commands(processor, commands.DeleteModel(request_id="del", model_id="a"))
    self.assertEqual(deleted["del"], {"status": "ok", "type": "model_deleted"})
    self.assertNotIn("a", processor.trainers)
    self.assertNotIn("a", processor.worker.peft_model.peft_config)

    after = await self.run_commands(processor, commands.ForwardBackward(request_id="fb2", model_id="a", data=training_data()))
    self.assertEqual(after["fb2"]["type"], "RequestFailedResponse")
    self.assertIn("a", after["fb2"]["error_message"])


if __name__ == "__main__":
  import unittest

  unittest.main()
