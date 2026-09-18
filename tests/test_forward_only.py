"""A forward-only pass (TrainingClient.forward) must not accumulate gradients.

The cookbook's NLL evaluator calls forward() on held-out data immediately
before a training step. The trainer only zeroes gradients in optim_step, so a
backward pass here would fold the test set into the next update.
"""

import torch

from tests.test_training_loop import TinyLlamaCase, training_data
from training import commands
from training.lora_trainer_worker import LoraTrainingWorker


class ForwardOnlyTest(TinyLlamaCase):
  def setUp(self) -> None:
    worker = LoraTrainingWorker()
    worker.device = torch.device("cpu")
    # No dropout, so eval and train mode agree on the logprobs.
    command = commands.CreateModel(request_id="c", model_id="a", base_model=self.base_model, lora_config={"rank": 4, "seed": 1, "lora_dropout": 0.0})
    self.trainer = worker.create(command)

  def grads(self) -> list[torch.Tensor]:
    return [p.grad.clone() for p in self.trainer.params if p.grad is not None]

  def test_forward_only_leaves_no_gradients(self) -> None:
    result = self.trainer.forward_backward(training_data(), "cross_entropy", forward_only=True)
    self.assertEqual(self.grads(), [])
    self.assertEqual(len(result["loss_fn_outputs"]), len(training_data()))
    self.assertGreater(result["metrics"]["loss:sum"], 0.0)

  def test_training_pass_still_accumulates(self) -> None:
    self.trainer.forward_backward(training_data(), "cross_entropy")
    self.assertGreater(sum(float(g.norm()) for g in self.grads()), 0.0)

  def test_forward_only_after_training_does_not_change_gradients(self) -> None:
    self.trainer.forward_backward(training_data(), "cross_entropy")
    before = self.grads()
    self.trainer.forward_backward(training_data(), "cross_entropy", forward_only=True)
    after = self.grads()
    self.assertEqual(len(before), len(after))
    for a, b in zip(before, after, strict=True):
      self.assertTrue(torch.equal(a, b))

  def test_forward_only_returns_the_same_logprobs_as_a_training_pass(self) -> None:
    forward_only = self.trainer.forward_backward(training_data(), "cross_entropy", forward_only=True)
    trained = self.trainer.forward_backward(training_data(), "cross_entropy")
    for lhs, rhs in zip(forward_only["loss_fn_outputs"], trained["loss_fn_outputs"], strict=True):
      for a, b in zip(lhs["logprobs"]["data"], rhs["logprobs"]["data"], strict=True):
        self.assertAlmostEqual(a, b, places=5)


if __name__ == "__main__":
  import unittest

  unittest.main()
