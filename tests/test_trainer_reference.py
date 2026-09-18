"""Numerical reference tests for the shared training algorithm.

The cross-entropy loss and the AdamW step run on a real tiny Llama on CPU
through the real LoRA and FFT trainers, and the results are checked against
values computed by hand with explicit tolerances. This pins the math the two
backends share, separately from the end-to-end loss-goes-down checks in
test_training_loop.
"""

import torch

from tests.test_training_loop import TinyLlamaCase, training_data
from training import commands
from training.fft_trainer_worker import FFTTrainingWorker
from training.lora_trainer_worker import LoraTrainingWorker
from training.types import Datum


class NumericalReferenceTest(TinyLlamaCase):
  def build_lora_trainer(self):
    worker = LoraTrainingWorker()
    worker.device = torch.device("cpu")
    command = commands.CreateModel(request_id="c", model_id="a", base_model=self.base_model, lora_config={"rank": 4, "seed": 1, "lora_dropout": 0.0})
    return worker.create(command)

  def build_fft_trainer(self):
    worker = FFTTrainingWorker()
    worker.device = torch.device("cpu")
    command = commands.CreateModel(
      request_id="c",
      model_id="m",
      base_model=self.base_model,
      fine_tuning_type="full",
      full_config={"seed": 1, "cpu_offload": False, "weight_sync_strategy": "full"},
    )
    return worker.create(command)

  def assert_first_adam_step_matches_the_formula(self, trainer) -> None:
    """On the first step AdamW's moments start at zero and its bias corrections
    cancel, so the update is exactly p - lr * g / (|g| + eps) per element."""
    data = training_data()
    trainer.forward_backward(data, "cross_entropy")
    before = [p.detach().clone() for p in trainer.params]
    grads = [None if p.grad is None else p.grad.detach().clone() for p in trainer.params]

    lr, eps = 5e-3, 1e-8
    trainer.optim_step({"learning_rate": lr, "beta1": 0.9, "beta2": 0.95, "eps": eps, "weight_decay": 0.0})

    updated = 0
    for param, start, grad in zip(trainer.params, before, grads):
      self.assertIsNotNone(grad, "cross_entropy on weighted data should leave a gradient on every trainable param")
      expected = start - lr * grad / (grad.abs() + eps)
      torch.testing.assert_close(param.detach(), expected, rtol=1e-4, atol=1e-6)
      if float(grad.abs().sum()) > 0:
        updated += 1
    self.assertGreater(updated, 0, "at least one param must have a nonzero gradient")

  def test_lora_first_step_matches_hand_computed_adamw(self) -> None:
    self.assert_first_adam_step_matches_the_formula(self.build_lora_trainer())

  def test_fft_first_step_matches_hand_computed_adamw(self) -> None:
    self.assert_first_adam_step_matches_the_formula(self.build_fft_trainer())

  def test_reported_loss_matches_the_models_own_logprobs(self) -> None:
    """loss:sum is the sum over datums of -logprob(target) * weight, computed
    from the same model the trainer used."""
    trainer = self.build_lora_trainer()
    data = training_data()
    result = trainer.forward_backward(data, "cross_entropy")

    expected_sum = 0.0
    trainer.model.eval()
    with torch.no_grad():
      for item in data:
        input_ids = torch.tensor([item.model_input])
        targets = torch.tensor([item.loss_fn_inputs["target_tokens"].data])
        logits = trainer.model(input_ids).logits[:, : targets.shape[1], :]
        logprobs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        expected_sum += float((-logprobs).sum())

    self.assertAlmostEqual(result["metrics"]["loss:sum"], expected_sum, places=3)
    self.assertAlmostEqual(result["metrics"]["loss:mean"], expected_sum / len(data), places=3)

  def test_zero_weight_data_produces_no_gradient(self) -> None:
    """Every target weight zero means the loss is zero and no gradient flows, so
    the params are unchanged after a step."""
    trainer = self.build_lora_trainer()
    zeroed = [
      Datum(
        model_input=item.model_input,
        loss_fn_inputs={
          "target_tokens": {"data": item.loss_fn_inputs["target_tokens"].data},
          "weights": {"data": [0.0] * len(item.loss_fn_inputs["target_tokens"].data)},
        },
      )
      for item in training_data()
    ]
    result = trainer.forward_backward(zeroed, "cross_entropy")
    self.assertEqual(result["metrics"]["loss:sum"], 0.0)
    for param in trainer.params:
      self.assertTrue(param.grad is None or float(param.grad.abs().sum()) == 0.0)


if __name__ == "__main__":
  import unittest

  unittest.main()
