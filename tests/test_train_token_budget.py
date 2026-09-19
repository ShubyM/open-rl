import os
import unittest
from unittest.mock import patch

import torch

from training.trainer_worker import BaseTrainerWorker, Datum, TensorData, estimate_train_token_budget

GIB = 2**30
QWEN3_8B = dict(vocab_size=151936, hidden_size=4096, num_layers=36, intermediate_size=12288, dtype_bytes=2, gradient_checkpointing=True)
QWEN3_0_6B = dict(vocab_size=151936, hidden_size=1024, num_layers=28, intermediate_size=3072, dtype_bytes=2, gradient_checkpointing=True)


def datum(length: int) -> Datum:
  row = list(range(length + 1))
  return Datum(model_input=row[:-1], loss_fn_inputs={"target_tokens": TensorData(data=row[1:]), "weights": TensorData(data=[1.0] * length)})


class EstimateTest(unittest.TestCase):
  def test_full_fine_tuning_an_8b_on_an_80gb_card_leaves_a_few_thousand_tokens(self) -> None:
    resident = 8_190_000_000 * 2 * (1 + 1 + 2)  # weights, grads, two AdamW moments, bf16
    budget = estimate_train_token_budget(80 * GIB, resident, **QWEN3_8B)
    self.assertTrue(4_000 <= budget <= 16_000, budget)

  def test_lora_on_the_same_card_gets_an_order_of_magnitude_more(self) -> None:
    resident = 8_190_000_000 * 2 + 40_000_000 * 2 * 3
    budget = estimate_train_token_budget(80 * GIB, resident, **QWEN3_8B)
    self.assertTrue(20_000 <= budget <= 60_000, budget)

  def test_a_small_model_on_an_l4_is_bounded_by_its_logits(self) -> None:
    resident = 600_000_000 * 2 * 4
    budget = estimate_train_token_budget(24 * GIB, resident, **QWEN3_0_6B)
    self.assertTrue(8_000 <= budget <= 30_000, budget)

  def test_no_room_means_no_budget(self) -> None:
    self.assertEqual(estimate_train_token_budget(24 * GIB, 30 * GIB, **QWEN3_0_6B), 0)

  def test_without_checkpointing_the_budget_shrinks(self) -> None:
    on = estimate_train_token_budget(80 * GIB, 16 * GIB, **QWEN3_8B)
    off = estimate_train_token_budget(80 * GIB, 16 * GIB, **{**QWEN3_8B, "gradient_checkpointing": False})
    self.assertLess(off, on // 3)


class BatchingTest(unittest.TestCase):
  def test_the_derived_budget_groups_examples_and_the_env_overrides_it(self) -> None:
    worker = BaseTrainerWorker()
    data = [datum(100) for _ in range(6)]
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "0"}):
      self.assertEqual(len(worker.make_training_batches(data)), 6)
      worker.token_budget = 250
      self.assertEqual([len(b) for b in worker.make_training_batches(data)], [2, 2, 2])
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "600"}):
      self.assertEqual([len(b) for b in worker.make_training_batches(data)], [6])


class ModelStub:
  def eval(self):
    return self

  def train(self):
    return self


class OOMFallbackWorker(BaseTrainerWorker):
  def __init__(self):
    super().__init__()
    self.device = torch.device("cpu")
    self.batch_sizes: list[int] = []

  def compute_target_logprobs(self, model, input_ids, attention_mask, target_token_ids):
    self.batch_sizes.append(input_ids.shape[0])
    if input_ids.shape[0] > 2:
      raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    return torch.zeros(target_token_ids.shape, dtype=torch.float32, requires_grad=True)


class OOMBackoffTest(unittest.TestCase):
  def test_an_out_of_memory_batch_halves_the_budget_and_finishes(self) -> None:
    worker = OOMFallbackWorker()
    worker.token_budget = 1000
    data = [datum(100) for _ in range(6)]
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "0"}):
      out = worker.forward_backward(model=ModelStub(), data=data, loss_fn="cross_entropy", forward_only=True)
    self.assertEqual(len(out["loss_fn_outputs"]), 6)
    # 1000 tokens fit all six; the budget halves to 500 (five fit, still too many), then 250.
    self.assertEqual(worker.batch_sizes[0], 6)
    self.assertLessEqual(worker.batch_sizes[-1], 2)
    self.assertEqual(worker.token_budget, 250)


if __name__ == "__main__":
  unittest.main()
