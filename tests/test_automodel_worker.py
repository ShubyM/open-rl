"""CPU regressions for the Automodel worker's training contract."""

import contextlib
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from training import automodel_worker as automodel
from training.trainer_worker import Datum, TensorData


class TinyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.embedding = torch.nn.Embedding(11, 4)
    self.model = torch.nn.Module()
    self.model.norm = torch.nn.LayerNorm(4)
    self.lm_head = torch.nn.Linear(4, 11)
    self.config = types.SimpleNamespace(final_logit_softcapping=2.0)
    self.config.get_text_config = lambda: self.config
    self.calls = []

  def get_output_embeddings(self):
    return self.lm_head

  def forward(self, input_ids, **kwargs):
    self.calls.append(kwargs)
    self.model.norm(self.embedding(input_ids))
    return types.SimpleNamespace(hidden_states=None)


class AutomodelWorkerTest(unittest.TestCase):
  def setUp(self):
    self.enterContext(patch.object(torch.cuda, "is_available", return_value=False))
    self.enterContext(patch.object(automodel, "AUTOMODEL_TP", 1))
    self.enterContext(patch.object(automodel, "AUTOMODEL_CP", 1))
    self.enterContext(patch.object(automodel, "AUTOMODEL_LORA_RANK", 8))
    self.worker = automodel.AutomodelTrainingWorker()

  def attach_model(self):
    model = TinyModel()
    self.worker.model = model
    self.worker.tokenizer = types.SimpleNamespace(pad_token_id=0)
    self.worker.prepare_model_for_training()
    return model

  def test_invalid_parallel_and_chunk_settings_fail_before_loading(self):
    for name, value in (
      ("AUTOMODEL_TP", 0),
      ("AUTOMODEL_CP", -1),
      ("AUTOMODEL_LORA_RANK", -1),
      ("RECOMPUTE_NUM_LAYERS", -1),
      ("LOGPROB_CHUNK", 0),
    ):
      with self.subTest(name=name), patch.object(automodel, name, value), self.assertRaises(ValueError):
        automodel.AutomodelTrainingWorker()
    with (
      patch.object(automodel, "AUTOMODEL_TP", 2),
      patch.object(automodel, "AUTOMODEL_LORA_RANK", 0),
      self.assertRaisesRegex(ValueError, "trainable output head"),
    ):
      automodel.AutomodelTrainingWorker()

  def test_create_model_resets_weights_optimizer_and_seeds_initialization(self):
    def load_model(_name):
      self.worker.model = torch.nn.Linear(4, 4)

    with patch.object(self.worker, "load_base_model", side_effect=load_model):
      self.worker.create_model("base", config=automodel.AutomodelConfig(seed=7))
      old_model = self.worker.model
      initial = old_model.weight.detach().clone()
      self.worker.optimizer = torch.optim.AdamW(old_model.parameters())
      self.worker.checkpointer = object()
      self.worker.create_model("base", config=automodel.AutomodelConfig(seed=7))
      torch.testing.assert_close(self.worker.model.weight, initial)
      self.assertIsNot(self.worker.model, old_model)
      self.assertIsNone(self.worker.optimizer)
      self.assertIsNone(self.worker.checkpointer)
      self.assertIs(self.worker.trainable_params[0], self.worker.model.weight)
      self.worker.create_model("base", config=automodel.AutomodelConfig(seed=8))
      self.assertFalse(torch.equal(self.worker.model.weight, initial))

  def test_forward_trims_mask_and_captures_only_current_final_states(self):
    model = self.attach_model()
    inputs = torch.tensor([[1, 2, 3, 0], [3, 4, 0, 0]])
    targets = torch.tensor([[2, 3, 4], [4, 5, 0]])
    mask = (inputs != 0).long()
    actual = self.worker.compute_target_logprobs(model, inputs, mask, targets)
    torch.testing.assert_close(model.calls[-1]["attention_mask"], mask[:, :3])
    self.assertNotIn("output_hidden_states", model.calls[-1])
    self.assertEqual(len(model.model.norm._forward_hooks), 0)
    with torch.no_grad():
      hidden = model.model.norm(model.embedding(inputs[:, :3]))
      logits = 2 * torch.tanh(model.lm_head(hidden) / 2)
      expected = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    with (
      patch.object(model, "forward", return_value=types.SimpleNamespace(hidden_states=(torch.randn(2, 3, 4),))),
      self.assertRaisesRegex(RuntimeError, "did not run the final norm"),
    ):
      self.worker.compute_target_logprobs(model, inputs, mask, targets)
    self.assertEqual(len(model.model.norm._forward_hooks), 0)

  def test_chunk_projection_matches_dense_values_and_gradients_with_softcap(self):
    model = TinyModel().double()
    for extra_hidden_rows in (0, 2):
      with self.subTest(extra_hidden_rows=extra_hidden_rows):
        hidden = torch.randn(2, 5 + extra_hidden_rows, 4, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(0, 11, (2, 5))
        logits = 2 * torch.tanh(model.lm_head(hidden[:, : targets.shape[1]].double()) / 2)
        expected = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        expected_grads = torch.autograd.grad(expected.sum(), [hidden, *model.lm_head.parameters()])
        with patch.object(automodel, "LOGPROB_CHUNK", 3):
          actual = self.worker.project_target_logprobs(model, hidden, targets)
        actual_grads = torch.autograd.grad(actual.sum(), [hidden, *model.lm_head.parameters()])
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
          torch.testing.assert_close(actual_grad, expected_grad)

  def test_cp_uses_upstream_layout_and_keeps_context_through_backward(self):
    model = self.attach_model()
    self.worker.cp_size = 2
    active = False
    events = []

    @contextlib.contextmanager
    def attention_context():
      nonlocal active
      self.assertFalse(active)
      active = True
      events.append("enter")
      try:
        yield
      finally:
        active = False
        events.append("exit")

    def check_backward(grad):
      self.assertTrue(active)
      events.append("backward")
      return grad

    model.embedding.weight.register_hook(check_backward)

    def shard(batch):
      return attention_context, {
        "input_ids": batch["input_ids"][:, :2],
        "labels": batch["labels"][:, :2],
        "position_ids": torch.tensor([[0, 1]]),
        "model_specific_metadata": "preserved",
      }

    sharder = Mock()
    sharder.shard.side_effect = shard
    sharder.gather_token_tensor.side_effect = lambda local, trim: local.repeat(1, 2)
    factory = Mock(return_value=sharder)
    datum = Datum(model_input=[1, 2, 1, 2], loss_fn_inputs={"target_tokens": TensorData(data=[2, 3, 2, 3])})
    module = "nemo_automodel.components.distributed.context_parallel.sharder"
    with patch.dict(sys.modules, {module: types.SimpleNamespace(ContextParallelSharder=factory)}):
      result = self.worker.forward_backward([datum, datum], "cross_entropy")
    self.assertEqual(len(result["loss_fn_outputs"]), 2)
    self.assertEqual(factory.call_count, 2)
    self.assertEqual(events, ["enter", "backward", "exit", "enter", "backward", "exit"])
    self.assertTrue(sharder.gather_token_tensor.call_args.kwargs["trim"])
    self.assertEqual(model.calls[-1]["model_specific_metadata"], "preserved")
    self.assertIsNone(self.worker.cp_context)
    self.assertEqual(len(model.model.norm._forward_hooks), 0)

  def test_forward_error_closes_context_and_releases_hidden_states(self):
    model = self.attach_model()
    closed = Mock()
    stack = contextlib.ExitStack()
    stack.callback(closed)
    self.worker.cp_context = stack
    datum = Datum(model_input=[1], loss_fn_inputs={"target_tokens": TensorData(data=[2])})
    with patch.object(model, "forward", side_effect=RuntimeError("forward failed")), self.assertRaisesRegex(RuntimeError, "forward failed"):
      self.worker.forward_backward([datum], "cross_entropy")
    closed.assert_called_once()
    self.assertIsNone(self.worker.cp_context)
    self.assertEqual(len(model.model.norm._forward_hooks), 0)


if __name__ == "__main__":
  unittest.main()
