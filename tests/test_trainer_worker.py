"""The trainer worker's math, checked against per-example references, torch's AdamW and PEFT's own loader.

Everything runs on a tiny Llama on the CPU. The invariants are the ones that
matter for training correctness: a padded, batched step equals the
per-example math; optim_step is torch.optim.AdamW to the bit; tenants on one
base never see each other; a saved adapter is exactly what PEFT loads back.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from peft import PeftModel
from transformers import LlamaConfig, LlamaForCausalLM

from training import losses
from training.trainer_worker import TrainerWorker, token_logprobs
from training.types import Datum, LoraConfig, TensorData

ADAM = {"learning_rate": 1e-2, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.01, "grad_clip_norm": 0.5}


def lora(rank: int, seed: int, **flags) -> LoraConfig:
  """A LoRA config without dropout, so two forwards with different RNG states agree."""
  return LoraConfig(rank=rank, seed=seed, lora_dropout=0.0, **flags)


def tiny_llama(seed: int = 0, tie_word_embeddings: bool = False, checkpointing: bool = False) -> LlamaForCausalLM:
  torch.manual_seed(seed)
  config = LlamaConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=64,
    tie_word_embeddings=tie_word_embeddings,
  )
  model = LlamaForCausalLM(config)
  if checkpointing:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
  return model


def worker_on(module: torch.nn.Module, name: str = "tiny") -> TrainerWorker:
  worker = TrainerWorker()
  worker.base, worker.base_name, worker.device = module, name, torch.device("cpu")
  return worker


def datum(model_input, target_tokens, weights=None, logprobs=None, advantages=None) -> Datum:
  inputs = {"target_tokens": TensorData(data=target_tokens)}
  if weights is not None:
    inputs["weights"] = TensorData(data=weights)
  if logprobs is not None:
    inputs["logprobs"] = TensorData(data=logprobs)
  if advantages is not None:
    inputs["advantages"] = TensorData(data=advantages)
  return Datum(model_input=model_input, loss_fn_inputs=inputs)


def datums() -> list[Datum]:
  return [
    datum([3, 4, 5, 6], [1, 2, 3, 4], weights=[1.0, 0.5, 0.25, 2.0], logprobs=[-0.1, -0.2, -0.3, -0.4], advantages=[1.0, -0.5, 2.0, 0.25]),
    datum([7, 8], [2, 3], logprobs=[-0.7, -0.8], advantages=[0.75, 1.25]),
    datum([9, 10, 11], [5, 6, 7, 8], weights=[0.2, 0.4, 0.6, 0.8], logprobs=[-0.9, -1.0, -1.1, -1.2], advantages=[-1.0, 0.3, 0.9, 1.7]),
  ]


def per_example_backward(module: torch.nn.Module, data: list[Datum]) -> None:
  """The reference: each datum alone, no padding, cross entropy summed into .grad."""
  for item in data:
    length = min(len(item.model_input), len(item.loss_fn_inputs["target_tokens"].data))
    input_ids = torch.tensor([item.model_input])
    targets = torch.tensor([item.loss_fn_inputs["target_tokens"].data[:length]])
    weights = torch.tensor([item.loss_fn_inputs["weights"].data[:length]]) if "weights" in item.loss_fn_inputs else torch.ones(1, length)
    logprobs = token_logprobs(module, input_ids, torch.ones_like(input_ids), targets)
    (-(logprobs * weights)).sum().backward()


class StepMathTest(unittest.TestCase):
  def test_a_padded_batched_step_matches_the_per_example_loop_with_torch_adamw(self) -> None:
    worker = worker_on(tiny_llama())
    worker.create_model("tiny", "full")
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "16"}):
      self.assertLess(len(worker.make_training_batches(datums())), len(datums()))
      worker.forward_backward("full", datums(), "cross_entropy")
    worker.optim_step("full", ADAM)

    reference = tiny_llama()
    optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, foreach=False)
    per_example_backward(reference, datums())
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
    optimizer.step()

    for (name, param), (_, expected) in zip(worker.base.named_parameters(), reference.named_parameters(), strict=True):
      torch.testing.assert_close(param, expected, rtol=1e-4, atol=1e-6, msg=name)

  def test_optim_step_is_torch_adamw_to_the_bit(self) -> None:
    worker = worker_on(tiny_llama())
    worker.create_model("tiny", "job-a", LoraConfig(rank=4, seed=1))
    params = worker.models["job-a"].params
    reference = [torch.nn.Parameter(param.detach().clone()) for param in params.values()]
    optimizer = torch.optim.AdamW(reference, lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, foreach=False)
    generator = torch.Generator().manual_seed(3)

    for _ in range(3):
      for param, expected in zip(params.values(), reference, strict=True):
        grad = torch.randn(param.shape, generator=generator)
        param.grad, expected.grad = grad.clone(), grad.clone()
      torch.nn.utils.clip_grad_norm_(reference, 0.5)
      optimizer.step()
      optimizer.zero_grad()
      worker.optim_step("job-a", ADAM)

      model = worker.models["job-a"]
      for (name, param), expected in zip(params.items(), reference, strict=True):
        self.assertTrue(torch.equal(param, expected), name)
        self.assertTrue(torch.equal(model.adam_m[name], optimizer.state[expected]["exp_avg"]), name)
        self.assertTrue(torch.equal(model.adam_v[name], optimizer.state[expected]["exp_avg_sq"]), name)
      self.assertTrue(all(param.grad is None for param in params.values()))

  def test_grad_norm_is_reported_before_clipping(self) -> None:
    worker = worker_on(tiny_llama())
    worker.create_model("tiny", "job-a", LoraConfig(rank=2, seed=1))
    for param in worker.models["job-a"].params.values():
      param.grad = torch.ones_like(param)
    expected = torch.sqrt(sum(param.numel() for param in worker.models["job-a"].params.values()) * torch.tensor(1.0))
    result = worker.optim_step("job-a", {**ADAM, "grad_clip_norm": 0.1})
    self.assertAlmostEqual(result["metrics"]["grad_norm:mean"], float(expected), places=4)


class TenancyTest(unittest.TestCase):
  def test_tenants_on_one_base_never_see_each_other(self) -> None:
    base = tiny_llama()
    worker = worker_on(base)
    worker.create_model("tiny", "job-a", lora(4, 1))
    worker.create_model("tiny", "job-b", lora(2, 2, train_mlp=False))
    base_before = [param.detach().clone() for param in base.parameters()]

    worker.forward_backward("job-a", datums()[:2], "cross_entropy")
    self.assertTrue(all(param.grad is None for param in worker.models["job-b"].params.values()))
    worker.optim_step("job-a", ADAM)
    worker.forward_backward("job-b", datums()[1:], "cross_entropy")
    worker.optim_step("job-b", ADAM)

    for model_id, config, data in (("job-a", lora(4, 1), datums()[:2]), ("job-b", lora(2, 2, train_mlp=False), datums()[1:])):
      alone = worker_on(tiny_llama())
      alone.create_model("tiny", model_id, config)
      alone.forward_backward(model_id, data, "cross_entropy")
      alone.optim_step(model_id, ADAM)
      shared, solo = worker.models[model_id].params, alone.models[model_id].params
      self.assertEqual(list(shared), list(solo))
      for name in shared:
        self.assertTrue(torch.equal(shared[name], solo[name]), f"{model_id} {name}")

    for before, after in zip(base_before, base.parameters(), strict=True):
      self.assertTrue(torch.equal(before, after))
      self.assertFalse(after.requires_grad)

  def test_lora_targets_follow_the_config(self) -> None:
    worker = worker_on(tiny_llama())
    worker.create_model("tiny", "attn", LoraConfig(rank=2, train_attn=True, train_mlp=False))
    suffixes = {name.split(".lora_")[0].rsplit(".", 1)[-1] for name in worker.models["attn"].params}
    self.assertEqual(suffixes, {"q_proj", "k_proj", "v_proj", "o_proj"})

    tied = worker_on(tiny_llama(tie_word_embeddings=True))
    tied.create_model("tiny", "unembed", LoraConfig(rank=2, train_attn=False, train_mlp=True, train_unembed=True))
    self.assertFalse(any("lm_head" in name for name in tied.models["unembed"].params))

  def test_gradient_checkpointing_does_not_change_lora_gradients(self) -> None:
    grads = []
    for checkpointing in (False, True):
      worker = worker_on(tiny_llama(checkpointing=checkpointing))
      worker.create_model("tiny", "job-a", lora(4, 1))
      worker.forward_backward("job-a", datums(), "cross_entropy")
      grads.append({name: param.grad.clone() for name, param in worker.models["job-a"].params.items()})
    # B starts at zero, so only the B gradients are nonzero on the first pass.
    self.assertTrue(all(grad.abs().sum() > 0 for name, grad in grads[0].items() if "lora_B" in name))
    for name in grads[0]:
      torch.testing.assert_close(grads[1][name], grads[0][name], msg=name)


class ForwardOnlyTest(unittest.TestCase):
  """A forward-only pass (TrainingClient.forward) must not accumulate gradients.

  The cookbook's NLL evaluator calls forward() on held-out data immediately
  before a training step. The worker only clears gradients in optim_step, so
  a backward pass here would fold the test set into the next update.
  """

  def setUp(self) -> None:
    self.worker = worker_on(tiny_llama())
    self.worker.create_model("tiny", "full")
    self.params = list(self.worker.base.parameters())

  def test_forward_only_leaves_no_gradients(self) -> None:
    result = self.worker.forward_backward("full", datums()[:2], "cross_entropy", forward_only=True)
    self.assertTrue(all(param.grad is None for param in self.params))
    self.assertEqual(len(result["loss_fn_outputs"]), 2)
    self.assertEqual(len(result["loss_fn_outputs"][0]["logprobs"]["data"]), 4)
    self.assertGreater(result["metrics"]["loss:sum"], 0.0)

  def test_forward_only_after_training_does_not_change_gradients(self) -> None:
    self.worker.forward_backward("full", datums(), "cross_entropy")
    before = [param.grad.clone() for param in self.params if param.grad is not None]
    self.assertTrue(before)
    self.worker.forward_backward("full", datums(), "cross_entropy", forward_only=True)
    after = [param.grad for param in self.params if param.grad is not None]
    for a, b in zip(before, after, strict=True):
      self.assertTrue(torch.equal(a, b))

  def test_forward_only_returns_the_same_logprobs_as_a_training_pass(self) -> None:
    # No dropout in the tiny config, so eval and train mode agree.
    forward_only = self.worker.forward_backward("full", datums(), "cross_entropy", forward_only=True)
    trained = self.worker.forward_backward("full", datums(), "cross_entropy")
    for lhs, rhs in zip(forward_only["loss_fn_outputs"], trained["loss_fn_outputs"], strict=True):
      for a, b in zip(lhs["logprobs"]["data"], rhs["logprobs"]["data"], strict=True):
        self.assertAlmostEqual(a, b, places=5)


class PaddedBatchingTest(unittest.TestCase):
  def setUp(self) -> None:
    self.worker = worker_on(tiny_llama())
    self.worker.create_model("tiny", "full")

  def training_tensors(self, data):
    input_ids, attention_mask, input_lengths = self.worker.pad_model_inputs(data)
    target_token_ids, weights, lengths = self.worker.pad_targets_and_weights(data, input_lengths)
    logprobs = token_logprobs(self.worker.base, input_ids, attention_mask, target_token_ids)
    old_logprobs = self.worker.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in data], lengths, torch.float32)
    advantages = self.worker.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in data], lengths, torch.float32)
    return logprobs, weights, old_logprobs, advantages, lengths

  def test_padded_batch_logprobs_and_losses_match_per_example_math(self) -> None:
    data = datums()
    batch_logprobs, batch_weights, batch_old, batch_adv, batch_lengths = self.training_tensors(data)
    singles = [self.training_tensors([item]) for item in data]

    for row, (logprobs, weights, old, adv, lengths) in enumerate(singles):
      length = batch_lengths[row]
      self.assertEqual(length, lengths[0])
      torch.testing.assert_close(batch_logprobs[row, :length], logprobs[0, :length])
      torch.testing.assert_close(batch_weights[row, :length], weights[0, :length])
      torch.testing.assert_close(batch_weights[row, length:], torch.zeros_like(batch_weights[row, length:]))
      torch.testing.assert_close(batch_old[row, :length], old[0, :length])
      torch.testing.assert_close(batch_adv[row, :length], adv[0, :length])

    def single_sum(fn):
      return torch.stack([fn(logprobs, weights, old, adv).sum() for logprobs, weights, old, adv, _ in singles]).sum()

    torch.testing.assert_close(
      losses.cross_entropy_loss(batch_logprobs, batch_weights).sum(), single_sum(lambda lp, w, _o, _a: losses.cross_entropy_loss(lp, w))
    )
    torch.testing.assert_close(
      losses.importance_sampling_loss(batch_logprobs, batch_weights, batch_old, batch_adv).sum(), single_sum(losses.importance_sampling_loss)
    )
    ppo_config = {"clip_range": 0.2, "kl_coeff": 0.03}
    torch.testing.assert_close(
      losses.ppo_loss(batch_logprobs, batch_weights, batch_old, batch_adv, ppo_config).sum(),
      single_sum(lambda lp, w, o, a: losses.ppo_loss(lp, w, o, a, ppo_config)),
    )

  def test_token_budget_batches_preserve_examples(self) -> None:
    data = datums()
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "6"}):
      batches = self.worker.make_training_batches(data)
    seen = [idx for batch in batches for idx, _ in batch]
    self.assertCountEqual(seen, range(len(data)))
    for batch in batches:
      padded_tokens = max(len(item.model_input) for _, item in batch) * len(batch)
      self.assertTrue(len(batch) == 1 or padded_tokens <= 6)

  def test_padded_batches_preserve_the_client_output_shape(self) -> None:
    data = datums()
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "12"}):
      self.assertTrue(any(len(batch) > 1 for batch in self.worker.make_training_batches(data)))
      result = self.worker.forward_backward("full", data, "cross_entropy")
    self.assertEqual(len(result["loss_fn_outputs"]), len(data))
    for item, output in zip(data, result["loss_fn_outputs"], strict=True):
      self.assertEqual(output["logprobs"]["shape"], [min(len(item.model_input), len(item.loss_fn_inputs["target_tokens"].data))])


class CheckpointTest(unittest.TestCase):
  def test_a_saved_adapter_is_what_peft_loads_and_what_load_state_restores(self) -> None:
    worker = worker_on(tiny_llama())
    worker.create_model("tiny", "job-a", lora(4, 1))
    worker.forward_backward("job-a", datums(), "cross_entropy")
    worker.optim_step("job-a", ADAM)
    model = worker.models["job-a"]
    model.module.eval()

    with tempfile.TemporaryDirectory() as tmp:
      worker.save_state("job-a", tmp, include_optimizer=True)
      self.assertTrue(os.path.exists(os.path.join(tmp, "job-a", "adapter_config.json")))
      self.assertTrue(os.path.exists(os.path.join(tmp, "job-a", "adapter_model.safetensors")))
      self.assertTrue(os.path.exists(os.path.join(tmp, "adam.safetensors")))

      input_ids = torch.tensor([datums()[0].model_input])
      loaded_by_peft = PeftModel.from_pretrained(tiny_llama(), os.path.join(tmp, "job-a"))
      torch.testing.assert_close(loaded_by_peft(input_ids=input_ids).logits, model.module(input_ids=input_ids).logits)

      restored = worker_on(tiny_llama())
      self.assertEqual(restored.load_state("job-a", tmp, restore_optimizer=True), {"model_id": "job-a", "base_model": "tiny"})
    restored_model = restored.models["job-a"]
    self.assertEqual(restored_model.adam_step, 1)
    for name, param in model.params.items():
      self.assertTrue(torch.equal(restored_model.params[name], param), name)
      self.assertTrue(torch.equal(restored_model.adam_m[name], model.adam_m[name]), name)
      self.assertTrue(torch.equal(restored_model.adam_v[name], model.adam_v[name]), name)
    self.assertTrue(restored_model.params[name].requires_grad)

  def test_the_old_torch_optimizer_file_restores(self) -> None:
    worker = worker_on(tiny_llama())
    worker.create_model("tiny", "job-a", LoraConfig(rank=2, seed=1))
    params = worker.models["job-a"].params
    optimizer = torch.optim.AdamW(params.values(), lr=1e-2)
    for param in params.values():
      param.grad = torch.ones_like(param)
    optimizer.step()

    with tempfile.TemporaryDirectory() as tmp:
      worker.save_state("job-a", tmp)
      torch.save(optimizer.state_dict(), os.path.join(tmp, "optimizer.pt"))
      restored = worker_on(tiny_llama())
      restored.load_state("job-a", tmp, restore_optimizer=True)
    model = restored.models["job-a"]
    self.assertEqual(model.adam_step, 1)
    for index, (name, param) in enumerate(params.items()):
      self.assertTrue(torch.equal(model.adam_m[name], optimizer.state[param]["exp_avg"]), name)
      self.assertEqual(index, list(params).index(name))

  def test_a_full_checkpoint_round_trips(self) -> None:
    worker = worker_on(tiny_llama())
    worker.create_model("tiny", "full")
    worker.forward_backward("full", datums(), "cross_entropy")
    worker.optim_step("full", ADAM)

    with tempfile.TemporaryDirectory() as tmp:
      worker.save_state("full", tmp, include_optimizer=True, kind="state")
      self.assertTrue(os.path.exists(os.path.join(tmp, "model.safetensors")))
      self.assertTrue(os.path.exists(os.path.join(tmp, "config.json")))
      restored = TrainerWorker()
      restored.device = torch.device("cpu")
      restored.load_state("full", tmp, restore_optimizer=True)
    self.assertEqual(restored.base_name, "tiny")
    for (name, param), (_, expected) in zip(restored.base.named_parameters(), worker.base.named_parameters(), strict=True):
      self.assertTrue(torch.equal(param, expected), name)
      self.assertTrue(param.requires_grad)
    self.assertEqual(restored.models["full"].adam_step, 1)


@unittest.skipUnless(torch.cuda.is_available(), "sleep and wake move tensors between the accelerator and pinned host memory")
class SleepWakeTest(unittest.TestCase):
  def test_sleep_empties_the_device_and_wake_restores_every_tensor(self) -> None:
    worker = TrainerWorker()
    worker.base, worker.base_name = tiny_llama().to(worker.device), "tiny"
    worker.create_model("tiny", "job-a", LoraConfig(rank=4, seed=1))
    worker.forward_backward("job-a", datums(), "cross_entropy")
    worker.optim_step("job-a", ADAM)
    worker.forward_backward("job-a", datums(), "cross_entropy")
    before = {name: tensor.detach().clone() for name, tensor in enumerate(worker.tensors())}

    worker.sleep()
    self.assertTrue(all(tensor.device.type == "cpu" for tensor in worker.tensors()))
    self.assertTrue(all(tensor.is_pinned() for tensor in worker.tensors()))
    worker.wake_up()
    self.assertTrue(all(tensor.device.type == "cuda" for tensor in worker.tensors()))
    for name, tensor in enumerate(worker.tensors()):
      self.assertTrue(torch.equal(tensor, before[name]))
    worker.optim_step("job-a", ADAM)


if __name__ == "__main__":
  unittest.main()
