"""Equivalence tests for the chunked/fused target-logprob path.

The trainer avoids materializing the full [batch, seq, vocab] logits tensor (the
large-vocab lm_head OOM) by running the backbone and projecting to the target
logprob in vocab-sized chunks. These tests assert that path is numerically
identical -- values and gradients -- to the original full-logits
log_softmax(...).gather(...) computation.
"""

import os
import sys
import unittest
import unittest.mock
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training import trainer_worker  # noqa: E402


class _Backbone(nn.Module):
  def __init__(self, vocab: int, hidden: int):
    super().__init__()
    self.embed = nn.Embedding(vocab, hidden)
    self.proj = nn.Linear(hidden, hidden)
    self.attention_masks = []

  def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
    self.attention_masks.append(attention_mask)
    hidden = self.proj(torch.tanh(self.embed(input_ids)))
    return SimpleNamespace(last_hidden_state=hidden)


class _FakeCausalLM(nn.Module):
  """Minimal stand-in for a HF CausalLM: a `.model` backbone exposing
  last_hidden_state, a tied-style lm_head, and a config with softcapping."""

  def __init__(self, vocab: int, hidden: int, softcap: float | None = None):
    super().__init__()
    self.model = _Backbone(vocab, hidden)
    self.lm_head = nn.Linear(hidden, vocab, bias=False)
    text_config = SimpleNamespace(final_logit_softcapping=softcap, _attn_implementation="sdpa")
    self.config = SimpleNamespace(get_text_config=lambda: text_config)

  def get_output_embeddings(self):
    return self.lm_head

  def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
    hidden = self.model(input_ids).last_hidden_state
    logits = self.lm_head(hidden)
    config = self.config.get_text_config()
    if config.final_logit_softcapping is not None:
      softcap = config.final_logit_softcapping
      logits = softcap * torch.tanh(logits / softcap)
    return SimpleNamespace(logits=logits)


class _FakeLoraModel(nn.Module):
  """peft LoraModel shim: holds the causal LM as `.model` and delegates."""

  def __init__(self, causal):
    super().__init__()
    self.model = causal

  def __getattr__(self, name):
    try:
      return super().__getattr__(name)
    except AttributeError:
      return getattr(self.model, name)


class _FakePeftModel(nn.Module):
  """peft PeftModelForCausalLM shim: attribute access delegates through
  base_model, so `.model` resolves to the FULL causal LM (lm_head included) —
  the exact trap backbone resolution must sidestep via get_base_model()."""

  def __init__(self, causal):
    super().__init__()
    self.base_model = _FakeLoraModel(causal)

  def get_base_model(self):
    return self.base_model.model

  def __getattr__(self, name):
    try:
      return super().__getattr__(name)
    except AttributeError:
      return getattr(self.base_model, name)

  def forward(self, *args, **kwargs):
    return self.base_model.model(*args, **kwargs)


def _reference_logprobs(model, input_ids, attention_mask, target_token_ids):
  """The original full-logits computation this change replaces."""
  logits = model(input_ids, attention_mask=attention_mask).logits[:, : target_token_ids.shape[1], :]
  return torch.nn.functional.log_softmax(logits, dim=-1).gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)


class ComputeTargetLogprobsTest(unittest.TestCase):
  def setUp(self):
    torch.manual_seed(0)
    self.worker = trainer_worker.BaseTrainerWorker()
    self.worker.device = torch.device("cpu")
    self.batch, self.seq, self.hidden, self.vocab = 3, 11, 16, 97
    self.input_ids = torch.randint(0, self.vocab, (self.batch, self.seq))
    self.attention_mask = torch.ones(self.batch, self.seq, dtype=torch.long)
    self.target_ids = torch.randint(0, self.vocab, (self.batch, self.seq))

  def _run(self, softcap, chunk, fused, target_len=None):
    target = self.target_ids if target_len is None else self.target_ids[:, :target_len]
    model = _FakeCausalLM(self.vocab, self.hidden, softcap=softcap).double()
    env = {"OPEN_RL_FUSED_LOGPROB": "1" if fused else "0", "OPEN_RL_LOGPROB_CHUNK": str(chunk)}
    with unittest.mock.patch.dict(os.environ, env):
      out = self.worker.compute_target_logprobs(model, self.input_ids, self.attention_mask, target)
    return model, out

  def test_peft_wrapper_resolves_true_backbone(self):
    """A peft-wrapped model must use the fused backbone path: its attribute
    delegation makes `.model` resolve to the full causal LM, which previously
    caused a discarded forward plus the full-logits fallback — at long context
    a tens-of-GiB logits tensor outside the activation-offload scope."""
    causal = _FakeCausalLM(self.vocab, self.hidden).double()
    peft = _FakePeftModel(causal)
    with unittest.mock.patch.dict(os.environ, {"OPEN_RL_FUSED_LOGPROB": "1"}):
      out = self.worker.compute_target_logprobs(peft, self.input_ids, self.attention_mask, self.target_ids)
    # The backbone must run EXACTLY once: twice means the old discarded-forward
    # + full-logits fallback behavior.
    self.assertEqual(len(causal.model.attention_masks), 1)
    reference = _reference_logprobs(causal, self.input_ids, self.attention_mask, self.target_ids)
    torch.testing.assert_close(out, reference)

  def test_values_match_reference(self):
    for softcap in (None, 30.0):
      for chunk in (1, 7, 100000):
        with self.subTest(softcap=softcap, chunk=chunk):
          model, fused = self._run(softcap, chunk, fused=True)
          ref = _reference_logprobs(model, self.input_ids, self.attention_mask, self.target_ids)
          self.assertEqual(fused.shape, ref.shape)
          torch.testing.assert_close(fused, ref, rtol=1e-9, atol=1e-9)

  def test_fallback_path_matches_reference(self):
    # OPEN_RL_FUSED_LOGPROB=0 exercises the logit - logsumexp full-logits path.
    model, out = self._run(softcap=30.0, chunk=8, fused=False)
    ref = _reference_logprobs(model, self.input_ids, self.attention_mask, self.target_ids)
    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)

  def test_shorter_target_length(self):
    model, fused = self._run(softcap=None, chunk=5, fused=True, target_len=6)
    ref = _reference_logprobs(model, self.input_ids, self.attention_mask, self.target_ids[:, :6])
    torch.testing.assert_close(fused, ref, rtol=1e-9, atol=1e-9)

  def test_all_ones_mask_is_omitted_for_attention_backend(self):
    model, _ = self._run(softcap=None, chunk=5, fused=True)
    self.assertIsNone(model.model.attention_masks[0])

  def test_gradients_match_reference(self):
    # Fused (chunked + checkpointed) gradients must equal full-logits gradients.
    model_ref = _FakeCausalLM(self.vocab, self.hidden, softcap=30.0).double()
    model_fused = _FakeCausalLM(self.vocab, self.hidden, softcap=30.0).double()
    model_fused.load_state_dict(model_ref.state_dict())

    ref = _reference_logprobs(model_ref, self.input_ids, self.attention_mask, self.target_ids)
    ref.sum().backward()

    with unittest.mock.patch.dict(os.environ, {"OPEN_RL_FUSED_LOGPROB": "1", "OPEN_RL_LOGPROB_CHUNK": "7"}):
      fused = self.worker.compute_target_logprobs(model_fused, self.input_ids, self.attention_mask, self.target_ids)
    fused.sum().backward()

    ref_grads = dict(model_ref.named_parameters())
    for name, param in model_fused.named_parameters():
      with self.subTest(param=name):
        torch.testing.assert_close(param.grad, ref_grads[name].grad, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
  unittest.main()
