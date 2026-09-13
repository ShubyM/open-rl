"""CPU contracts for model policy and forward execution in AutoModel."""

import fnmatch
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from training import automodel_models as models


class Layer(torch.nn.Module):
  def __init__(self, scale: float):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.tensor(scale))

  def forward(self, x: torch.Tensor, *, shift: torch.Tensor) -> torch.Tensor:
    return x * self.weight + shift


def make_config(model_type, **kwargs):
  text = SimpleNamespace(model_type=model_type, **kwargs)
  return SimpleNamespace(get_text_config=lambda: text, output_hidden_states=False)


def policy_for(config, *, tp_size=1, cp_size=1, recompute_num_layers=4, attention="auto"):
  return models.model_policy(config, tp_size=tp_size, cp_size=cp_size, recompute_num_layers=recompute_num_layers, attention=attention)


def make_model(config, *, native):
  model = torch.nn.Module()
  model.config = config
  model.model = torch.nn.Module()
  model.model.norm = torch.nn.Identity()
  layer = torch.nn.Linear(2, 2)
  model.model.layers = torch.nn.ModuleDict({"0": layer}) if native else torch.nn.ModuleList([layer])
  model.gradient_checkpointing_enable = Mock()
  return model


class ForwardModel(torch.nn.Module):
  def __init__(self, *, nested=False):
    super().__init__()
    decoder = torch.nn.Module()
    decoder.norm = torch.nn.LayerNorm(4).double()
    self.model = torch.nn.Module() if nested else decoder
    if nested:
      self.model.language_model = decoder
    self.calls = []
    self.failure = None

  def forward(self, inputs_embeds, **kwargs):
    self.calls.append(kwargs)
    if self.failure == "before_norm":
      raise RuntimeError("model failed before norm")
    if self.failure == "skip_norm":
      return SimpleNamespace(hidden_states=(torch.zeros_like(inputs_embeds),))
    decoder = self.model.language_model if "language_model" in self.model._modules else self.model
    decoder.norm(inputs_embeds)
    if self.failure == "after_norm":
      raise RuntimeError("model failed after norm")
    return SimpleNamespace(hidden_states=None)


class AutomodelModelsTest(unittest.TestCase):
  def test_forward_hidden_states_preserves_values_gradients_and_existing_hooks(self):
    for nested in (False, True):
      with self.subTest(nested=nested):
        model = ForwardModel(nested=nested)
        norm = model.model.language_model.norm if nested else model.model.norm
        seen = Mock(return_value=None)
        hook = norm.register_forward_hook(seen)
        self.addCleanup(hook.remove)
        hooks = dict(norm._forward_hooks)
        for shift in (0.0, 0.5):
          inputs = (torch.randn(2, 3, 4, dtype=torch.double) + shift).requires_grad_()
          expected = norm(inputs)
          expected_grads = torch.autograd.grad(expected.square().sum(), [inputs, *norm.parameters()])
          seen.reset_mock()
          forward_kwargs = {"kernel_options": {"fwd_BLOCK_M": 64}}
          actual = models.forward_hidden_states(model, forward_kwargs, inputs_embeds=inputs, model_specific_metadata="preserved")
          self.assertEqual(norm._forward_hooks, hooks)
          seen.assert_called_once()
          self.assertFalse(model.calls[-1]["use_cache"])
          self.assertEqual(model.calls[-1]["logits_to_keep"], 1)
          self.assertNotIn("output_hidden_states", model.calls[-1])
          self.assertEqual(model.calls[-1]["model_specific_metadata"], "preserved")
          self.assertEqual(model.calls[-1]["kernel_options"], forward_kwargs["kernel_options"])
          actual_grads = torch.autograd.grad(actual.square().sum(), [inputs, *norm.parameters()])
          torch.testing.assert_close(actual, expected)
          for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

  def test_forward_hidden_states_removes_capture_hook_after_model_failures(self):
    model = ForwardModel()
    norm = model.model.norm
    inputs = torch.randn(1, 2, 4, dtype=torch.double, requires_grad=True)
    for failure in ("before_norm", "after_norm", "skip_norm"):
      with self.subTest(failure=failure):
        model.failure = failure
        message = "did not run the final norm" if failure == "skip_norm" else "model failed"
        with self.assertRaisesRegex(RuntimeError, message):
          models.forward_hidden_states(model, {}, inputs_embeds=inputs)
        self.assertEqual(len(norm._forward_hooks), 0)
        model.failure = None
        actual = models.forward_hidden_states(model, {}, inputs_embeds=inputs)
        torch.testing.assert_close(actual, norm(inputs))
        self.assertEqual(len(norm._forward_hooks), 0)

  def test_group_checkpointing_matches_plain_forward_and_backward(self):
    layers = torch.nn.ModuleDict({str(i): Layer(1.0 + i / 10) for i in range(6)})
    x = torch.randn(4, requires_grad=True)
    shift = torch.ones(4)

    def run():
      h = x
      for layer in layers.values():
        h = layer(x=h, shift=shift)
      return h.sum()

    expected = run()
    expected_grads = torch.autograd.grad(expected, [x, *layers.parameters()])
    names = list(layers.state_dict())
    model = make_model(make_config("qwen3_5_text"), native=True)
    model.model.layers = layers
    policy_for(model.config).apply(model)
    self.assertEqual(len(list(layers.values())), 2)
    actual = run()
    actual_grads = torch.autograd.grad(actual, [x, *layers.parameters()])
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
      torch.testing.assert_close(actual_grad, expected_grad)
    self.assertEqual(list(layers.state_dict()), names)
    layers.eval()
    self.assertEqual(len(list(layers.values())), 6)

  def test_per_layer_head_width_does_not_read_ambiguous_config_width(self):
    class LayeredConfig:
      model_type = "gemma4_text"
      per_layer_config = [SimpleNamespace(head_dim=256), SimpleNamespace(head_dim=512)]

      @property
      def head_dim(self):
        raise ValueError("Head width differs between layers")

    config = SimpleNamespace(get_text_config=LayeredConfig)
    policy = policy_for(config, attention="flex_attention")
    self.assertEqual(policy.forward_kwargs["kernel_options"]["fwd_BLOCK_M"], 16)
    self.assertEqual(policy.forward_kwargs["kernel_options"]["bwd_BLOCK_M1"], 16)
    for config in (make_config("llama", head_dim=128), make_config("llama")):
      self.assertEqual(policy_for(config, attention="flex_attention").forward_kwargs, {})

  def test_backbone_resolution_preserves_dense_multimodal_models_without_proxies(self):
    decoder = torch.nn.Module()
    decoder.norm = torch.nn.Identity()
    decoder.layers = torch.nn.ModuleDict({str(i): Layer(1.0) for i in range(6)})
    multimodal_model = torch.nn.Module()
    multimodal_model.config = make_config("qwen3_5_text")
    multimodal_model.model = torch.nn.Module()
    multimodal_model.model.language_model = decoder
    policy_for(multimodal_model.config).apply(multimodal_model)
    self.assertEqual(len(list(decoder.layers.values())), 2)
    decoder.layers.eval()
    self.assertEqual(len(list(decoder.layers.values())), 6)

  def test_lora_targets_cover_text_and_multimodal_backbones(self):
    targets = models.lora_target_modules(["q_proj", "k_proj"])
    for name in ("model.layers.0.self_attn.q_proj", "model.language_model.layers.0.self_attn.q_proj"):
      self.assertTrue(any(fnmatch.fnmatchcase(name, pattern) for pattern in targets), name)
    for name in ("mtp.layers.0.q_proj", "model.visual.blocks.0.q_proj", "lm_head"):
      self.assertFalse(any(fnmatch.fnmatchcase(name, pattern) for pattern in targets), name)

  def test_shared_kv_models_always_use_upstream_checkpointing(self):
    for model_type in ("gemma4_text", "qwen3_5_text"):
      with self.subTest(model_type=model_type):
        config = make_config(model_type, enable_moe_block=True, num_kv_shared_layers=2)
        policy = policy_for(config)
        self.assertEqual(policy.checkpoint_group_size, 0)
        self.assertTrue(policy.activation_checkpointing)

  def test_grouped_path_rejects_changed_backbone_without_enabling_hf_checkpointing(self):
    model = make_model(make_config("qwen3_5_text"), native=False)
    with self.assertRaisesRegex(RuntimeError, "requires a native ModuleDict"):
      policy_for(model.config).apply(model)
    model.gradient_checkpointing_enable.assert_not_called()

  def test_grouped_path_preserves_parameter_names_and_objects(self):
    model = make_model(make_config("qwen3_5_text"), native=True)
    parameters = dict(model.named_parameters())
    policy_for(model.config).apply(model)
    self.assertEqual(parameters.keys(), dict(model.named_parameters()).keys())
    for name, param in model.named_parameters():
      self.assertIs(param, parameters[name])
    model.gradient_checkpointing_enable.assert_not_called()

  def test_default_policy_uses_upstream_checkpointing(self):
    config = make_config("llama", head_dim=128)
    policy = policy_for(config)
    self.assertEqual(policy.load_kwargs, {"attn_implementation": "sdpa"})
    self.assertEqual(policy.forward_kwargs, {})
    self.assertEqual(policy.checkpoint_group_size, 0)
    self.assertTrue(policy.activation_checkpointing)
    self.assertIsNone(policy.tp_plan)
    model = make_model(config, native=False)
    model.config.output_hidden_states = True
    layers = model.model.layers
    policy.apply(model)
    self.assertFalse(model.config.output_hidden_states)
    self.assertIs(model.model.layers, layers)
    model.gradient_checkpointing_enable.assert_not_called()

  def test_native_policy_selects_grouping_and_disables_unused_prediction_layers(self):
    for model_type in ("qwen3_5_text", "qwen3_5_moe_text", "qwen3_next"):
      with self.subTest(model_type=model_type):
        config = make_config(model_type)
        policy = policy_for(config, cp_size=4, recompute_num_layers=3)
        self.assertEqual(policy.load_kwargs["backend"], {"attn": "sdpa"})
        self.assertEqual(policy.load_kwargs["attn_implementation"], "sdpa")
        if model_type.startswith("qwen3_5"):
          self.assertEqual(policy.load_kwargs["num_nextn_predict_layers"], 0)
        else:
          self.assertNotIn("num_nextn_predict_layers", policy.load_kwargs)
        self.assertEqual(policy.checkpoint_group_size, 3)
        self.assertFalse(policy.activation_checkpointing)
        self.assertEqual(policy.forward_kwargs, {})

  def test_gemma_policy_preserves_attention_options_with_context_parallelism(self):
    config = make_config("gemma4_text", head_dim=256, global_head_dim=512, num_kv_shared_layers=2)
    for cp_size in (1, 4):
      with self.subTest(cp_size=cp_size):
        policy = policy_for(config, cp_size=cp_size)
        self.assertEqual(policy.load_kwargs["attn_implementation"], "ffpa" if cp_size == 1 else "sdpa")
        self.assertFalse(policy.load_kwargs["use_sdpa_patching"])
        if cp_size > 1:
          self.assertEqual(policy.load_kwargs["text_config"], {"use_cache": False, "cp_full_attn_backend": "ffpa"})
        self.assertEqual(policy.forward_kwargs["kernel_options"]["fwd_BLOCK_M"], 64)
        self.assertEqual(policy.forward_kwargs["kernel_options"]["bwd_BLOCK_M1"], 32)
        self.assertEqual(policy.checkpoint_group_size, 0)
        self.assertTrue(policy.activation_checkpointing)
    explicit = policy_for(config, attention="flex_attention")
    self.assertEqual(explicit.load_kwargs, {"attn_implementation": "flex_attention"})
    self.assertEqual(explicit.forward_kwargs["kernel_options"]["fwd_BLOCK_M"], 16)
    self.assertEqual(explicit.forward_kwargs["kernel_options"]["bwd_BLOCK_M1"], 16)

  def test_gemma_moe_always_uses_upstream_checkpointing(self):
    for enable_moe_block in (False, True):
      policy = policy_for(make_config("gemma4_text", enable_moe_block=enable_moe_block))
      self.assertEqual(policy.checkpoint_group_size, 0)
      self.assertTrue(policy.activation_checkpointing)

  def test_disabled_checkpointing_disables_grouped_and_upstream_paths(self):
    for model_type in ("llama", "qwen3_5_text", "gemma4_text"):
      with self.subTest(model_type=model_type):
        policy = policy_for(make_config(model_type), recompute_num_layers=0)
        self.assertEqual(policy.checkpoint_group_size, 0)
        self.assertFalse(policy.activation_checkpointing)

  def test_tensor_parallel_plan_prefixes_backbone_and_preserves_output_layouts(self):
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    projection = ColwiseParallel()

    def translate(style):
      if style == "colwise":
        return projection
      if style == "replicate":
        return None
      raise ValueError("Unsupported packed expert style")

    for architecture, prefix in (("Qwen3_5ForCausalLM", "model"), ("Qwen3_5ForConditionalGeneration", "model.language_model")):
      with self.subTest(architecture=architecture):
        config = make_config(
          "qwen3_5_text",
          base_model_tp_plan={"layers.*.self_attn.q_proj": "colwise", "layers.*.linear_attn": "replicate", "layers.*.mlp.experts": "packed"},
        )
        config.architectures = [architecture]
        module = "nemo_automodel.components.distributed.parallelizer"
        translator = Mock(side_effect=translate)
        with patch.dict(sys.modules, {module: SimpleNamespace(translate_to_torch_parallel_style=translator)}):
          plan = policy_for(config, tp_size=2).tp_plan
        self.assertEqual(set(plan), {f"{prefix}.embed_tokens", f"{prefix}.layers.*.self_attn.q_proj", "lm_head"})
        self.assertIs(plan[f"{prefix}.layers.*.self_attn.q_proj"], projection)
        self.assertIsInstance(plan[f"{prefix}.embed_tokens"], RowwiseParallel)
        self.assertIsInstance(plan[f"{prefix}.embed_tokens"].input_layouts[0], Replicate)
        self.assertIsInstance(plan["lm_head"], ColwiseParallel)
        self.assertIsInstance(plan["lm_head"].output_layouts[0], Replicate)
        self.assertEqual(translator.call_count, 3)

  def test_output_projection_preserves_parameters_and_optional_bias_and_softcap(self):
    for with_bias in (False, True):
      with self.subTest(with_bias=with_bias):
        head = torch.nn.Linear(4, 11) if with_bias else torch.nn.Embedding(11, 4)
        config = make_config("gemma4_text", final_logit_softcapping=2.0) if with_bias else make_config("llama")
        model = SimpleNamespace(config=config, get_output_embeddings=Mock(return_value=head))
        projection = models.output_projection(model)
        self.assertIs(projection.weight, head.weight)
        self.assertIs(projection.bias, head.bias if with_bias else None)
        self.assertEqual(projection.softcap, 2.0 if with_bias else None)

  def test_adapter_conversion_keeps_tensor_identity_and_strips_only_peft_prefix(self):
    tensor = torch.randn(2, 4)
    state = {"base_model.model.model.layers.0.q_proj.lora_A.weight": tensor, "model.layers.0.q_proj.lora_B.weight": tensor}
    converted = models.adapter_state_from_hf(torch.nn.Module(), state)
    self.assertEqual(set(converted), {"model.layers.0.q_proj.lora_A.weight", "model.layers.0.q_proj.lora_B.weight"})
    self.assertTrue(all(value is tensor for value in converted.values()))
    self.assertIn("base_model.model.model.layers.0.q_proj.lora_A.weight", state)
    adapter = SimpleNamespace(from_hf=Mock(return_value={"base_model.model.model.layers.0.native_qkv.lora_A.weight": tensor}))
    converted = models.adapter_state_from_hf(SimpleNamespace(state_dict_adapter=adapter), state)
    adapter.from_hf.assert_called_once_with(state, device_mesh=None)
    self.assertEqual(set(converted), {"model.layers.0.native_qkv.lora_A.weight"})
    self.assertIs(converted["model.layers.0.native_qkv.lora_A.weight"], tensor)


if __name__ == "__main__":
  unittest.main()
