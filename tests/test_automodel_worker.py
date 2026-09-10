import torch

from training.automodel_worker import GroupCheckpointedLayers, round_robin_permutation


def rank_shards(cp_size: int, padded: int) -> list[torch.Tensor]:
  """Positions each rank holds under torch's load-balanced CP layout."""
  chunk = padded // (2 * cp_size)
  chunks = torch.arange(padded).split(chunk)
  return [torch.cat((chunks[r], chunks[2 * cp_size - 1 - r])) for r in range(cp_size)]


def test_round_robin_permutation_restores_position_order():
  for cp_size, padded in ((2, 16), (4, 64), (8, 4096)):
    positions = torch.arange(padded, dtype=torch.float32)
    gathered = torch.cat([positions[idx] for idx in rank_shards(cp_size, padded)]).unsqueeze(0)
    perm = round_robin_permutation(cp_size, padded, torch.device("cpu"))
    restored = torch.zeros_like(gathered).index_copy(1, perm, gathered)
    assert torch.equal(restored[0], positions)
    assert perm.unique().numel() == padded


class Layer(torch.nn.Module):
  def __init__(self, scale: float):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.tensor(scale))

  def forward(self, x: torch.Tensor, *, shift: torch.Tensor) -> torch.Tensor:
    return x * self.weight + shift


def test_group_checkpointing_matches_plain_forward_and_backward():
  layers = torch.nn.ModuleDict({str(i): Layer(1.0 + i / 10) for i in range(6)})
  x = torch.randn(4, requires_grad=True)
  shift = torch.ones(4)

  def run(module_dict):
    h = x
    for layer in module_dict.values():
      h = layer(x=h, shift=shift)
    return h.sum()

  expected = run(layers)
  expected_grads = torch.autograd.grad(expected, [x, *layers.parameters()])

  layers.__class__ = GroupCheckpointedLayers
  layers.group_size = 4
  assert len(list(layers.values())) == 2
  actual = run(layers)
  actual_grads = torch.autograd.grad(actual, [x, *layers.parameters()])
  assert torch.allclose(actual, expected)
  for a, e in zip(actual_grads, expected_grads):
    assert torch.allclose(a, e)

  layers.eval()
  assert len(list(layers.values())) == 6
