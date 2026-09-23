import unittest

import torch
import torch.nn as nn

from training.host_mirror import HostMirror


class TwoLayer(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(8, 8, bias=False)
    self.fc2 = nn.Linear(8, 8, bias=False)
    self.register_buffer("scale", torch.ones(8))


class HostMirrorBaselineTest(unittest.TestCase):
  def test_first_diff_seeds_and_reports_nothing(self):
    model = TwoLayer()
    mirror = HostMirror()
    self.assertIsNone(mirror.diff(model.fc1.weight))
    with torch.no_grad():
      model.fc1.weight[0, 0] += 1.0
    indices, values = mirror.diff(model.fc1.weight)
    self.assertEqual(indices.tolist(), [0])
    self.assertEqual(indices.dtype, torch.int32)
    self.assertTrue(torch.equal(values, model.fc1.weight.data.view(-1)[:1]))

  def test_diff_advances_the_baseline(self):
    model = TwoLayer()
    mirror = HostMirror()
    mirror.sync(model)
    with torch.no_grad():
      model.fc1.weight[1, 2] = 5.0
      model.fc2.weight[3, 4] = -5.0
    self.assertEqual(mirror.diff(model.fc1.weight)[0].tolist(), [1 * 8 + 2])
    self.assertEqual(mirror.diff(model.fc2.weight)[0].tolist(), [3 * 8 + 4])
    self.assertIsNone(mirror.diff(model.fc1.weight))
    self.assertIsNone(mirror.diff(model.fc2.weight))

  def test_sync_resets_the_baseline(self):
    model = TwoLayer()
    mirror = HostMirror()
    mirror.sync(model)
    with torch.no_grad():
      model.fc1.weight[0, 0] += 1.0
    mirror.sync(model)
    self.assertIsNone(mirror.diff(model.fc1.weight))


@unittest.skipUnless(torch.cuda.is_available(), "sleep/wake_up move state between CUDA and pinned host memory")
class HostMirrorLeaseTest(unittest.TestCase):
  def _trained(self) -> tuple[TwoLayer, torch.optim.Optimizer]:
    model = TwoLayer().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(4, 8, device="cuda")).sum().backward()
    optimizer.step()
    model(torch.randn(4, 8, device="cuda")).sum().backward()  # leave a pending gradient
    return model, optimizer

  def test_sleep_and_wake_round_trip(self):
    model, optimizer = self._trained()
    weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    moments = {name: optimizer.state[p]["exp_avg"].clone() for name, p in model.named_parameters()}

    mirror = HostMirror()
    mirror.sleep(model, optimizer)
    self.assertTrue(mirror.offloaded)
    for name, p in model.named_parameters():
      self.assertEqual(p.device.type, "cpu")
      self.assertEqual(p.grad.device.type, "cpu")
      self.assertEqual(optimizer.state[p]["exp_avg"].device.type, "cpu")
      self.assertEqual(p.shape, weights[name].shape)  # offloaded parameters keep their shape
    self.assertEqual(model.scale.device.type, "cpu")

    mirror.wake_up(model, optimizer)
    self.assertFalse(mirror.offloaded)
    for name, p in model.named_parameters():
      self.assertEqual(p.device.type, "cuda")
      self.assertTrue(torch.equal(p, weights[name]))
      self.assertTrue(torch.equal(p.grad, grads[name]))
      self.assertTrue(torch.equal(optimizer.state[p]["exp_avg"], moments[name]))
    self.assertEqual(model.scale.device.type, "cuda")

  def test_host_buffers_are_reused_across_leases(self):
    model, optimizer = self._trained()
    mirror = HostMirror()
    mirror.sleep(model, optimizer)
    first = {name: p.data_ptr() for name, p in model.named_parameters()}
    mirror.wake_up(model, optimizer)
    optimizer.step()
    optimizer.zero_grad()
    mirror.sleep(model, optimizer)
    self.assertEqual({name: p.data_ptr() for name, p in model.named_parameters()}, first)
    mirror.wake_up(model, optimizer)

  def test_baseline_survives_a_lease(self):
    model, optimizer = self._trained()
    mirror = HostMirror()
    mirror.sync(model)
    optimizer.step()  # changes the weights on the device
    changed_before = {name: mirror.diff(p)[0].numel() for name, p in model.named_parameters()}
    self.assertTrue(all(changed_before.values()))
    mirror.sleep(model, optimizer)
    mirror.wake_up(model, optimizer)
    for p in model.parameters():
      self.assertIsNone(mirror.diff(p))


if __name__ == "__main__":
  unittest.main()
