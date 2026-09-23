"""Offload between GPU leases and the delta baseline, on the FFT worker itself."""

import unittest

import torch
import torch.nn as nn

from training.fft_trainer_worker import FFTTrainingWorker


class TwoLayer(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(8, 8, bias=False)
    self.fc2 = nn.Linear(8, 8, bias=False)
    self.register_buffer("scale", torch.ones(8))

  def forward(self, x):
    return self.fc2(self.fc1(x) * self.scale)


def worker_for(model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> FFTTrainingWorker:
  worker = FFTTrainingWorker()
  worker.weight_sync_strategy = "delta"
  worker.model = model
  worker.optimizer = optimizer
  worker.prepare_model_for_training()  # seeds the delta baseline
  return worker


class DeltaBaselineTest(unittest.TestCase):
  def test_delta_advances_the_baseline(self):
    model = TwoLayer()
    worker = worker_for(model)
    with torch.no_grad():
      model.fc1.weight[1, 2] = 5.0
      model.fc2.weight[3, 4] = -5.0
    delta = worker.compute_weight_delta()
    self.assertEqual(delta.names, ["fc1.weight", "fc2.weight"])
    self.assertEqual([indices.tolist() for indices in delta.indices], [[1 * 8 + 2], [3 * 8 + 4]])
    self.assertEqual(delta.indices[0].dtype, torch.int32)
    self.assertEqual([values.tolist() for values in delta.values], [[5.0], [-5.0]])
    self.assertEqual(worker.compute_weight_delta().changed_elements, 0)

  def test_prepare_resets_the_baseline(self):
    model = TwoLayer()
    worker = worker_for(model)
    with torch.no_grad():
      model.fc1.weight[0, 0] += 1.0
    worker.prepare_model_for_training()
    self.assertEqual(worker.compute_weight_delta().changed_elements, 0)


@unittest.skipUnless(torch.cuda.is_available(), "sleep/wake_up move state between CUDA and pinned host memory")
class GpuLeaseTest(unittest.TestCase):
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

    worker = worker_for(model, optimizer)
    worker.sleep()
    self.assertTrue(worker.offloaded)
    for name, p in model.named_parameters():
      self.assertEqual(p.device.type, "cpu")
      self.assertEqual(p.grad.device.type, "cpu")
      self.assertEqual(optimizer.state[p]["exp_avg"].device.type, "cpu")
      self.assertEqual(p.shape, weights[name].shape)  # offloaded parameters keep their shape
    self.assertEqual(model.scale.device.type, "cpu")

    worker.wake_up()
    self.assertFalse(worker.offloaded)
    for name, p in model.named_parameters():
      self.assertEqual(p.device.type, "cuda")
      self.assertTrue(torch.equal(p, weights[name]))
      self.assertTrue(torch.equal(p.grad, grads[name]))
      self.assertTrue(torch.equal(optimizer.state[p]["exp_avg"], moments[name]))
    self.assertEqual(model.scale.device.type, "cuda")

  def test_host_buffers_are_reused_across_leases(self):
    model, optimizer = self._trained()
    worker = worker_for(model, optimizer)
    worker.sleep()
    first = {name: p.data_ptr() for name, p in model.named_parameters()}
    worker.wake_up()
    optimizer.step()
    optimizer.zero_grad()
    worker.sleep()
    self.assertEqual({name: p.data_ptr() for name, p in model.named_parameters()}, first)
    worker.wake_up()

  def test_baseline_survives_a_lease(self):
    model, optimizer = self._trained()
    worker = worker_for(model, optimizer)
    optimizer.step()  # changes the weights on the device
    self.assertEqual(len(worker.compute_weight_delta().names), 2)
    worker.sleep()
    worker.wake_up()
    self.assertEqual(worker.compute_weight_delta().changed_elements, 0)


if __name__ == "__main__":
  unittest.main()
