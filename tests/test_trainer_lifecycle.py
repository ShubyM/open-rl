"""Process and command lifecycle: what the worker imports on the CPU path, how
the request processor treats failures, and the ordering and optimizer-rebinding
contracts.

The distributed-failure and ordering tests use a small fake worker and trainer
so they need no GPU; the CPU-import test runs a real subprocess against the tiny
Llama; the reload and restore tests use the real FFT worker on CPU.
"""

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from server.store import InMemoryStore
from server.training_requests_processor import Deployment, FatalWorkerError, TrainingRequestsProcessor, run_training_requests_processor
from tests.test_training_loop import TinyLlamaCase, training_data
from training import commands
from training.fft_trainer_worker import FFTTrainingWorker
from training.lora_trainer_worker import LoraTrainingWorker
from training.trainer_worker import Trainer, TrainingWorker

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs the CPU LoRA and FFT training path and asserts that no GPU-only backend
# was imported and that CUDA was never initialized as a side effect.
CPU_IMPORT_PROBE = """
import sys
import torch
from training import commands
from training.types import Datum
from training.lora_trainer_worker import LoraTrainingWorker
from training.fft_trainer_worker import FFTTrainingWorker

base = sys.argv[1]
data = [Datum(model_input=[3, 4, 5, 6], loss_fn_inputs={"target_tokens": {"data": [4, 5, 6, 7]}, "weights": {"data": [1.0, 1.0, 1.0, 1.0]}})]

lora = LoraTrainingWorker()
lora.device = torch.device("cpu")
lt = lora.create(commands.CreateModel(request_id="c", model_id="a", base_model=base, lora_config={"rank": 4, "seed": 1, "lora_dropout": 0.0}))
lt.forward_backward(data, "cross_entropy")
lt.optim_step({"learning_rate": 1e-2})

fft = FFTTrainingWorker()
fft.device = torch.device("cpu")
full_config = {"seed": 1, "cpu_offload": False, "weight_sync_strategy": "full"}
ft = fft.create(commands.CreateModel(request_id="c2", model_id="m", base_model=base, fine_tuning_type="full", full_config=full_config))
ft.forward_backward(data, "cross_entropy")
ft.optim_step({"learning_rate": 1e-2})

assert "nemo_automodel" not in sys.modules, "nemo_automodel was imported on the CPU path"
assert "torch_xla" not in sys.modules, "torch_xla was imported on the CPU path"
assert not torch.cuda.is_initialized(), "CUDA was initialized on the CPU path"
print("CPU_PATH_OK")
"""


class FakeWorker(TrainingWorker):
  """A shared worker whose saves stay off the GPU, so a SaveState runs without
  the lease and its barrier. Nothing here needs a real model."""

  def save_needs_gpu(self) -> bool:
    return False


class RaisingTrainer(Trainer):
  def __init__(self, exc: Exception):
    model = torch.nn.Linear(1, 1)
    super().__init__("m", model, list(model.parameters()))
    self.exc = exc

  def save_state(self, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict:
    raise self.exc


class RecordingTrainer(Trainer):
  def __init__(self, ops: list[str]):
    model = torch.nn.Linear(1, 1)
    super().__init__("m", model, list(model.parameters()))
    self.ops = ops

  def forward_backward(self, data, loss_fn, loss_config=None, forward_only=False) -> dict:
    self.ops.append("fb")
    return {"metrics": {}, "loss_fn_outputs": [], "loss_fn_output_type": "ArrayRecord"}

  def optim_step(self, adam_params) -> dict:
    self.ops.append("step")
    return {"metrics": {}}

  def save_state(self, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict:
    self.ops.append("save")
    return {"path": state_path}


def processor_with(trainer: Trainer) -> TrainingRequestsProcessor:
  processor = TrainingRequestsProcessor(InMemoryStore(), FakeWorker(), Deployment())
  processor.trainers["m"] = trainer
  return processor


class CpuImportIsolationTest(TinyLlamaCase):
  def test_cpu_path_imports_no_gpu_only_backends(self) -> None:
    env = {**os.environ, "OPEN_RL_TMP_DIR": os.path.join(self.tmp.name, "probe-tmp"), "RANK": "0", "WORLD_SIZE": "1", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-c", CPU_IMPORT_PROBE, self.base_model], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    self.assertEqual(result.returncode, 0, f"probe failed:\n{result.stdout}\n{result.stderr}")
    self.assertIn("CPU_PATH_OK", result.stdout)


class DistributedFailurePolicyTest(unittest.IsolatedAsyncioTestCase):
  async def test_unexpected_error_on_a_distributed_worker_is_fatal(self) -> None:
    processor = processor_with(RaisingTrainer(RuntimeError("kaboom")))
    command = commands.SaveState(request_id="s", model_id="m", state_path="/tmp/none")
    with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}):
      with self.assertRaises(FatalWorkerError):
        await processor.handle(command)
      answered, fatal = await processor.run_batch([command])
    self.assertIsInstance(fatal, FatalWorkerError)
    self.assertNotIn("s", answered)

  async def test_request_error_is_never_fatal(self) -> None:
    processor = processor_with(RaisingTrainer(ValueError("bad path")))
    command = commands.SaveState(request_id="s", model_id="m", state_path="/tmp/none")
    with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}):
      _, result = await processor.handle(command)
    self.assertEqual(result["type"], "RequestFailedResponse")
    self.assertIn("bad path", result["error_message"])

  async def test_unexpected_error_off_a_distributed_worker_is_a_request_failure(self) -> None:
    processor = processor_with(RaisingTrainer(RuntimeError("kaboom")))
    command = commands.SaveState(request_id="s", model_id="m", state_path="/tmp/none")
    with patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"}):
      _, result = await processor.handle(command)
    self.assertEqual(result["type"], "RequestFailedResponse")
    self.assertIn("kaboom", result["error_message"])

  async def test_hf_worker_refuses_to_run_under_torchrun(self) -> None:
    with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}), self.assertRaisesRegex(RuntimeError, "WORLD_SIZE>1"):
      await run_training_requests_processor(LoraTrainingWorker())


class FifoOrderingTest(unittest.IsolatedAsyncioTestCase):
  async def test_a_save_is_not_hoisted_ahead_of_later_gpu_work(self) -> None:
    """Grouping by needs_gpu must keep FIFO order: fb, then save, then step. If
    GPU work were hoisted the save would run last."""
    ops: list[str] = []
    processor = processor_with(RecordingTrainer(ops))
    batch = [
      commands.ForwardBackward(request_id="fb", model_id="m", data=training_data()),
      commands.SaveState(request_id="save", model_id="m", state_path="/tmp/none"),
      commands.OptimStep(request_id="step", model_id="m", adam_params={}),
    ]
    answered, fatal = await processor.run_batch(batch)
    self.assertIsNone(fatal)
    self.assertEqual(ops, ["fb", "save", "step"])
    self.assertEqual(set(answered), {"fb", "save", "step"})


class FFTReloadTest(TinyLlamaCase):
  def build_fft_worker_and_trainer(self):
    worker = FFTTrainingWorker()
    worker.device = torch.device("cpu")
    trainer = worker.create(self.create_full("m"))
    return worker, trainer

  def test_loadweights_rebinds_the_optimizer_to_the_fresh_params(self) -> None:
    worker, trainer = self.build_fft_worker_and_trainer()
    data = training_data()
    trainer.forward_backward(data, "cross_entropy")
    trainer.optim_step({"learning_rate": 1e-2})
    self.assertIsNotNone(trainer.optimizer)
    old_param_ids = {id(p) for p in trainer.params}

    state_path = os.path.join(self.tmp.name, "fft-reload")
    trainer.save_state(state_path)
    worker.reload(trainer, commands.LoadWeights(request_id="l", model_id="m", state_path=state_path))
    self.assertIsNone(trainer.optimizer, "reload without restore_optimizer drops the old optimizer")

    new_param_ids = {id(p) for p in trainer.params}
    self.assertTrue(old_param_ids.isdisjoint(new_param_ids), "reload swaps in fresh parameter objects")

    trainer.forward_backward(data, "cross_entropy")
    trainer.optim_step({"learning_rate": 1e-2})
    optimized = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    self.assertEqual(optimized, new_param_ids, "the rebuilt optimizer must own the new params, not the freed ones")

  def test_strict_optimizer_restore_fails_without_a_saved_optimizer(self) -> None:
    worker, trainer = self.build_fft_worker_and_trainer()
    trainer.forward_backward(training_data(), "cross_entropy")
    trainer.optim_step({"learning_rate": 1e-2})
    state_path = os.path.join(self.tmp.name, "fft-noopt")
    trainer.save_state(state_path)
    self.assertFalse(os.path.exists(os.path.join(state_path, "optimizer.pt")))

    fresh = FFTTrainingWorker()
    fresh.device = torch.device("cpu")
    with self.assertRaisesRegex(ValueError, "optimizer"):
      fresh.restore(
        commands.CreateModelFromState(request_id="r", model_id="m2", state_path=state_path, restore_optimizer=True, fine_tuning_type="full")
      )

  def test_sleep_and_wake_are_noops_on_cpu(self) -> None:
    """CPU has no offload target, so sleep/wake do nothing here. Real GPU offload
    is not exercised by this test."""
    _, trainer = self.build_fft_worker_and_trainer()
    before = [p.detach().clone() for p in trainer.params]
    trainer.sleep()
    trainer.wake_up()
    self.assertFalse(trainer._is_offloaded)
    for param, saved in zip(trainer.params, before):
      self.assertTrue(torch.equal(param.detach(), saved))


if __name__ == "__main__":
  unittest.main()
