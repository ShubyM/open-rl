"""Megatron backend pieces that do not need megatron installed.

Neither megatron-core nor megatron-bridge is a dependency, so most of the worker
cannot run here. Two parts can, and they are the two that would fail silently:
the data-parallel sharding hooks (wrong shard -> tensor-parallel peers train on
different tokens) and the DDP buffer offload's pointer arithmetic (wrong offset
-> parameters alias the wrong weights after a wake_up).
"""

import unittest
import unittest.mock

import torch

from training.megatron_worker import MegatronTrainingWorker
from training.trainer_worker import BaseTrainerWorker, Datum


def datum(token: int) -> Datum:
  return Datum(
    model_input=[token, token],
    loss_fn_inputs={"target_tokens": {"data": [token, token]}, "weights": {"data": [1.0, 1.0]}},
  )


class RecordingWorker(BaseTrainerWorker):
  """A BaseTrainerWorker that records which datums its shard actually saw."""

  def __init__(self, shard_rank: int = 0, shard_count: int = 1):
    super().__init__()
    self._shard_rank = shard_rank
    self._shard_count = shard_count
    self.seen: list[int] = []

  def shard_rank(self) -> int:
    return self._shard_rank

  def shard_count(self) -> int:
    return self._shard_count

  def shard_all_reduce_sum(self, value: float) -> float:
    return value

  def shard_all_gather_object(self, value):
    # Stand in for the other ranks: every datum this shard did not own comes
    # back from a peer, exactly as the real all_gather does.
    peers = {idx: {"logprobs": {"data": [0.0, 0.0], "dtype": "float32", "shape": [2]}} for idx in range(6)}
    return [value, peers]

  def compute_target_logprobs(self, model, input_ids, attention_mask, target_token_ids):
    self.seen.extend(int(row[0]) for row in input_ids)
    return torch.zeros_like(target_token_ids, dtype=torch.float32, requires_grad=True)


class ShardingHookTest(unittest.TestCase):
  """forward_backward must shard through the hooks, not through global rank.

  Under tensor parallelism the ranks holding one model are not independent data
  shards -- they hold different slices of the same weights and have to be fed
  the same tokens. The Megatron worker points these hooks at its data-parallel
  subgroup; if forward_backward ignored them it would hand each tensor-parallel
  peer a different batch and every gradient would be wrong.
  """

  def test_overridden_hooks_select_the_shard(self) -> None:
    worker = RecordingWorker(shard_rank=1, shard_count=2)
    result = worker.forward_backward(torch.nn.Linear(1, 1), [datum(i) for i in range(6)], "cross_entropy")

    self.assertEqual(sorted(worker.seen), [1, 3, 5])
    # Every datum still gets an output; the ones this shard skipped come back
    # from the gather.
    self.assertEqual(len(result["loss_fn_outputs"]), 6)

  def test_default_hooks_keep_single_process_behaviour(self) -> None:
    worker = RecordingWorker()
    worker.forward_backward(torch.nn.Linear(1, 1), [datum(i) for i in range(6)], "cross_entropy")

    self.assertEqual(sorted(worker.seen), [0, 1, 2, 3, 4, 5])


class FakeBucket:
  def __init__(self, param_data: torch.Tensor):
    self.param_data = param_data
    self.grad_data = None


class FakeBuffer:
  """The shape of megatron-core's _ParamAndGradBuffer that offload depends on."""

  def __init__(self, numel: int):
    self.param_data = torch.arange(numel, dtype=torch.float32)
    self.grad_data = torch.zeros(numel, dtype=torch.float32)
    self.buckets = [FakeBucket(self.param_data[: numel // 2]), FakeBucket(self.param_data[numel // 2 :])]


class FakeChunk:
  def __init__(self, buffer: FakeBuffer, shapes: list[tuple[int, int]]):
    self.buffers = [buffer]
    self._params = []
    offset = 0
    for shape in shapes:
      size = shape[0] * shape[1]
      param = torch.nn.Parameter(torch.empty(shape))
      # Exactly how Megatron builds them: a view into the flat buffer.
      param.data = buffer.param_data[offset : offset + size].view(shape)
      param.grad = buffer.grad_data[offset : offset + size].view(shape)
      self._params.append(param)
      offset += size

  def parameters(self):
    return iter(self._params)


class BufferOffloadTest(unittest.TestCase):
  """sleep()/wake_up() move the flat buffer and rebuild every view into it.

  nn.Module.to() cannot do this: each parameter's .data is a view, so .to()
  copies the views and leaves the flat buffer -- the tensor that holds the
  memory -- on the GPU. The offload frees nothing and the next grad reduction
  writes into storage no parameter reads. move_flat_buffers instead moves the
  flat tensor and re-derives each view's offset from its old data_ptr, so the
  arithmetic below is what keeps weights attached to themselves.
  """

  def setUp(self) -> None:
    self.buffer = FakeBuffer(24)
    self.chunk = FakeChunk(self.buffer, [(2, 4), (4, 4)])
    self.worker = MegatronTrainingWorker.__new__(MegatronTrainingWorker)
    self.worker.model_chunks = [self.chunk]

  def test_flat_buffers_are_discovered(self) -> None:
    self.assertEqual(self.worker.flat_buffers(), [self.buffer])

  def test_a_buffer_without_param_data_disables_offload(self) -> None:
    self.worker.model_chunks = [type("Bare", (), {"buffers": [object()], "parameters": lambda self: iter(())})()]
    self.assertIsNone(self.worker.flat_buffers())

  def test_views_into_finds_params_grads_and_buckets(self) -> None:
    flat = self.buffer.param_data
    low = flat.data_ptr()
    high = low + flat.numel() * flat.element_size()
    views = self.worker.views_into(self.buffer, low, high)

    # Two parameters plus two bucket views alias param_data; the two gradient
    # views alias grad_data and must not be swept up with them.
    self.assertEqual(len(views), 4)
    self.assertEqual(sorted(view.numel() for view in views), [8, 12, 12, 16])

  def test_offsets_reconstruct_every_view_after_a_move(self) -> None:
    flat = self.buffer.param_data
    base, itemsize = flat.data_ptr(), flat.element_size()
    views = self.worker.views_into(self.buffer, base, base + flat.numel() * itemsize)
    originals = [view.clone() for view in views]

    # Same arithmetic move_flat_buffers runs, against a relocated storage.
    moved = flat.clone()
    for view, original in zip(views, originals, strict=True):
      offset = (view.data_ptr() - base) // itemsize
      rebuilt = moved[offset : offset + view.numel()].view(view.shape)
      self.assertTrue(torch.equal(rebuilt, original))
      self.assertEqual(rebuilt.data_ptr(), moved.data_ptr() + offset * itemsize)

  def test_move_to_the_same_device_is_a_no_op(self) -> None:
    pointers = [param.data.data_ptr() for param in self.chunk.parameters()]
    self.worker.move_flat_buffers("cpu")
    self.assertEqual([param.data.data_ptr() for param in self.chunk.parameters()], pointers)


class BackendGuardTest(unittest.TestCase):
  def test_pipeline_parallelism_is_refused_at_construction(self) -> None:
    import training.megatron_worker as megatron_worker

    with unittest.mock.patch.object(megatron_worker, "MEGATRON_PP", 2):
      with self.assertRaisesRegex(RuntimeError, "tensor and data parallelism only"):
        MegatronTrainingWorker()

  def test_missing_dependency_names_both_packages(self) -> None:
    with self.assertRaisesRegex(RuntimeError, "megatron-core and megatron-bridge"):
      MegatronTrainingWorker().load_base_model("google/gemma-4-12B-it")


if __name__ == "__main__":
  unittest.main()
