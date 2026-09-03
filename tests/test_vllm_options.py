import os
import unittest
from unittest.mock import patch

from server.vllm_options import gpu_memory_utilization, split_stop


class SplitStopTest(unittest.TestCase):
  def test_none(self):
    self.assertEqual(split_stop(None), (None, None))

  def test_bare_string(self):
    self.assertEqual(split_stop("\n\nUser:"), (["\n\nUser:"], None))

  def test_token_ids(self):
    self.assertEqual(split_stop([151645, 151643]), (None, [151645, 151643]))

  def test_strings(self):
    self.assertEqual(split_stop(["</s>", "\n\n"]), (["</s>", "\n\n"], None))

  def test_mixed(self):
    self.assertEqual(split_stop(["</s>", 151645]), (["</s>"], [151645]))

  def test_empty_sequence(self):
    self.assertEqual(split_stop([]), (None, None))

  def test_bool_is_not_a_token_id(self):
    self.assertEqual(split_stop([True]), (None, None))


if __name__ == "__main__":
  unittest.main()


class GpuMemoryUtilizationTest(unittest.TestCase):
  def test_explicit_fraction_wins(self) -> None:
    with patch.dict(os.environ, {"VLLM_GPU_MEMORY_UTILIZATION": "0.5", "OPEN_RL_ACCELERATOR_MEMORY": str(70 * 1024**3)}):
      self.assertEqual(gpu_memory_utilization(device_bytes=80 * 1024**3), 0.5)

  def test_budget_over_the_actual_device_capped(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_ACCELERATOR_MEMORY": str(7 * 1024**3)}, clear=True):
      self.assertAlmostEqual(gpu_memory_utilization(device_bytes=24 * 1024**3), 7 / 24)
      self.assertAlmostEqual(gpu_memory_utilization(device_bytes=80 * 1024**3), 7 / 80)
    with patch.dict(os.environ, {"OPEN_RL_ACCELERATOR_MEMORY": str(100 * 1024**3)}, clear=True):
      self.assertEqual(gpu_memory_utilization(device_bytes=80 * 1024**3), 0.90)

  def test_without_a_budget_the_default_holds(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertEqual(gpu_memory_utilization(device_bytes=80 * 1024**3), 0.90)
