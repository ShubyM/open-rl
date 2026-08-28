import unittest

from server.vllm_options import split_stop


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
