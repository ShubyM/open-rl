import json
import tempfile
import unittest
from pathlib import Path

from server.scripts import run_vllm_eval


class ClusterEvalScriptTest(unittest.TestCase):
  def test_extract_answer_prefers_gsm8k_marker(self) -> None:
    self.assertEqual(run_vllm_eval.extract_answer("reasoning\n#### 1,234"), "1234")

  def test_extract_answer_falls_back_to_last_number_before_next_question(self) -> None:
    text = "first try 7, then final answer is -42\nQuestion: ignore 99"
    self.assertEqual(run_vllm_eval.extract_answer(text), "-42")

  def test_load_eval_data_reads_custom_json(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "data.json"
      payload = [{"prompt": "Question: 1+1\nAnswer:", "gold": "2"}]
      path.write_text(json.dumps(payload), encoding="utf-8")

      self.assertEqual(run_vllm_eval.load_eval_data(str(path), examples=100), payload)


if __name__ == "__main__":
  unittest.main()
