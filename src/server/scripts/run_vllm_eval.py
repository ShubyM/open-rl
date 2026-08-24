import json
import os
import re
import time
from pathlib import Path
from typing import Any

_ANSWER_RE = re.compile(r"-?\d[\d,]*")


def extract_answer(text: str) -> str | None:
  text = re.split(r"\n\s*Question:", text)[0]
  if "####" in text:
    match = _ANSWER_RE.search(text.split("####")[-1])
    if match:
      return match.group(0).replace(",", "")
  numbers = _ANSWER_RE.findall(text)
  return numbers[-1].replace(",", "") if numbers else None


def load_eval_data(data_path: str | None, examples: int) -> list[dict[str, Any]]:
  if data_path:
    with Path(data_path).open(encoding="utf-8") as f:
      return json.load(f)

  from datasets import load_dataset

  dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{examples}]")
  return [
    {
      "prompt": f"Question: {item['question']}\nAnswer:",
      "gold": item["answer"].split("####")[-1].strip().replace(",", ""),
    }
    for item in dataset
  ]


def main() -> None:
  from vllm import LLM, SamplingParams

  model_path = os.environ["MODEL_PATH"]
  data = load_eval_data(os.getenv("DATA_PATH"), int(os.getenv("EVAL_EXAMPLES", "100")))
  llm = LLM(
    model=model_path,
    dtype="bfloat16",
    gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
    max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "1024")),
    enforce_eager=True,
  )
  params = SamplingParams(temperature=0.0, max_tokens=256, stop=["\nQuestion:"])
  start = time.time()
  outputs = llm.generate([item["prompt"] for item in data], params)
  elapsed = time.time() - start
  generations = [output.outputs[0].text for output in outputs]
  graded = [item for item in data if item.get("gold") is not None]
  correct = sum(int(extract_answer(gen) == str(item["gold"])) for item, gen in zip(data, generations) if item.get("gold") is not None)
  results = {
    "model_path": model_path,
    "num_examples": len(data),
    "elapsed_s": round(elapsed, 1),
    "accuracy": correct / len(graded) if graded else None,
    "generations": generations,
  }
  print("***************************************************************")
  if graded:
    print(f"[EVAL] {model_path}: {correct}/{len(graded)} = {correct / len(graded):.1%} in {elapsed:.1f}s")
  else:
    print(f"[EVAL] {model_path}: generated {len(data)} completions in {elapsed:.1f}s")
  print("***************************************************************")
  results_path = os.getenv("RESULTS_PATH")
  if results_path:
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
      json.dump(results, f)
    print(f"[EVAL] wrote {results_path}")


if __name__ == "__main__":
  main()
