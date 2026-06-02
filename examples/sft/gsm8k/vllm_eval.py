import argparse
import json
import re
import time

from vllm import LLM, SamplingParams

ANS_RE = re.compile(r"-?\d[\d,]*")


def extract(t):
  t = re.split(r"\n\s*Question:", t)[0]
  if "####" in t:
    m = ANS_RE.search(t.split("####")[-1])
    if m:
      return m.group(0).replace(",", "")
  n = ANS_RE.findall(t)
  return n[-1].replace(",", "") if n else None


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", required=True)
  parser.add_argument("--data", default="gsm8k_test.json")
  args = parser.parse_args()
  data = json.load(open(args.data))
  llm = LLM(model=args.path, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=1024, enforce_eager=True)
  sp = SamplingParams(temperature=0.0, max_tokens=256, stop=["\nQuestion:"])
  t0 = time.time()
  outs = llm.generate([d["prompt"] for d in data], sp)
  dt = time.time() - t0
  correct = sum(int(extract(o.outputs[0].text) == d["gold"]) for d, o in zip(data, outs))
  print("***************************************************************")
  print(f"[VLLM] {args.path} 0-shot GSM8K acc = {correct / len(data):.1%} on {len(data)} problems in {dt:.1f}s")
  print("***************************************************************")


if __name__ == "__main__":
  main()
