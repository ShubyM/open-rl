"""Draw one chat-format sample from a running vLLM server for the probe's sampler mode.

  python scripts/sample_for_probe.py MODEL URL OUT.pt [MAX_TOKENS]

Writes prompt token ids, the sampled completion ids and vLLM's logprob of every
sampled token (temperature 1, no top-p), which scripts/automodel_probe.py
PROBE_MODE=sampler scores again with the trainer.
"""

import json
import sys
import urllib.request

import torch
from transformers import AutoTokenizer

model, url, out = sys.argv[1], sys.argv[2], sys.argv[3]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
tokenizer = AutoTokenizer.from_pretrained(model)
messages = [
  {
    "role": "user",
    "content": (
      "You are outside counsel to Cascadia Industrial Holdings. Draft a detailed memorandum to the General Counsel "
      "comparing the target's representations and warranties in the draft share purchase agreement with what the "
      "diligence reports found, flagging every gap and proposing specific indemnity language. Think step by step first."
    ),
  }
]
prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
if not isinstance(prompt_ids, list):
  prompt_ids = prompt_ids["input_ids"]
request = {
  "model": model,
  "prompt": prompt_ids,
  "max_tokens": max_tokens,
  "temperature": 1.0,
  "top_p": 1.0,
  "logprobs": 0,
  "return_tokens_as_token_ids": True,
  "seed": 1234,
}
response = urllib.request.urlopen(
  urllib.request.Request(f"{url}/v1/completions", data=json.dumps(request).encode(), headers={"Content-Type": "application/json"}),
  timeout=3600,
)
choice = json.loads(response.read())["choices"][0]
completion_ids = [int(token.split(":")[1]) for token in choice["logprobs"]["tokens"]]
completion_logprobs = choice["logprobs"]["token_logprobs"]
torch.save(
  {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "completion_logprobs": completion_logprobs, "sampler": f"vllm@{url}", "model": model},
  out,
)
print(f"wrote {out}: prompt {len(prompt_ids)} tokens, completion {len(completion_ids)} tokens, finish={choice['finish_reason']}, mean logprob {sum(completion_logprobs) / len(completion_logprobs):.3f}")
print(tokenizer.decode(completion_ids[:80]))
