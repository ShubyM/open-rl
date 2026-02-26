# Open-RL Client

This directory contains the client-side scripts for interacting with the Open-RL API.

## RLVR Demo

The RLVR (Reinforcement Learning with Verifiable Rewards) demo showcases training a model to answer questions in a specific format using a reward function that verifies the correctness and format of the answer.

It supports parallel training jobs, allowing you to train multiple behaviors simultaneously (e.g., answering capital cities vs. just providing the answer).

![RLVR Result](./rlvr_result.png)

## FunctionGemma SFT

Use `functiongemma_sft.py` to reproduce tool-calling SFT:

```bash
uv run --python 3.12 python functiongemma_sft.py
```

Dataset source:
- Primary: Hugging Face `bebechien/SimpleToolCalling`
- Fallback: `client/data/functiongemma_simple_tool_calling.json`
