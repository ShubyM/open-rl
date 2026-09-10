# OpenRL Examples

This directory contains examples, demos, and helper scripts for using the OpenRL framework. These are not part of the core library but serve as recipes for training and evaluation.

## Prerequisites

* **Install [uv](https://docs.astral.sh/uv/):** Follow the official installation guide to install the fast Python package manager.
* **Synchronize Dependencies:** Run the following command to set up the environment:
  ```bash
  cd examples
  uv sync
  ```

## Client commands

Run client tools from `examples/`; uv discovers this directory's
`pyproject.toml`, installs its declared commands, and keeps the environment
in sync automatically:

```bash
uv run harvey-train --help
uv run harvey-eval --help
uv run harvey-results log_dir=artifacts/harvey-labs/my-run json=True plot=True
uv run harvey-plot log_dir=artifacts/harvey-labs/my-run out=run.png
```

From the repository root, use `uv run --project examples <command>`.
Paths remain relative to your current directory. The root project is the
server runtime; the client has its own lockfile and CPU dependencies.
`[project.scripts]` declares CLI commands. Use `--locked` in CI to reject
lockfile drift.

Run the Harvey renderer regression test from the repository root with
`uv run --project examples python -m harvey_labs.renderers.qwen35_renderer`.
It uses standard `unittest` in the examples environment.

See [Harvey LAB RL](harvey_labs) for setup and recipe options.

---

## Examples Overview

### Supervised Fine-Tuning (SFT)
* **[Pig Latin Translation](sft/pig-latin):** Teaches a model to perform specialized Pig Latin transformations, demonstrating custom token-level targets and loss masks.
* **[Text-to-SQL SFT](sft/text-to-sql):** Adapts Gemma 3 into a specialized database query assistant capable of generating SQL statements.
* **[FunctionGemma](sft/function-gemma):** Provides a recipe specifically targeted at fine-tuning tool-use capabilities, enabling models to reliably select and invoke functions.

### Reinforcement Learning (RL)
* **[Text-to-SQL RL](rl/text-to-sql):** Runs the Gemma 4 SFT+RL recipe with SQL execution rewards and curve plotting.

### Autoresearch
* **[Autoresearch Demo](autoresearch):** Runs code-RL researchers against the same OpenRL gateway using cookbook DeepCoder rewards, Sandbox Fusion, and optional Agent Sandbox CRDs.

### Tinker Cookbook
* **[Tinker Cookbook Recipes](tinker-cookbook):** Examples showing how to run [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) recipes with OpenRL.

---
