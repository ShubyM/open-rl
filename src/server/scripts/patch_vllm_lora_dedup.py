"""Patch vLLM's LoRA activation to avoid duplicate Gemma4 module aliases.

Gemma4's YOCO decoder split introduces shared module references. Older vLLM
builds register those aliases separately with `named_modules(remove_duplicate=False)`.
The aliases are needed by vLLM 0.20's torch compile path, but activation then
iterates those aliases and can set LoRA weights on one path before immediately
resetting the same wrapper object through an alias with no matching adapter
weights.

This helper keeps the alias registration intact and dedupes only activation by
wrapper object identity. Run it after installing or upgrading vLLM until the
upstream fix lands everywhere we care about.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

BAD = """        for module_name, module in self.modules.items():
            module_lora = self._get_lora_layer_weights(lora_model, module_name)
            if not module_lora:
                module.reset_lora(index)
                logger.debug(
                    "No LoRA weights found for module %s, skipping.", module_name
                )
                continue

            module.set_lora(
                index,
                module_lora.lora_a,
                module_lora.lora_b,
            )
            logger.debug("Successfully loaded LoRA weights for module %s.", module_name)
"""

GOOD = """        seen_modules: set[int] = set()
        for module_name, module in self.modules.items():
            module_key = id(module)
            if module_key in seen_modules:
                continue
            seen_modules.add(module_key)

            module_lora = self._get_lora_layer_weights(lora_model, module_name)
            if not module_lora:
                module.reset_lora(index)
                logger.debug(
                    "No LoRA weights found for module %s, skipping.", module_name
                )
                continue

            module.set_lora(
                index,
                module_lora.lora_a,
                module_lora.lora_b,
            )
            logger.debug("Successfully loaded LoRA weights for module %s.", module_name)
"""


def find_model_manager(venv: str | None = None) -> Path:
  if venv:
    candidates = list(Path(venv).rglob("vllm/lora/model_manager.py"))
    if candidates:
      return candidates[0]
  spec = importlib.util.find_spec("vllm.lora.model_manager")
  if spec and spec.origin:
    return Path(spec.origin)
  raise FileNotFoundError("Cannot find vllm/lora/model_manager.py")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--check", action="store_true", help="Check only, do not patch")
  parser.add_argument("--venv", type=str, help="Optional venv path to patch explicitly")
  args = parser.parse_args()

  path = find_model_manager(args.venv)
  source = path.read_text()

  if BAD not in source:
    if "seen_modules: set[int] = set()" in source:
      print(f"OK: {path} is already patched")
      return 0
    print(f"WARN: {path} has neither the buggy nor fixed pattern")
    return 1

  if args.check:
    print(f"NEEDS_PATCH: {path} activates duplicate LoRA module aliases")
    return 2

  path.write_text(source.replace(BAD, GOOD, 1))
  print(f"PATCHED: {path}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
