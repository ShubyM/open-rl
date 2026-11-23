"""Checkpoint, sampler-weight, and disk path helpers shared by the gateway and workers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

TMP_DIR = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")


def sampling_ref(sampling_session_id: str) -> str:
  return re.sub(r"@[0-9a-f]{32}$", "", sampling_session_id)


def base_model_id_from_sampling_ref(model_id: str | None) -> str | None:
  if not model_id:
    return None
  if model_id.startswith("tinker://"):
    parts = model_id[9:].split("/")
    return parts[0] if len(parts) >= 3 and parts[1] == "sampler_weights" else model_id[9:]
  return model_id.split("-samp-")[0]


def resolve_sampler_weights_path(model_id: str, *, base_dir: str | None = None) -> str:
  rel = model_id[9:] if model_id.startswith("tinker://") else model_id.lstrip("/")
  path = os.path.join(base_dir or TMP_DIR, "sampler_full", rel)
  if not os.path.basename(path).startswith("sampler-") and os.path.exists(sw := os.path.join(path, "sampler_weights")):
    try:
      if steps := [int(d.split("-")[1]) for d in os.listdir(sw) if d.startswith("sampler-")]:
        path = os.path.join(sw, f"sampler-{max(steps)}")
    except Exception as exc:
      print(f"[API_SERVER] Warning: Failed parsing step subdirectories in {sw}: {exc}")
  return path


def tinker_checkpoint_dir(path: str, *, base_dir: str | None = None) -> str | None:
  if not path.startswith("tinker://"):
    return None
  owner, sep, rest = path[9:].partition("/weights/")
  return os.path.join(base_dir or TMP_DIR, "checkpoints", owner, "weights", rest) if (owner and sep) else None


def checkpoint_state_path(model_id: str, name: str, *, base_dir: str | None = None) -> str:
  if (state_dir := tinker_checkpoint_dir(name, base_dir=base_dir)) is not None:
    return state_dir
  return name if os.path.isabs(name) else os.path.join(base_dir or TMP_DIR, "checkpoints", model_id, "weights", name)


def tinker_state_path(state_path: str, *, base_dir: str | None = None) -> str:
  root = os.path.join(base_dir or TMP_DIR, "checkpoints") + os.sep
  if state_path.startswith(root):
    model_id, sep, rest = state_path[len(root) :].partition("/weights/")
    if model_id and sep:
      return f"tinker://{model_id}/weights/{rest}"
  return state_path


def checkpoint_info(path: str, *, base_dir: str | None = None) -> dict[str, Any]:
  state_dir = tinker_checkpoint_dir(path, base_dir=base_dir) or (path if os.path.isabs(path) else None)
  if not state_dir or not os.path.exists(meta_path := os.path.join(state_dir, "metadata.json")):
    raise FileNotFoundError(f"No checkpoint at {path}")
  with open(meta_path) as f:
    saved = json.load(f)
  if not isinstance(saved, dict) or not isinstance(saved.get("base_model"), str) or not saved["base_model"]:
    raise ValueError("Checkpoint metadata must specify base_model")
  if not isinstance(saved_id := saved.get("model_id", ""), str):
    raise ValueError("Checkpoint model_id must be a string")
  cfg_path = os.path.join(state_dir, saved_id, "adapter_config.json")
  if not os.path.exists(cfg_path):
    cfg_path = os.path.join(state_dir, "adapter_config.json")
  is_lora, rank = os.path.exists(cfg_path), None
  if is_lora:
    with open(cfg_path) as f:
      cfg = json.load(f)
    if not isinstance(cfg, dict):
      raise ValueError("Checkpoint adapter config must be an object")
    rank = cfg.get("r")
  return {"base_model": saved["base_model"], "is_lora": is_lora, "lora_rank": rank, "type": "weights_info"}


def visible(path: Path) -> bool:
  if path.is_file():
    return True
  for parent in path.parents:
    try:
      os.listdir(parent)
      break
    except OSError:
      continue
  return path.is_file()
