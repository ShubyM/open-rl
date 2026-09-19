"""Checkpoint references, filesystem resolution, and saved model information."""

import json
import os
from typing import Any


class CheckpointStore:
  def __init__(self, tmp_dir: str):
    self.root = os.path.join(tmp_dir, "checkpoints")

  def from_uri(self, path: str) -> str | None:
    if not path.startswith("tinker://"):
      return None
    owner, sep, rest = path[len("tinker://") :].partition("/weights/")
    if not (owner and sep):
      return None
    return os.path.join(self.root, owner, "weights", rest)

  def resolve(self, model_id: str, name: str) -> str:
    """A tinker reference retains its original owner when another model loads it."""
    if name.startswith("tinker://"):
      path = self.from_uri(name)
      if path is None:
        raise ValueError(f"{name} is not a tinker://<model>/weights/<name> path")
      return path
    if os.path.isabs(name):
      return name
    return os.path.join(self.root, model_id, "weights", name)

  def restore_path(self, path: str) -> str:
    """Legacy restore names are relative to the checkpoint root, not a new model."""
    return self.from_uri(path) or (path if os.path.isabs(path) else os.path.join(self.root, path))

  def to_uri(self, state_path: str) -> str:
    prefix = self.root + os.sep
    if state_path.startswith(prefix):
      model_id, sep, rest = state_path[len(prefix) :].partition("/weights/")
      if model_id and sep:
        return f"tinker://{model_id}/weights/{rest}"
    return state_path

  def info(self, path: str) -> dict[str, Any] | None:
    state_dir = self.from_uri(path)
    metadata_path = os.path.join(state_dir, "metadata.json") if state_dir else None
    if not metadata_path or not os.path.exists(metadata_path):
      return None
    with open(metadata_path) as f:
      saved = json.load(f)
    adapter_config_path = os.path.join(state_dir, saved.get("model_id", ""), "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)
    rank = None
    if is_lora:
      with open(adapter_config_path) as f:
        rank = json.load(f).get("r")
    return {"base_model": saved["base_model"], "is_lora": is_lora, "lora_rank": rank, "type": "weights_info"}
