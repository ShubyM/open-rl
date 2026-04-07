import json
import os
import re
from typing import Any

DEFAULT_STORAGE_ROOT = "/tmp/open-rl"
STATE_KIND_CHECKPOINT = "state"
STATE_KIND_SNAPSHOT = "snapshot"
STATE_METADATA_FILENAME = "open_rl_state.json"
OPTIMIZER_STATE_FILENAME = "optimizer.pt"


def get_storage_root() -> str:
  return os.environ.get("OPEN_RL_TMP_DIR", DEFAULT_STORAGE_ROOT)


def get_peft_root() -> str:
  return os.path.join(get_storage_root(), "peft")


def get_checkpoints_root() -> str:
  return os.path.join(get_storage_root(), "checkpoints")


def get_snapshots_root() -> str:
  return os.path.join(get_storage_root(), "snapshots")


def ensure_dir(path: str) -> str:
  os.makedirs(path, exist_ok=True)
  return path


def sanitize_artifact_name(name: str | None, default_name: str) -> str:
  raw_name = (name or "").strip()
  if not raw_name:
    raw_name = default_name
  raw_name = raw_name.replace("\\", "/")
  raw_name = "_".join(part for part in raw_name.split("/") if part not in {"", ".", ".."})
  safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._")
  return safe_name or default_name


def build_artifact_path(name: str | None, kind: str, model_id: str) -> str:
  default_name = f"{model_id}_{kind}"
  safe_name = sanitize_artifact_name(name, default_name)
  if kind == STATE_KIND_SNAPSHOT:
    root = ensure_dir(get_snapshots_root())
  else:
    root = ensure_dir(get_checkpoints_root())
  return os.path.join(root, safe_name)


def encode_tinker_path(model_id: str, model_path: str) -> str:
  return f"tinker://{model_id}|{model_path}"


def decode_tinker_path(model_path: str | None) -> tuple[str | None, str | None]:
  if not model_path or not model_path.startswith("tinker://"):
    return None, model_path

  encoded = model_path[len("tinker://") :]
  if "|" not in encoded:
    return None, encoded

  model_id, path = encoded.split("|", 1)
  return model_id or None, path or None


def write_state_metadata(state_path: str, metadata: dict[str, Any]) -> None:
  ensure_dir(state_path)
  with open(os.path.join(state_path, STATE_METADATA_FILENAME), "w", encoding="utf-8") as handle:
    json.dump(metadata, handle)


def read_state_metadata(state_path: str) -> dict[str, Any]:
  metadata_path = os.path.join(state_path, STATE_METADATA_FILENAME)
  if not os.path.exists(metadata_path):
    return {}

  with open(metadata_path, encoding="utf-8") as handle:
    data = json.load(handle)
  if not isinstance(data, dict):
    return {}
  return data
