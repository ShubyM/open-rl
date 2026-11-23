"""Tinker wire format: end-to-end binary protobuf for forward_backward and SampleResponse.

The gateway validates and forwards protobuf bytes untouched; workers read/write NumPy buffers
directly (zero `.tolist()`). JSON edge converters handle <0.25 SDKs. Never imports `torch`.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from google.protobuf.message import DecodeError

from server.proto import tinker_public_pb2 as pb

assert sys.byteorder == "little", "the wire format is little endian"

PROTO, JSON, MASK_LOGPROB = "application/x-protobuf", "application/json", -99999.0
NUMPY_DTYPES: dict[int, tuple[Any, str]] = {
  pb.DTYPE_FLOAT32: (np.float32, "float32"),
  pb.DTYPE_INT64: (np.int64, "int64"),
  pb.DTYPE_INT32: (np.int32, "int64"),
}
_LABEL_DTYPES = {"float32": (pb.DTYPE_FLOAT32, np.float32), "int64": (pb.DTYPE_INT64, np.int64), "int32": (pb.DTYPE_INT32, np.int32)}


class ProtoDecodeError(ValueError):
  """Malformed or unsupported protobuf request/response."""


def _parse_pb(cls: type, raw: bytes) -> Any:
  try:
    return cls.FromString(raw)
  except DecodeError as exc:
    raise ProtoDecodeError(f"malformed {cls.__name__} protobuf: {exc}") from exc


def _f32_bytes(values: Any) -> bytes:
  if not isinstance(values, np.ndarray):
    values = [math.nan if v is None else v for v in values]
  return np.asarray(values, dtype=np.float32).reshape(-1).tobytes()


def proto_result(result_type: str, proto_bytes: bytes, *, metrics: dict[str, float] | None = None) -> dict[str, Any]:
  return {"type": result_type, "proto_bytes": bytes(proto_bytes), **({"metrics": metrics} if metrics is not None else {})}


def is_proto_result(result: Any) -> bool:
  return isinstance(result, dict) and isinstance(result.get("proto_bytes"), (bytes, bytearray, memoryview))


@dataclass(frozen=True)
class Datum:
  tokens: np.ndarray
  inputs: dict[str, np.ndarray]


@dataclass(frozen=True)
class SampleOut:
  stop: bool
  tokens: np.ndarray | Sequence[int]
  logprobs: np.ndarray | Sequence[float | None] | None = None
  topk: tuple[np.ndarray, np.ndarray] | None = None


# ***** requests *****


def parse(raw: bytes, *, validate: bool = True) -> pb.ForwardBackwardRequest:
  """Parse and validate a protobuf `ForwardBackwardRequest` without expanding tensors."""
  msg = _parse_pb(pb.ForwardBackwardRequest, raw)
  if not validate:
    return msg
  for d in msg.data:
    for chunk in d.model_input:
      if (kind := chunk.WhichOneof("chunk")) != "encoded_text":
        raise ProtoDecodeError(f"model_input chunk of kind {kind or 'unset'!r} is not supported; text only")
      if len(chunk.encoded_text.tokens) % 4:
        raise ProtoDecodeError(f"encoded_text.tokens length {len(chunk.encoded_text.tokens)} is not a multiple of 4")
    for name, t in d.loss_fn_inputs.items():
      if t.WhichOneof("encoding") == "sparse_csr":
        raise ProtoDecodeError(f"loss_fn_inputs[{name!r}] is sparse; only dense tensors are supported")
      if (spec := NUMPY_DTYPES.get(t.dtype)) is None:
        raise ProtoDecodeError(f"loss_fn_inputs[{name!r}] has unsupported dtype {pb.DType.Name(t.dtype)}")
      if len(t.dense) % np.dtype(spec[0]).itemsize:
        raise ProtoDecodeError(f"loss_fn_inputs[{name!r}] payload length {len(t.dense)} is misaligned")
      if len(t.shape) > 1:
        raise ProtoDecodeError(f"loss_fn_inputs[{name!r}] expected 1-D tensor, got shape {list(t.shape)}")
  return msg


def cost(req: pb.ForwardBackwardRequest) -> int:
  return sum(len(c.encoded_text.tokens) // 4 for d in req.data for c in d.model_input)


def loss_config(req: pb.ForwardBackwardRequest) -> dict[str, Any]:
  if req.loss_fn_config_v2:
    return {k: v.text if v.WhichOneof("value") == "text" else v.number for k, v in req.loss_fn_config_v2.items()}
  return dict(req.loss_fn_config)


def datums(req: pb.ForwardBackwardRequest | bytes) -> list[Datum]:
  """Decode `ForwardBackwardRequest` (or raw bytes) directly to NumPy `Datum` objects."""
  msg = parse(req) if isinstance(req, (bytes, bytearray, memoryview)) else req
  out: list[Datum] = []
  for d in msg.data:
    chunks = [np.frombuffer(c.encoded_text.tokens, np.int32) for c in d.model_input]
    toks = np.concatenate(chunks) if chunks else np.zeros(0, np.int32)
    inputs = {k: np.frombuffer(t.dense, NUMPY_DTYPES[t.dtype][0]).copy() for k, t in d.loss_fn_inputs.items()}
    out.append(Datum(toks if toks.flags.writeable else toks.copy(), inputs))
  return out


# ***** outputs *****


def output(logprobs: Sequence[Any], metrics: dict[str, float]) -> bytes:
  """Pack per-datum logprobs directly into binary `ForwardBackwardOutput` bytes."""
  out = pb.ForwardBackwardOutput(loss_fn_output_type="ArrayRecord", metrics={k: float(v) for k, v in metrics.items()})
  if logprobs:
    arrs = [np.asarray(x, dtype=np.float32).reshape(-1) for x in logprobs]
    field = out.loss_fn_outputs.add(num_datums=len(arrs)).fields["logprobs"]
    field.dtype, field.data = pb.DTYPE_FLOAT32, np.concatenate(arrs).tobytes()
    field.offsets = np.cumsum([0] + [4 * len(a) for a in arrs], dtype=np.int64).tobytes()
  return out.SerializeToString()


def pack_topk(dst: Any, topk: tuple[np.ndarray, np.ndarray]) -> None:
  ids, lps = np.asarray(topk[0], dtype=np.int32), np.asarray(topk[1], dtype=np.float32)
  dst.token_ids, dst.logprobs, dst.k, dst.length = ids.tobytes(), lps.tobytes(), int(ids.shape[-1]), int(ids.shape[0])


def sample_output(
  sequences: Sequence[SampleOut],
  prompt_logprobs: Any = None,
  topk_prompt: tuple[np.ndarray, np.ndarray] | None = None,
  prompt_cache_hit_tokens: int = 0,
) -> bytes:
  """Pack sampled sequences directly into binary `SampleResponse` bytes."""
  out = pb.SampleResponse()
  for s in sequences:
    seq = out.sequences.add()
    seq.stop_reason = pb.STOP_REASON_STOP if s.stop else pb.STOP_REASON_LENGTH
    seq.tokens = np.asarray(s.tokens, dtype=np.int32).reshape(-1).tobytes()
    if s.logprobs is not None and len(s.logprobs) > 0:
      seq.logprobs = _f32_bytes(s.logprobs)
    if s.topk is not None:
      pack_topk(seq.topk_sampled_logprobs, s.topk)
  if prompt_logprobs is not None and len(prompt_logprobs) > 0:
    out.prompt_logprobs = _f32_bytes(prompt_logprobs)
  if topk_prompt is not None:
    pack_topk(out.topk_prompt_logprobs, topk_prompt)
  if prompt_cache_hit_tokens:
    out.prompt_cache_hit_tokens = int(prompt_cache_hit_tokens)
  return out.SerializeToString()


# ***** JSON edge converters (gateway only, <0.25 SDK compat) *****


def _infer_dtype(name: str, raw: Any) -> tuple[int, Any, Sequence[Any], list[int] | None]:
  is_d = isinstance(raw, dict)
  vals, shape, label = (raw.get("data") or [], raw.get("shape"), str(raw.get("dtype") or "").lower()) if is_d else (raw or [], None, "")
  if label in _LABEL_DTYPES:
    return (*_LABEL_DTYPES[label], vals, shape)
  is_int = ("token" in name or name in ("target_tokens", "input_ids")) and not any(isinstance(v, float) for v in vals)
  return (pb.DTYPE_INT64, np.int64, vals, shape) if is_int else (pb.DTYPE_FLOAT32, np.float32, vals, shape)


def json_to_fb_proto(payload: dict[str, Any], *, forward_only: bool | None = None) -> bytes:
  """Encode a JSON `ForwardBackwardRequest` dict to binary protobuf bytes."""
  inp = payload.get("forward_backward_input") or payload.get("forward_input") or {}
  msg = pb.ForwardBackwardRequest(
    model_id=str(payload.get("model_id") or ""),
    seq_id=int(payload.get("seq_id") or 0),
    loss_fn=str(inp.get("loss_fn") or "cross_entropy"),
    forward_only=bool(payload.get("forward_only", False) if forward_only is None else forward_only),
  )
  for k, v in (inp.get("loss_fn_config") or {}).items():
    if isinstance(v, str):
      msg.loss_fn_config_v2[k].text = v
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
      msg.loss_fn_config[k] = msg.loss_fn_config_v2[k].number = float(v)
  for d in inp.get("data") or []:
    dst, mi = msg.data.add(), d.get("model_input")
    for ch in mi.get("chunks") or [] if isinstance(mi, dict) else [{"tokens": mi or []}]:
      dst.model_input.add().encoded_text.tokens = np.asarray(ch.get("tokens") or [], dtype=np.int32).tobytes()
    for name, raw_t in (d.get("loss_fn_inputs") or {}).items():
      dt, np_dt, vals, shape = _infer_dtype(name, raw_t)
      t = dst.loss_fn_inputs[name]
      t.dtype = dt
      t.dense = _f32_bytes(vals) if np_dt is np.float32 else np.asarray(vals, dtype=np_dt).tobytes()
      t.shape.extend(int(s) for s in (shape if shape is not None else [len(vals)]))
  return msg.SerializeToString()


def fb_output_to_json(raw: bytes) -> dict[str, Any]:
  """Decode binary `ForwardBackwardOutput` bytes to JSON dict at the gateway edge."""
  msg, outputs = _parse_pb(pb.ForwardBackwardOutput, raw), []
  for rec in msg.loss_fn_outputs:
    items: list[dict[str, Any]] = [{} for _ in range(rec.num_datums)]
    for name, field in rec.fields.items():
      np_dt, label = NUMPY_DTYPES[field.dtype]
      offs, trailing = np.frombuffer(field.offsets, np.int64).tolist(), list(field.trailing_shape)
      tprod = math.prod(trailing) if trailing else 1
      for i in range(rec.num_datums):
        vals = np.frombuffer(field.data[offs[i] : offs[i + 1]], np_dt).tolist()
        items[i][name] = {"data": vals, "dtype": label, "shape": [len(vals) // tprod if tprod else len(vals), *trailing]}
    outputs.extend(items)
  out_type = msg.loss_fn_output_type or "ArrayRecord"
  return {"type": "forward_backward", "loss_fn_output_type": out_type, "metrics": dict(msg.metrics), "loss_fn_outputs": outputs}


def sample_response_to_json(raw: bytes) -> dict[str, Any]:
  """Decode binary `SampleResponse` bytes to JSON dict at the gateway edge."""
  msg = _parse_pb(pb.SampleResponse, raw)
  seqs = []
  for s in msg.sequences:
    stop = "stop" if s.stop_reason == pb.STOP_REASON_STOP else "length"
    item: dict[str, Any] = {"tokens": np.frombuffer(s.tokens, np.int32).tolist(), "stop_reason": stop}
    if s.HasField("logprobs") or s.logprobs:
      item["logprobs"] = [None if math.isnan(v) else v for v in np.frombuffer(s.logprobs, np.float32).tolist()]
    seqs.append(item)
  out: dict[str, Any] = {"type": "sample", "sequences": seqs}
  if msg.HasField("prompt_logprobs") or msg.prompt_logprobs:
    out["prompt_logprobs"] = [None if math.isnan(v) else v for v in np.frombuffer(msg.prompt_logprobs, np.float32).tolist()]
  if msg.prompt_cache_hit_tokens:
    out["prompt_cache_hit_tokens"] = int(msg.prompt_cache_hit_tokens)
  return out
