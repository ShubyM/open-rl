"""Engine and sampling options shared by the two vLLM sampler workers."""

from collections.abc import Sequence


def split_stop(stop: str | Sequence[str] | Sequence[int] | None) -> tuple[list[str] | None, list[int] | None]:
  """Split a Tinker `stop` into vLLM's `stop` and `stop_token_ids`.

  Tinker types stop as `str | Sequence[str] | Sequence[int]` and which one
  arrives depends on the renderer; vLLM keeps strings and token ids in separate
  arguments. Each half is None when empty so callers do not override a vLLM
  default with [].
  """
  if stop is None:
    return None, None
  if isinstance(stop, str):
    return [stop], None
  strings = [s for s in stop if isinstance(s, str)]
  # bool is an int subclass; a stray True would be read as token id 1.
  token_ids = [t for t in stop if isinstance(t, int) and not isinstance(t, bool)]
  return (strings or None), (token_ids or None)
