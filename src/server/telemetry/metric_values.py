"""Client metric values, checked the same way by the gateway and by the exporter that sends
them; standard library only, so the exporter runs outside the server environment."""

import math
import re

RUN_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
METRICS_PER_ROW = 500


def finite_metrics(values: dict) -> dict[str, float]:
  """Normalize once at each untrusted boundary, before JSON encoding or storage."""
  result = {}
  for name, value in values.items():
    if len(name) > 200 or not isinstance(value, int | float) or isinstance(value, bool):
      continue
    try:
      value = float(value)
    except OverflowError:
      continue
    if math.isfinite(value):
      result[name] = value
      if len(result) == METRICS_PER_ROW:
        break
  return result
