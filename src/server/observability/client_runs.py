"""A run's client log: logs.log under the openrl.log_path the client declared, such as the tinker-cookbook's log_path on a
volume the server also mounts. Every read is blocking, so callers run it in a thread."""

import os
import re

SCAN_BYTES = 4 << 20  # of logs.log per page
MESSAGE_CHARS = 16384
# The cookbook's file format, "%(name)s:%(lineno)d [%(levelname)s] %(message)s"; other lines continue the record above.
HEADER = re.compile(r"\S+:\d+ \[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\] ")


def log_file(log_path: str | None) -> str:
  """<log_path>/logs.log, resolved, when it lies under one of OPEN_RL_CLIENT_LOG_ROOTS. The path is the client's word, so
  one that leads elsewhere, by .. or a symlink, is refused rather than read."""
  roots = [os.path.realpath(root.strip()) for root in os.getenv("OPEN_RL_CLIENT_LOG_ROOTS", "").split(",") if root.strip()]
  if not log_path:
    raise LookupError("This run declared no openrl.log_path")
  if not roots:
    raise LookupError("No client log roots configured. Set OPEN_RL_CLIENT_LOG_ROOTS; see docs/configuration.md.")
  path = os.path.realpath(os.path.join(log_path, "logs.log"))
  if not any(os.path.commonpath([path, root]) == root for root in roots):
    raise LookupError(f"{log_path} is not under OPEN_RL_CLIENT_LOG_ROOTS")
  return path


def log_page(path: str, before: int | None, limit: int, q: str = "", severity: str | None = None) -> dict:
  """logs.log, newest first, in the shape of a Cloud Logging page. Its lines carry no time, so a record is a header line
  and the lines under it, such as a traceback, and next_cursor is the byte offset the next older page ends at."""
  with open(path, "rb") as f:
    size = f.seek(0, os.SEEK_END)
    end = size if before is None else min(before, size)
    start = max(0, end - SCAN_BYTES)
    f.seek(start)
    block = f.read(end - start)
  if start and (cut := block.find(b"\n") + 1):  # the first line began before the block
    block, start = block[cut:], start + cut
  lines, at = [], start
  for raw in block.splitlines(keepends=True):
    lines.append((at, raw.decode("utf-8", "replace").rstrip("\r\n")))
    at += len(raw)
  records, under, cursor = [], [], None
  for at, text in reversed(lines):
    under.append(text)
    # The file's first line starts a record, and so does a block's first line when the block has no header at all.
    if not (header := HEADER.match(text)) and at and not (at == start and cursor is None):
      continue
    message, under, cursor = "\n".join(reversed(under)), [], at
    level = header[1] if header else "UNKNOWN"
    if (not severity or level == severity) and q.lower() in message.lower():
      record = {"id": f"logs.log:{at}", "timestamp": None, "pod": "client", "role": "client", "node": None, "severity": level}
      records.append(record | {"message": message[:MESSAGE_CHARS], "message_truncated": len(message) > MESSAGE_CHARS})
      if len(records) == limit:
        break
  return {"records": records, "next_cursor": str(cursor) if cursor else None}
