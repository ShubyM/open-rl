"""Which sampler weight versions to apply, and in what order.

A version directory holds a sparse delta from the version before it; every
Nth one also holds a full snapshot under full/. A sampler that has applied
`current` needs only what follows it. A sampler starting cold needs the last
full snapshot at or before the target, then every delta after it.
"""

import os


def versions_in(folder: str) -> list[str]:
  """Version directories in write order."""
  try:
    names = os.listdir(folder)
  except OSError:
    return []
  dirs = [p for p in (os.path.join(folder, n) for n in names) if os.path.isdir(p)]
  return sorted(dirs, key=os.path.getmtime)


def version_chain(current: str | None, target: str) -> list[str]:
  """Paths to apply, oldest first, to bring a sampler at `current` to `target`."""
  target = target.rstrip("/")
  versions = versions_in(os.path.dirname(target))
  if target not in versions:
    return [target]
  upto = versions[: versions.index(target) + 1]
  if current is not None:
    current = current.rstrip("/")
    # The usual step: the next delta. Anything else, apply the target alone,
    # as before; the chain is only reconstructed from a cold start.
    return upto[upto.index(current) + 1 :] if current in upto else [target]
  for i in range(len(upto) - 1, -1, -1):
    full = os.path.join(upto[i], "full")
    if os.path.isdir(full):
      return [full] + upto[i + 1 :]
  # No full snapshot yet: the first version is a delta from the base model.
  return upto
