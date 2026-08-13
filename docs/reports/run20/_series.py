"""Fork-aware reader for a cookbook metrics.jsonl.

metrics.jsonl is append-only across resumes, so run20's step column reads
...12, 13, 10, 11 after it crashed at step 14 and restarted from checkpoint
000010. Replaying in file order and dropping every step at or past a restart
keeps the branch actually being trained; a flat read splices the dead attempt
onto the live one. Every plot in this directory needs that, so it lives here
rather than being copied into each -- one copy going stale silently produces a
wrong-but-plausible curve, which is the worst failure mode a plot has.
"""

import json


def _raw(path):
    # The step-None rows are end-of-run evals with no training step attached;
    # they would plot at x=0 and misread as a starting score.
    return [r for r in (json.loads(l) for l in open(path)) if r.get("step") is not None]


def load(path):
    """Rows for the branch currently being trained. No-op for runs that never resumed."""
    out = {}
    for r in _raw(path):
        for stale in [k for k in out if k >= r["step"]]:
            del out[stale]
        out[r["step"]] = r
    return [out[k] for k in sorted(out)]


def resume_steps(path):
    """Steps where training restarted from an older checkpoint.

    The pruning matters: after the fork at 10 the file replays 10, 11, 12, 13,
    and without dropping the dead branch from `seen` every replayed step looks
    like a fresh resume. One restart would be drawn as four.
    """
    seen, found = set(), []
    for r in _raw(path):
        if any(k >= r["step"] for k in seen):
            found.append(r["step"])
            seen -= {k for k in seen if k >= r["step"]}
        seen.add(r["step"])
    return found
