"""Assemble the committed record of run19 from a full artifact mirror.

The raw run is ~1.1 GB, most of it tool-result text repeated inside
train_rollout_summaries.jsonl. That does not belong in git, but it should not
be the only copy either -- the boxes are spot instances. So:

  * eval_episodes.jsonl -> committed. One line per held-out episode with every
                        metric the judge emitted. ~100 KB, and every number in
                        the report is recomputable from it.
  * metrics, config, code.diff, checkpoints, logs -> committed as-is.
  * evals/*.jsonl.gz  -> full eval transcripts, ~46 MB. NOT committed; the
                        repo ignores run data (.gitignore `runs/`).
  * train_episodes.jsonl.gz -> training rollouts as a slim index: all metrics,
                        plus lengths and heads of each assistant message and
                        tool result instead of their full text. Preserves the
                        overflow / tool-volume / churn analyses at ~2% of the
                        raw size, but still 26 MB, so also NOT committed.

The two heavy outputs are listed in this directory's .gitignore. They rebuild
from the mirror in one command, so git holds the report and the numbers while
the mirror holds the transcripts.

Full fidelity lives in the mirror (see README). Re-run this after the training
job finishes to pick up the last iterations:

    python docs/runs/run19/build.py ~/open-rl-runs/harvey-labs
"""

import gzip
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ["run19", "run19r"]
# run19's judge died partway through step 15; its 15-19 are all-zero artefacts
# that run19r re-ran. Keep them anyway -- they are the evidence for the
# post-mortem -- but the dump/plot path filters them out.
FLAT = ["metrics.jsonl", "config.json", "code.diff", "checkpoints.jsonl"]

HEAD = 240  # chars of each message kept for grepping (errors, churn phrases)


def slim_episode(e, run, it, split):
    metrics = {}
    for s in e["steps"]:
        metrics.update(s.get("metrics") or {})
    metrics.update(e.get("trajectory_metrics") or {})
    steps = []
    for s in e["steps"]:
        logs = s["logs"]
        ac = logs.get("assistant_content") or ""
        rec = {"ac_chars": len(ac), "ac_head": ac[:HEAD]}
        calls, results = [], []
        for k in sorted(logs):
            if k.startswith("tool_call_"):
                calls.append(logs[k])
            elif k.startswith("tool_result_"):
                v = logs[k] or ""
                results.append({"chars": len(v), "head": v[:HEAD]})
        if calls:
            rec["calls"] = calls
        if results:
            rec["results"] = results
        steps.append(rec)
    return {"run": run, "iter": it, "split": split,
            "group_idx": e["group_idx"], "metrics": metrics, "steps": steps}


def main(mirror):
    out_meta = os.path.join(HERE, "meta")
    out_eval = os.path.join(HERE, "evals")
    os.makedirs(out_meta, exist_ok=True)
    os.makedirs(out_eval, exist_ok=True)

    train_path = os.path.join(HERE, "train_episodes.jsonl.gz")
    eval_index = os.path.join(HERE, "eval_episodes.jsonl")
    n_train = n_eval = 0
    index_out = open(eval_index, "w")
    with gzip.open(train_path, "wt") as train_out:
        for run in RUNS:
            root = os.path.join(mirror, run)
            if not os.path.isdir(root):
                print("skip missing %s" % root)
                continue
            for name in FLAT:
                src = os.path.join(root, name)
                if os.path.exists(src):
                    shutil.copyfile(src, os.path.join(out_meta, "%s.%s" % (run, name)))
            src = os.path.join(root, "logs.log")
            if os.path.exists(src):
                # .log is gitignored; the .gz is both smaller and committable
                with open(src, "rb") as fh, gzip.open(
                    os.path.join(out_meta, "%s.logs.log.gz" % run), "wb"
                ) as gz:
                    shutil.copyfileobj(fh, gz)

            for d in sorted(os.listdir(root)):
                if not d.startswith("iteration_"):
                    continue
                it = int(d.split("_")[1])
                ev = os.path.join(root, d, "eval_test_rollout_summaries.jsonl")
                if os.path.exists(ev):
                    dst = os.path.join(out_eval, "%s-%06d.jsonl.gz" % (run, it))
                    with open(ev, "rb") as fh, gzip.open(dst, "wb") as gz:
                        shutil.copyfileobj(fh, gz)
                    for line in open(ev):
                        e = json.loads(line)
                        m = {}
                        for s in e["steps"]:
                            m.update(s.get("metrics") or {})
                        m.update(e.get("trajectory_metrics") or {})
                        index_out.write(json.dumps({
                            "run": run, "iter": it, "group_idx": e["group_idx"],
                            "turns": len(e["steps"]), "metrics": m}) + "\n")
                    n_eval += 1
                tr = os.path.join(root, d, "train_rollout_summaries.jsonl")
                if os.path.exists(tr):
                    for line in open(tr):
                        train_out.write(
                            json.dumps(slim_episode(json.loads(line), run, it, "train")) + "\n")
                        n_train += 1

    index_out.close()
    print("wrote %d eval files, %d slim train episodes" % (n_eval, n_train))
    for p in (out_meta, out_eval, train_path, eval_index):
        size = (os.path.getsize(p) if os.path.isfile(p)
                else sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)))
        print("  %-14s %6.1f MB" % (os.path.basename(p), size / 1048576))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/open-rl-runs/harvey-labs"))
