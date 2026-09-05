# Consuming Harvey run results

Use the recipe's canonical results command before analyzing a run:

```bash
uv --project examples run --no-sync python -m harvey_labs.results <run-dir> --json
```

Omit `--json` for a human-readable summary. Add `--metrics` to the JSON command
to include every original metric (timings, entropy, grading errors, etc.).
Generate the standard plot with:

```bash
uv --project examples run --no-sync python -m harvey_labs.plot_run <run-dir>
```

Training automatically writes `results.json` and `run_plot.png` after a run
(also after a failure when metrics exist). For a running job, invoke the
results command to read current data; saved reports are snapshots.

Use `results.read_results` when adding analysis or plots. Extend this reader
and the existing commands when information is missing, so people and agents
use the same interpretation of steps, metric namespaces, and pooled scores.
Do not create another parser or plotting script for metrics already supported.

The report's step unit is completed training batches. Stored cookbook batch
indices are zero-based: training batch 0 finishes at step 1, while its
pre-update evaluation measures step 0. A final evaluation uses the checkpoint's
completed-batch count. Streaming log item ordinals are displayed one-based;
`Minibatch 8/8: Will train` identifies the item being started, not completion.
