# Text-SQL Autoresearch Recipe

This is the fast deterministic recipe for checking the autoresearch loop. It
uses the minimal recipe contract: `program.md` tells the agent to edit
`train.py`, and `autoresearch.toml` declares this recipe's command and metric.

```toml
command = "python -m rl.autoresearch.text_sql.train run_dir={run_dir} cache_root={log_root} attempt_timeout_minutes={attempt_timeout_minutes}"
editable = ["rl/autoresearch/text_sql/train.py"]
metric = "accuracy"
```

There is no model server for this recipe. `prepare.py` owns the fixed dataset
and scoring helpers; `train.py` is the editable runnable attempt.

Use the parent [autoresearch README](../README.md) for the common cluster run
flow. This page only covers text-SQL-specific run behavior and settings.

`train.py` samples a fixed Text-SQL dataset slice through `prepare.py`: 5,000
train examples and 50 held-out SQLite-executable scoring examples. It runs the
training loop, scores execution `accuracy`, and logs metrics with the cookbook
`ml_logger`. During training the editable code sees only the question, schema,
and its own prediction; it does not see the correct SQL or the held-out
execution result. `SELECT` queries compare result rows; mutation queries compare
the resulting database state.

## Local Run

From `examples`:

```bash
uv run python -m rl.autoresearch.run_attempt \
  recipe=rl/autoresearch/text_sql/autoresearch.toml \
  researcher=local-text-sql \
  name=baseline \
  log_root=artifacts/autoresearch/text_sql
```

Serve the UI for local artifacts:

```bash
uv run python -m rl.autoresearch.ui.observer \
  log_root=artifacts/autoresearch/text_sql \
  port=8080 \
  serve=True
```

Open:

```text
http://localhost:8080/experiments.html
```

Clear local text-SQL artifacts:

```bash
uv run python -m rl.autoresearch.run_attempt \
  clean=True \
  log_root=artifacts/autoresearch/text_sql
```

## Kubernetes Run

This assumes your cluster and shared storage already exist. From the repo root:

```bash
kubectl apply -k examples/rl/autoresearch/text_sql
kubectl port-forward svc/open-rl-autoresearch-ui 8080:8080
```

The recipe directory customizes the shared `k8s/base` resources through
Kustomize. There is no separate hand-written Kubernetes YAML for this recipe.

## Overlay Settings

The text-SQL overlay sets:

- `RECIPE=rl/autoresearch/text_sql/autoresearch.toml`
- `LOG_ROOT=/mnt/shared/open-rl/autoresearch/text_sql`
- `ATTEMPT_TIMEOUT_MINUTES=5`
- `AGENT_TIMEOUT_MINUTES=10`

The mounted `program.md` tells each researcher pod to edit `train.py`, run
`RUN_ATTEMPT_COMMAND`, keep concise notes, and keep only improved commits using
`accuracy`.
