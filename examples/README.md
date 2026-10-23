# OpenRL Examples

This directory contains examples, demos, and helper scripts for using the OpenRL framework. These are not part of the core library but serve as recipes for training and evaluation.

## Prerequisites

* **Install [uv](https://docs.astral.sh/uv/):** Follow the official installation guide to install the fast Python package manager.
* **Synchronize Dependencies:** Run the following command to set up the environment:
  ```bash
  cd examples
  uv sync
  ```

## Client commands

Run client tools from `examples/`; uv discovers this directory's
`pyproject.toml`, installs its declared commands, and keeps the environment
in sync automatically:

```bash
uv run harvey-train --help
uv run harvey-eval --help
uv run harvey-results log_dir=artifacts/harvey-labs/my-run json=True plot=True
uv run harvey-plot log_dir=artifacts/harvey-labs/my-run out=run.png
```

From the repository root, use `uv run --project examples <command>`.
Paths remain relative to your current directory. The root project is the
server runtime; the client has its own lockfile and CPU dependencies.
`[project.scripts]` declares CLI commands. Use `--locked` in CI to reject
lockfile drift.

Run the Harvey renderer regression test from the repository root with
`uv run --project examples python -m harvey_labs.renderers.qwen35_renderer`.
It uses standard `unittest` in the examples environment.

See [Harvey LAB RL](harvey_labs) for setup and recipe options.

## Custom loop telemetry

Custom loops using the cookbook logger can append one OpenTelemetry sink:

```python
from common.otel_logger import OTelLogger
from tinker_cookbook.utils import ml_log

session_id = service_client.holder.get_session_id()
logger = ml_log.setup_logging(log_dir=log_dir, config=config)
logger.loggers.append(OTelLogger(session_id=session_id, model_id=model_id))
try:
  # Existing metric calls continue to write files and now also export telemetry.
  logger.log_metrics({"phase": "train", "loss": loss}, step=step)
finally:
  logger.close()
```

`session_id` is the existing Tinker client session; `model_id` is optional.
With our pinned Tinker 0.29.0, `service_client.holder.get_session_id()` obtains
the session ID. This is a deprecated SDK compatibility accessor, used only in
the caller until Tinker exposes a public accessor; the adapter does not depend
on SDK internals. Untouched cookbook recipes do not enable client metric export.

The standard OTLP **HTTP/protobuf** exporters use the collector endpoint and
authentication configured through `OTEL_EXPORTER_OTLP_*`. For a collector
running locally or port-forwarded to port 4318:

```sh
export OTEL_SERVICE_NAME=openrl-client
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export OTEL_METRIC_EXPORT_INTERVAL=5000
```

Production uses the collector's protected HTTPS endpoint and configured headers;
clients send to the collector, never to the OpenRL API directly. This adapter selects HTTP
exporters explicitly, so setting the protocol to gRPC does not switch it.

Each `log_metrics` call emits an `openrl.training.metrics` log event with finite
numeric `metrics` and an optional `step`. Attributes are `session.id`, optional
`openrl.model.id`, and `openrl.phase` when supplied in the metric dictionary.
Repeated steps remain separate timestamped events. Gauges `openrl.client.metric`
(with a `metric` attribute for the original name) and `openrl.client.step` expose
the latest values per export interval for Prometheus; step is never a label.
These are sampled time series: several observations between exports can become
one gauge sample. Use the log events or original JSONL for per-call history.
Non-numeric values, booleans, NaN, and infinities are excluded. Configuration and
text remain with the existing cookbook loggers.

Each sink admits at most 500 gauge series over its lifetime. A metric name with
a different phase is a separate series, and each phase's step gauge also counts.
Existing series keep updating after the cap; new series are omitted from gauges
while their observations remain in log events. Set `max_series=` on `OTelLogger`
to change this budget. Events retain at most 500 numeric values per call, with
nonempty metric names of at most 200 characters; larger records keep the first
500 valid names in input order. Counts appear as `dropped_gauge_updates` or
`dropped_metrics` in affected log events, and each limit produces one warning
per sink. The existing cookbook JSONL logger continues to receive the original
records.

OTel owns buffering, retries, and shutdown. Delivery is best effort; keep the
cookbook JSONL files for recovery. The adapter does not install global OTel
providers. The collector in `k8s/deploy/telemetry` forwards the log events to
the API server, which keeps one row per step for the run's Metrics tab, and
exports the gauges to Prometheus for runs without forwarded events; see
[client metrics by step](../docs/dashboard.md#client-metrics-by-step). Names
are discovered from the recorded metrics; custom metrics need no frontend
registration. See the
[OTel exporters](https://opentelemetry.io/docs/languages/python/exporters/) and
[Google-built collector](https://docs.cloud.google.com/stackdriver/docs/instrumentation/opentelemetry-collector-gke)
for backend setup. Run its local export regression checks with
`make test-client-telemetry` from the repository root.

---

## Examples Overview

### Supervised Fine-Tuning (SFT)
* **[Pig Latin Translation](sft/pig-latin):** Teaches a model to perform specialized Pig Latin transformations, demonstrating custom token-level targets and loss masks.
* **[Text-to-SQL SFT](sft/text-to-sql):** Adapts Gemma 3 into a specialized database query assistant capable of generating SQL statements.
* **[FunctionGemma](sft/function-gemma):** Provides a recipe specifically targeted at fine-tuning tool-use capabilities, enabling models to reliably select and invoke functions.

### Reinforcement Learning (RL)
* **[Text-to-SQL RL](text-to-sql):** Runs the Gemma 4 SFT+RL recipe with SQL execution rewards and curve plotting.

### Autoresearch
* **[Autoresearch Demo](autoresearch):** Runs code-RL researchers against the same OpenRL API server using cookbook DeepCoder rewards, Sandbox Fusion, and optional Agent Sandbox CRDs.

### Tinker Cookbook
* **[Tinker Cookbook Recipes](tinker-cookbook):** Examples showing how to run [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) recipes with OpenRL.

---
