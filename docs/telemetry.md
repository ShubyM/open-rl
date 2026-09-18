# Training and sampling telemetry

Set `ENABLE_GCP_TRACE=1` in each process that should export to Google Cloud Trace.
The API server, standalone trainer, and sampler initialize tracing at startup
with service names `open-rl-api-server`, `open-rl-trainer`, and `open-rl-sampler`.
Single-process training uses the API server's provider. Each process shuts down
its exporter on exit, including the workers' explicit process-exit paths.

Queue submission registers the pending result and injects the active trace
context. Consumers create a span per command, with `request_id` and `model_id`
attributes. A command without a trace carrier starts a new trace. Exceptions
are recorded before conversion into failed-result responses. No tracing
arguments are required by the worker's training methods.

Trainer commands contain spans for `prepare_batch`, `forward`, `loss`, and
`backward`. Forward-only requests omit backward. Optimizer and checkpoint
operations are timed by their command spans. Sampling has a request span,
a weight-reload span (including the reload-lock wait), and a generation span
with the output-token count. These do not yet split vLLM prefill from decode,
or measure time to first token. Consumer spans start after queue retrieval
and accelerator admission; they do not include that waiting time.

Forward/backward results also expose metrics, even with trace export disabled:

- `time/<phase>_host:sum`: host elapsed seconds summed across microbatches.
- `time/<phase>_gpu:sum`: CUDA stream elapsed seconds, present on CUDA only.

Host timings measure execution/submission on the CPU, not completion of all
GPU work. CUDA events surround each phase on the current training stream;
their durations are read after the existing `loss.item()` synchronization.
No extra synchronization is added. These are stream elapsed times, not sums
of individual kernel durations or device-wide utilization. Activation
checkpoint recomputation is included in backward. CPU and MPS runs report
only host timings.
