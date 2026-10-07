"""Telemetry recorded by the workers themselves. ops.py records every op
(forward_backward, optim_step, sample, wake_up, sleep, weight_sync, ...) and
the exclusive GPU turns those ops ran in, for the dashboard to read."""
