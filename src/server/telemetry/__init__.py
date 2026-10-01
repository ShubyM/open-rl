"""Telemetry: what OpenRL was doing, and what the hardware under it was doing.

  ops.py         Recorded by the workers themselves: every op (forward_backward,
                 optim_step, sample, wake_up, sleep, weight_sync, ...) and the
                 exclusive GPU turns those ops ran in. Only OpenRL can know this.
  backends.py    Everything else, read from the platform. One of two backends
                 with the same interface is picked per process:
                   Cluster  Kubernetes inventory, Prometheus hardware series
                            (DCGM GPUs, GKE container CPU/memory), Cloud Logging.
                   Host     one machine: nvidia-smi + psutil, worker log files.
  kubernetes.py, prometheus.py, gke.py, local.py   the clients behind them.

Workers import only ops.py; nothing here runs at import time.
"""
