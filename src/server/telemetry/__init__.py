"""Telemetry: what OpenRL was doing, and what the hardware under it was doing.

  ops.py         Recorded by the workers themselves: every op (forward_backward,
                 optim_step, sample, wake_up, sleep, weight_sync, ...) and the
                 exclusive GPU turns those ops ran in. Only OpenRL can know this.
  kubernetes.py  Nodes, GPUs, pods, events, scheduler Workloads and DRA claims.
  prometheus.py  GPU (DCGM) and worker CPU/memory series.
  gke.py         GKE identity and Cloud Logging.

Workers import only ops.py; nothing here runs at import time.
"""
