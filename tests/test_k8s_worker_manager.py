import os
import tempfile
import types
import unittest
from unittest.mock import patch

from server.k8s_worker_manager import (
  WORKER_INSTANCE_ANNOTATION,
  WORKER_MODEL_ANNOTATION,
  WORKER_REVISION_ANNOTATION,
  WORKER_SOURCE_ANNOTATION,
  KubernetesFFTWorkerManager,
  node_summary,
  pod_summary,
  sanitize_job_id,
)
from server.worker_manager import FFTWorkerManager, create_fft_worker_manager

POD_TEMPLATE = """\
apiVersion: v1
kind: Pod
spec:
  restartPolicy: OnFailure
  containers:
  - name: trainer-worker
    image: example/server:latest
    command: ["python", "-m", "server.training_requests_processor"]
    env:
    - name: REDIS_URL
      value: "redis://redis-service:6379"
"""


class _ApiError(Exception):
  def __init__(self, status: int):
    super().__init__(f"api error {status}")
    self.status = status


class _FakeCoreApi:
  def __init__(self, pod_phases: dict[str, str] | None = None):
    self.pod_phases = pod_phases or {}
    self.pod_annotations: dict[str, dict[str, str]] = {}
    self.created: list[tuple[str, dict]] = []
    self.deleted: list[str] = []
    self.create_error: Exception | None = None
    self.listed_namespaces: list[tuple[str, int | None]] = []
    self.pods_by_selector: dict[str, list[object]] = {}
    self.log_reads: list[dict[str, object]] = []
    self.events: list[object] = []

  def read_namespaced_pod(self, name: str, namespace: str):
    if name not in self.pod_phases:
      raise _ApiError(404)
    return types.SimpleNamespace(
      metadata=types.SimpleNamespace(annotations=self.pod_annotations.get(name, {})),
      status=types.SimpleNamespace(phase=self.pod_phases[name]),
    )

  def create_namespaced_pod(self, namespace: str, body: dict):
    if self.create_error is not None:
      if self.create_error.status == 409:
        name = body["metadata"]["name"]
        self.pod_phases[name] = "Running"
        self.pod_annotations[name] = body["metadata"]["annotations"]
      raise self.create_error
    self.created.append((namespace, body))
    name = body["metadata"]["name"]
    self.pod_phases[name] = "Running"
    self.pod_annotations[name] = body["metadata"]["annotations"]

  def delete_namespaced_pod(self, name: str, namespace: str):
    self.deleted.append(name)
    self.pod_phases.pop(name, None)
    self.pod_annotations.pop(name, None)

  def list_node(self):
    node = types.SimpleNamespace(
      metadata=types.SimpleNamespace(name="kind-worker", labels={}),
      spec=types.SimpleNamespace(taints=[]),
      status=types.SimpleNamespace(
        conditions=[types.SimpleNamespace(type="Ready", status="True", reason=None, message=None)],
        capacity={},
        allocatable={},
        addresses=[],
      ),
    )
    return types.SimpleNamespace(items=[node])

  def list_namespaced_pod(self, namespace: str, limit: int | None = None, label_selector: str | None = None):
    self.listed_namespaces.append((namespace, limit))
    return types.SimpleNamespace(items=self.pods_by_selector.get(label_selector or "", []))

  def read_namespaced_pod_log(self, **kwargs):
    self.log_reads.append(kwargs)
    return "2026-07-10T12:00:00Z component started\n"

  def list_namespaced_event(self, namespace: str, limit: int | None = None):
    return types.SimpleNamespace(items=self.events[:limit])

  def list_pod_for_all_namespaces(self):
    raise AssertionError("control-plane inventory must not inspect unrelated namespaces")


class KubernetesFFTWorkerManagerTest(unittest.TestCase):
  def setUp(self) -> None:
    self.template_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    self.template_file.write(POD_TEMPLATE)
    self.template_file.close()
    self.addCleanup(os.unlink, self.template_file.name)
    self.env = {
      "REDIS_URL": "redis://redis-service:6379",
      "OPEN_RL_WORKER_POD_TEMPLATE": self.template_file.name,
      "OPEN_RL_WORKER_NAMESPACE": "training",
    }

  def _manager(self, core_api: _FakeCoreApi) -> KubernetesFFTWorkerManager:
    with patch.dict(os.environ, self.env, clear=True):
      return KubernetesFFTWorkerManager(core_api=core_api)

  def manager(self, core_api: _FakeCoreApi) -> KubernetesFFTWorkerManager:
    return self._manager(core_api)

  def test_launch_stamps_name_labels_args_and_job_id_env(self) -> None:
    api = _FakeCoreApi()
    self._manager(api).launch("Model_A.1")

    self.assertEqual(len(api.created), 1)
    namespace, body = api.created[0]
    self.assertEqual(namespace, "training")
    self.assertEqual(body["metadata"]["name"], "open-rl-trainer-model-a-1")
    self.assertEqual(
      body["metadata"]["labels"],
      {
        "app": "open-rl-trainer-worker",
        "accel-timeslicer": "true",
        "timeslice.io/group": "trainers",
        "timeslice.io/job-id": "trainer-model-a-1",
      },
    )
    container = body["spec"]["containers"][0]
    self.assertEqual(container["args"], ["model_id=Model_A.1"])
    self.assertIn({"name": "OPEN_RL_TIME_SLICE_JOB_ID", "value": "trainer-model-a-1"}, container["env"])
    self.assertIn({"name": "OPEN_RL_TIME_SLICE_GROUP", "value": "trainers"}, container["env"])

  def test_launch_replaces_stale_job_id_env_from_template(self) -> None:
    api = _FakeCoreApi()
    manager = self._manager(api)
    manager.pod_template["spec"]["containers"][0]["env"].append({"name": "OPEN_RL_TIME_SLICE_JOB_ID", "value": "stale-job"})
    manager.pod_template["spec"]["containers"][0]["env"].append({"name": "OPEN_RL_TIME_SLICE_GROUP", "value": "stale-group"})

    manager.launch("Model_A.1")

    container = api.created[0][1]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}
    self.assertEqual(env["OPEN_RL_TIME_SLICE_JOB_ID"], "trainer-model-a-1")
    self.assertEqual(env["OPEN_RL_TIME_SLICE_GROUP"], "trainers")

  def test_launch_sampler_stamps_sampler_identity(self) -> None:
    api = _FakeCoreApi()
    self._manager(api).launch_sampler("Model_A.1")

    self.assertEqual(len(api.created), 1)
    _, body = api.created[0]
    self.assertEqual(body["metadata"]["name"], "open-rl-sampler-model-a-1")
    self.assertEqual(
      body["metadata"]["labels"],
      {
        "app": "open-rl-sampler-worker",
        "accel-timeslicer": "true",
        "timeslice.io/group": "samplers",
        "timeslice.io/job-id": "sampler-model-a-1",
      },
    )
    container = body["spec"]["containers"][0]
    self.assertEqual(container["command"], ["uv", "run", "--no-sync", "python", "-u", "-m", "server.vllm_sampler"])
    self.assertEqual(container["args"], ["model_id=Model_A.1"])
    self.assertIn({"name": "OPEN_RL_TIME_SLICE_JOB_ID", "value": "sampler-model-a-1"}, container["env"])
    self.assertIn({"name": "OPEN_RL_TIME_SLICE_GROUP", "value": "samplers"}, container["env"])

  def test_source_deploy_revision_is_injected_into_new_workers(self) -> None:
    api = _FakeCoreApi()
    env = {
      **self.env,
      "OPEN_RL_SOURCE_REVISION": "abc123",
      "PYTHONPATH": "/mnt/shared/open-rl/source/abc123",
    }

    with patch.dict(os.environ, env, clear=True):
      KubernetesFFTWorkerManager(core_api=api).launch("model-a")

    body = api.created[0][1]
    self.assertEqual(body["metadata"]["annotations"][WORKER_SOURCE_ANNOTATION], "abc123")
    worker_env = {item["name"]: item.get("value") for item in body["spec"]["containers"][0]["env"]}
    self.assertEqual(worker_env["PYTHONPATH"], "/mnt/shared/open-rl/source/abc123")
    self.assertEqual(worker_env["OPEN_RL_SOURCE_REVISION"], "abc123")

  def test_launch_is_idempotent_while_pod_is_live(self) -> None:
    api = _FakeCoreApi(pod_phases={"open-rl-trainer-model-a": "Running"})
    manager = self._manager(api)
    api.pod_annotations["open-rl-trainer-model-a"] = {
      WORKER_INSTANCE_ANNOTATION: "instance-a",
      WORKER_REVISION_ANNOTATION: manager.worker_revision("trainer"),
    }

    instance_id = manager.launch("model-a")

    self.assertEqual(instance_id, "instance-a")
    self.assertEqual(api.created, [])
    self.assertEqual(api.deleted, [])

  def test_launch_replaces_live_pod_from_old_revision(self) -> None:
    api = _FakeCoreApi(pod_phases={"open-rl-trainer-model-a": "Running"})
    api.pod_annotations["open-rl-trainer-model-a"] = {
      WORKER_INSTANCE_ANNOTATION: "instance-a",
      WORKER_REVISION_ANNOTATION: "old-revision",
    }

    instance_id = self._manager(api).launch("model-a")

    self.assertNotEqual(instance_id, "instance-a")
    self.assertEqual(api.deleted, ["open-rl-trainer-model-a"])
    self.assertEqual(len(api.created), 1)

  def test_launch_replaces_terminal_pod(self) -> None:
    api = _FakeCoreApi(pod_phases={"open-rl-trainer-model-a": "Failed"})
    self._manager(api).launch("model-a")

    self.assertEqual(api.deleted, ["open-rl-trainer-model-a"])
    self.assertEqual(len(api.created), 1)

  def test_launch_tolerates_conflict_on_create(self) -> None:
    api = _FakeCoreApi()
    api.create_error = _ApiError(409)
    self._manager(api).launch("model-a")  # must not raise

  def test_launch_raises_on_other_api_errors(self) -> None:
    api = _FakeCoreApi()
    api.create_error = _ApiError(403)
    with self.assertRaises(_ApiError):
      self._manager(api).launch("model-a")

  def test_launch_queries_model_metadata_for_pod_env(self) -> None:
    import json

    from server.store import InMemoryStore

    s = InMemoryStore()
    s.kv_store["open_rl:model_meta:Model_A.1"] = json.dumps(
      {
        "base_model": "Qwen/Qwen2.5-0.5B",
        "weight_sync_strategy": "full",
        "training_kind": "full",
      }
    )
    api = _FakeCoreApi()

    with patch("server.store.get_store", return_value=s):
      manager = self.manager(api)
      manager.launch("Model_A.1")
      manager.launch_sampler("Model_A.1")

    trainer_container = api.created[0][1]["spec"]["containers"][0]
    trainer_env = {item["name"]: item["value"] for item in trainer_container["env"] if "value" in item}
    self.assertEqual(trainer_env.get("BASE_MODEL"), "Qwen/Qwen2.5-0.5B")
    self.assertEqual(trainer_env.get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "full")

    sampler_container = api.created[1][1]["spec"]["containers"][0]
    sampler_env = {item["name"]: item["value"] for item in sampler_container["env"] if "value" in item}
    self.assertEqual(sampler_env.get("BASE_MODEL"), "Qwen/Qwen2.5-0.5B")
    self.assertEqual(sampler_env.get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "full")

  def test_requires_template_and_redis(self) -> None:
    with patch.dict(os.environ, {"REDIS_URL": "redis://r:6379"}, clear=True), self.assertRaisesRegex(RuntimeError, "POD_TEMPLATE"):
      KubernetesFFTWorkerManager(core_api=_FakeCoreApi())
    with (
      patch.dict(os.environ, {"OPEN_RL_WORKER_POD_TEMPLATE": self.template_file.name}, clear=True),
      self.assertRaisesRegex(RuntimeError, "REDIS_URL"),
    ):
      KubernetesFFTWorkerManager(core_api=_FakeCoreApi())

  def test_sanitize_job_id(self) -> None:
    self.assertEqual(sanitize_job_id("Model_A.1"), "model-a-1")
    self.assertEqual(sanitize_job_id("a" * 80), "a" * 63)
    with self.assertRaises(ValueError):
      sanitize_job_id("___")

  def test_pod_summary_surfaces_waiting_reason_and_restarts(self) -> None:
    waiting = types.SimpleNamespace(reason="ImagePullBackOff", message="pull failed")
    state = types.SimpleNamespace(waiting=waiting, terminated=None)
    container_status = types.SimpleNamespace(ready=False, restart_count=3, state=state)
    pod = types.SimpleNamespace(
      metadata=types.SimpleNamespace(
        name="open-rl-sampler-run-a",
        namespace="training",
        labels={"app": "open-rl-sampler-worker"},
        annotations={WORKER_MODEL_ANNOTATION: "run-a"},
      ),
      spec=types.SimpleNamespace(node_name="kind-worker", containers=[types.SimpleNamespace(image="open-rl:test")]),
      status=types.SimpleNamespace(phase="Pending", container_statuses=[container_status], reason=None, message=None, start_time=None),
    )

    summary = pod_summary(pod)

    self.assertEqual(summary["status"], "failed")
    self.assertEqual(summary["reason"], "ImagePullBackOff")
    self.assertEqual(summary["restarts"], 3)
    self.assertEqual(summary["model_id"], "run-a")

  def test_pod_summary_surfaces_unschedulable_condition(self) -> None:
    condition = types.SimpleNamespace(
      type="PodScheduled",
      status="False",
      reason="Unschedulable",
      message="0/2 nodes are available: insufficient cpu",
    )
    pod = types.SimpleNamespace(
      metadata=types.SimpleNamespace(name="open-rl-trainer-run-a", namespace="training", labels={}, annotations={}),
      spec=types.SimpleNamespace(node_name=None, containers=[]),
      status=types.SimpleNamespace(
        phase="Pending",
        container_statuses=[],
        conditions=[condition],
        reason=None,
        message=None,
        start_time=None,
      ),
    )

    summary = pod_summary(pod)

    self.assertEqual(summary["status"], "pending")
    self.assertEqual(summary["reason"], "Unschedulable")
    self.assertIn("insufficient cpu", summary["message"])

  def test_pod_summary_treats_oom_killed_as_failed(self) -> None:
    terminated = types.SimpleNamespace(reason="OOMKilled", message="memory limit exceeded")
    state = types.SimpleNamespace(waiting=None, terminated=terminated)
    container_status = types.SimpleNamespace(ready=False, restart_count=1, state=state)
    pod = types.SimpleNamespace(
      metadata=types.SimpleNamespace(name="open-rl-trainer-run-a", namespace="training", labels={}, annotations={}),
      spec=types.SimpleNamespace(node_name="kind-worker", containers=[]),
      status=types.SimpleNamespace(
        phase="Running",
        container_statuses=[container_status],
        conditions=[],
        reason=None,
        message=None,
        start_time=None,
      ),
    )

    summary = pod_summary(pod)

    self.assertEqual(summary["status"], "failed")
    self.assertEqual(summary["reason"], "OOMKilled")

  def test_pod_summary_treats_nonzero_terminated_container_as_failed(self) -> None:
    terminated = types.SimpleNamespace(reason="Error", message="process exited", exit_code=17)
    state = types.SimpleNamespace(waiting=None, terminated=terminated)
    container_status = types.SimpleNamespace(ready=False, restart_count=2, state=state)
    pod = types.SimpleNamespace(
      metadata=types.SimpleNamespace(name="open-rl-trainer-run-a", namespace="training", labels={}, annotations={}),
      spec=types.SimpleNamespace(node_name="kind-worker", containers=[]),
      status=types.SimpleNamespace(
        phase="Running",
        container_statuses=[container_status],
        init_container_statuses=[],
        conditions=[],
        reason=None,
        message=None,
        start_time=None,
      ),
    )

    summary = pod_summary(pod)

    self.assertEqual(summary["status"], "failed")
    self.assertEqual(summary["reason"], "Error")
    self.assertEqual(summary["restarts"], 2)

  def test_pod_summary_surfaces_init_container_image_pull_and_restarts(self) -> None:
    waiting = types.SimpleNamespace(reason="ImagePullBackOff", message="init image could not be pulled")
    init_state = types.SimpleNamespace(waiting=waiting, terminated=None)
    init_status = types.SimpleNamespace(ready=False, restart_count=4, state=init_state)
    pod = types.SimpleNamespace(
      metadata=types.SimpleNamespace(name="open-rl-trainer-run-a", namespace="training", labels={}, annotations={}),
      spec=types.SimpleNamespace(node_name="kind-worker", containers=[]),
      status=types.SimpleNamespace(
        phase="Pending",
        container_statuses=[],
        init_container_statuses=[init_status],
        conditions=[],
        reason=None,
        message=None,
        start_time=None,
      ),
    )

    summary = pod_summary(pod)

    self.assertEqual(summary["status"], "failed")
    self.assertEqual(summary["reason"], "ImagePullBackOff")
    self.assertIn("init image", summary["message"])
    self.assertEqual(summary["restarts"], 4)
    self.assertFalse(summary["ready"])

  def test_completed_init_container_does_not_override_regular_readiness(self) -> None:
    completed = types.SimpleNamespace(reason="Completed", message=None, exit_code=0)
    init_status = types.SimpleNamespace(
      ready=False,
      restart_count=1,
      state=types.SimpleNamespace(waiting=None, terminated=completed),
    )
    container_status = types.SimpleNamespace(
      ready=True,
      restart_count=0,
      state=types.SimpleNamespace(waiting=None, terminated=None),
    )
    pod = types.SimpleNamespace(
      metadata=types.SimpleNamespace(name="open-rl-trainer-run-a", namespace="training", labels={}, annotations={}),
      spec=types.SimpleNamespace(node_name="kind-worker", containers=[]),
      status=types.SimpleNamespace(
        phase="Running",
        container_statuses=[container_status],
        init_container_statuses=[init_status],
        conditions=[],
        reason=None,
        message=None,
        start_time=None,
      ),
    )

    summary = pod_summary(pod)

    self.assertEqual(summary["status"], "ready")
    self.assertTrue(summary["ready"])
    self.assertEqual(summary["restarts"], 1)

  def test_node_summary_includes_gpu_capacity_and_readiness(self) -> None:
    node = types.SimpleNamespace(
      metadata=types.SimpleNamespace(name="kind-worker", labels={"node-role.kubernetes.io/worker": ""}),
      spec=types.SimpleNamespace(taints=[]),
      status=types.SimpleNamespace(
        conditions=[types.SimpleNamespace(type="Ready", status="True", reason=None, message=None)],
        capacity={"cpu": "8", "nvidia.com/gpu": "1"},
        allocatable={"cpu": "7", "nvidia.com/gpu": "1"},
        addresses=[types.SimpleNamespace(type="InternalIP", address="10.0.0.2")],
      ),
    )

    summary = node_summary(node)

    self.assertTrue(summary["ready"])
    self.assertEqual(summary["capacity"]["nvidia.com/gpu"], "1")
    self.assertEqual(summary["internal_ip"], "10.0.0.2")

  def test_cluster_snapshot_includes_kubernetes_events(self) -> None:
    api = _FakeCoreApi()
    api.events = [
      types.SimpleNamespace(
        metadata=types.SimpleNamespace(name="pod-a.123", creation_timestamp=None),
        involved_object=types.SimpleNamespace(kind="Pod", name="pod-a"),
        type="Warning",
        reason="FailedScheduling",
        message="Insufficient nvidia.com/gpu",
        count=4,
        first_timestamp=None,
        last_timestamp=None,
      )
    ]

    snapshot = self.manager(api).cluster_snapshot()

    self.assertEqual(snapshot["events"][0]["type"], "warning")
    self.assertEqual(snapshot["events"][0]["object_name"], "pod-a")
    self.assertEqual(snapshot["events"][0]["count"], 4)

  def test_cluster_snapshot_reads_time_slicer_queue_on_gpu_pool_nodes(self) -> None:
    api = _FakeCoreApi()
    node = api.list_node().items[0]
    node.metadata.labels = {"group.timeslice.io/trainers": "true"}
    node.status.addresses = [types.SimpleNamespace(type="InternalIP", address="10.0.0.2")]
    api.list_node = lambda: types.SimpleNamespace(items=[node])
    time_slicer = {"ok": True, "active_workload": "trainers:a", "waiting_workloads": ["trainers:b"], "workloads": []}

    with patch("server.k8s_worker_manager.read_time_slicer_status", return_value=time_slicer) as read_status:
      snapshot = self.manager(api).cluster_snapshot()

    read_status.assert_called_once_with("10.0.0.2")
    self.assertEqual(snapshot["nodes"][0]["time_slicer"], time_slicer)

  def test_cluster_snapshot_only_lists_configured_namespace(self) -> None:
    api = _FakeCoreApi()

    snapshot = self.manager(api).cluster_snapshot()

    self.assertEqual(api.listed_namespaces, [("training", 500)])
    self.assertEqual(snapshot["namespace"], "training")
    self.assertEqual(snapshot["status"], "healthy")

  def test_cluster_snapshot_is_degraded_for_actionable_pending_pod(self) -> None:
    api = _FakeCoreApi()
    unschedulable = types.SimpleNamespace(
      type="PodScheduled",
      status="False",
      reason="Unschedulable",
      message="0/1 nodes are available: insufficient memory",
    )
    api.pods_by_selector[""] = [
      types.SimpleNamespace(
        metadata=types.SimpleNamespace(name="open-rl-trainer-run-a", namespace="training", labels={}, annotations={}),
        spec=types.SimpleNamespace(node_name=None, containers=[]),
        status=types.SimpleNamespace(
          phase="Pending",
          container_statuses=[],
          init_container_statuses=[],
          conditions=[unschedulable],
          reason=None,
          message=None,
          start_time=None,
        ),
      )
    ]

    snapshot = self.manager(api).cluster_snapshot()

    self.assertEqual(snapshot["status"], "degraded")
    self.assertEqual(snapshot["summary"]["pending_pods"], 1)
    self.assertEqual(snapshot["summary"]["actionable_pending_pods"], 1)
    self.assertEqual(snapshot["pods"][0]["reason"], "Unschedulable")

  def test_gateway_logs_select_first_namespaced_gateway_pod(self) -> None:
    api = _FakeCoreApi()
    api.pods_by_selector["app=open-rl-gateway"] = [
      types.SimpleNamespace(metadata=types.SimpleNamespace(name="gateway-a")),
      types.SimpleNamespace(metadata=types.SimpleNamespace(name="gateway-b")),
    ]

    result = self.manager(api).read_logs("run-a", "gateway", tail_lines=25, previous=True)

    self.assertEqual(result["pod_name"], "gateway-a")
    self.assertIn("component started", result["logs"])
    self.assertEqual(api.listed_namespaces, [("training", 1)])
    self.assertEqual(
      api.log_reads,
      [
        {
          "name": "gateway-a",
          "namespace": "training",
          "tail_lines": 25,
          "timestamps": True,
          "previous": True,
        }
      ],
    )

  def test_timeslicer_logs_use_namespaced_app_selector(self) -> None:
    api = _FakeCoreApi()
    api.pods_by_selector["app=open-rl-accel-timeslicer"] = [types.SimpleNamespace(metadata=types.SimpleNamespace(name="timeslicer-a"))]

    result = self.manager(api).read_logs("run-a", "timeslicer")

    self.assertEqual(result["pod_name"], "timeslicer-a")
    self.assertEqual(api.listed_namespaces, [("training", 1)])

  def test_pod_logs_read_exact_name_in_manager_namespace(self) -> None:
    api = _FakeCoreApi()

    result = self.manager(api).read_pod_logs("open-rl-client-job-abc", tail_lines=25, previous=True)

    self.assertEqual(result["pod_name"], "open-rl-client-job-abc")
    self.assertIn("component started", result["logs"])
    self.assertEqual(api.listed_namespaces, [])
    self.assertEqual(
      api.log_reads,
      [
        {
          "name": "open-rl-client-job-abc",
          "namespace": "training",
          "tail_lines": 25,
          "timestamps": True,
          "previous": True,
        }
      ],
    )

  def test_pod_logs_clamp_tail_bound(self) -> None:
    api = _FakeCoreApi()
    manager = self.manager(api)

    manager.read_pod_logs("client-a", tail_lines=0)
    manager.read_pod_logs("client-b", tail_lines=6000)

    self.assertEqual([read["tail_lines"] for read in api.log_reads], [1, 5000])


class CreateFFTWorkerManagerTest(unittest.TestCase):
  def test_default_launcher_is_subprocess(self) -> None:
    with patch.dict(os.environ, {"REDIS_URL": "redis://r:6379"}, clear=True):
      manager = create_fft_worker_manager()
    self.assertIsInstance(manager, FFTWorkerManager)

  def test_kubernetes_launcher_is_selected_by_env(self) -> None:
    env = {"REDIS_URL": "redis://r:6379", "OPEN_RL_WORKER_MANAGER": "kubernetes"}
    with (
      patch.dict(os.environ, env, clear=True),
      patch("server.k8s_worker_manager.KubernetesFFTWorkerManager") as manager_cls,
    ):
      manager = create_fft_worker_manager()
    self.assertIs(manager, manager_cls.return_value)


if __name__ == "__main__":
  unittest.main()
