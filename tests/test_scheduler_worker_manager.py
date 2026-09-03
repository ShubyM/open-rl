import json
import os
import unittest
from typing import Any
from unittest.mock import patch

from server.estimator import footprint
from server.scheduler_worker_manager import GROUP, PLURAL, VERSION, SchedulerWorkerManager
from server.store import InMemoryStore


class _ApiError(Exception):
  def __init__(self, status: int):
    super().__init__(f"api error {status}")
    self.status = status


class _FakeCustomObjectsApi:
  def __init__(self):
    self.created: list[dict[str, Any]] = []
    self.deleted: list[str] = []
    self.existing: set[str] = set()

  def create_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str, body: dict) -> dict:
    assert (group, version, plural) == (GROUP, VERSION, PLURAL)
    name = body["metadata"]["name"]
    if name in self.existing:
      raise _ApiError(409)
    self.existing.add(name)
    self.created.append(body)
    return body

  def delete_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str, name: str) -> dict:
    if name not in self.existing:
      raise _ApiError(404)
    self.existing.remove(name)
    self.deleted.append(name)
    return {}

  def list_namespaced_custom_object(self, group: str, version: str, namespace: str, plural: str) -> dict:
    return {"items": [{"metadata": {"name": name}} for name in sorted(self.existing)]}


class SchedulerWorkerManagerTest(unittest.TestCase):
  def setUp(self) -> None:
    self._env = patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}, clear=False)
    self._env.start()
    self.addCleanup(self._env.stop)
    self.api = _FakeCustomObjectsApi()
    self.manager = SchedulerWorkerManager(custom_api=self.api)

  def _store_with(self, model_id: str, meta: dict) -> InMemoryStore:
    s = InMemoryStore()
    s.kv_store[f"open_rl:model_meta:{model_id}"] = json.dumps(meta)
    return s

  def test_lora_trainer_and_sampler_share_an_owner(self) -> None:
    s = self._store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.store.get_store", return_value=s):
      self.manager.ensure("job-lora-1", "trainer")
      self.manager.ensure("job-lora-1", "sampler")

    trainer, sampler = self.api.created
    self.assertEqual(trainer["metadata"]["name"], "lora-qwen-qwen2-5-0-5b-0-trainer")
    self.assertEqual(sampler["metadata"]["name"], "lora-qwen-qwen2-5-0-5b-0-sampler")
    # Same owner: one turn for the pair, and they hold the devices together.
    self.assertEqual(trainer["spec"]["ownerID"], "qwen-qwen2-5-0-5b")
    self.assertEqual(trainer["spec"]["ownerID"], sampler["spec"]["ownerID"])
    self.assertEqual(trainer["spec"]["trainingKind"], "lora")
    self.assertEqual(trainer["spec"]["accelerator"], {"mode": "SingleGPU", "memory": footprint("Qwen/Qwen2.5-0.5B", "lora", "trainer").accelerator})
    t_container = trainer["spec"]["template"]["spec"]["containers"][0]
    s_container = sampler["spec"]["template"]["spec"]["containers"][0]
    self.assertEqual(t_container["command"][-1], "server.training_requests_processor")
    self.assertEqual(s_container["command"][-1], "server.lora_sampler")
    self.assertIn("--active-tenant-set-id", t_container["args"])

  def test_fft_worker_is_its_own_owner(self) -> None:
    s = self._store_with("Model_A.1", {"base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "full"})
    with patch("server.store.get_store", return_value=s):
      self.manager.ensure("Model_A.1", "trainer")

    (worker,) = self.api.created
    self.assertEqual(worker["spec"]["role"], "trainer")
    # An FFT job is its own owner: its trainer (and sampler, if launched)
    # share one turn, and no other job ever matches this ID.
    self.assertEqual(worker["metadata"]["name"], "fft-model-a-1-trainer")
    self.assertEqual(worker["spec"]["ownerID"], "model-a-1")
    self.assertEqual(worker["spec"]["trainingKind"], "fft")
    self.assertEqual(worker["spec"]["accelerator"]["memory"], footprint("Qwen/Qwen3-8B", "full", "trainer").accelerator)
    container = worker["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e.get("value") for e in container["env"]}
    self.assertEqual(env["OPEN_RL_ENABLE_FFT"], "true")
    self.assertEqual(env["OPEN_RL_FINE_TUNING_TYPE"], "full")
    self.assertEqual(env["OPEN_RL_WORKLOAD_ID"], worker["metadata"]["name"])

  def test_launch_is_idempotent(self) -> None:
    s = self._store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.store.get_store", return_value=s):
      self.manager.ensure("job-lora-1", "trainer")
      self.manager.ensure("job-lora-1", "trainer")
    self.assertEqual(len(self.api.created), 1)

  def test_placement_knowledge_stays_out_of_the_template(self) -> None:
    s = self._store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.store.get_store", return_value=s):
      self.manager.ensure("job-lora-1", "trainer")

    (worker,) = self.api.created
    template_spec = worker["spec"]["template"]["spec"]
    env_names = {e["name"] for e in template_spec["containers"][0]["env"]}
    # The group is the claim name: placement's output, stamped by the
    # controller. Node selection and claims likewise never appear here --
    # the controller rejects a template that carries them.
    self.assertNotIn("OPEN_RL_TIME_SLICE_GROUP", env_names)
    self.assertNotIn("nodeSelector", template_spec)
    self.assertNotIn("nodeName", template_spec)
    self.assertNotIn("affinity", template_spec)
    self.assertNotIn("resourceClaims", template_spec)

  def test_release_deletes_an_fft_jobs_workloads_and_tolerates_absence(self) -> None:
    s = self._store_with("Model_A.1", {"base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "full"})
    with patch("server.store.get_store", return_value=s):
      self.manager.ensure("Model_A.1", "trainer")
      self.manager.release("Model_A.1")
      self.manager.release("Model_A.1")

    self.assertEqual(self.api.deleted, ["fft-model-a-1-trainer"])

  def test_release_leaves_a_shared_lora_runtime_alone(self) -> None:
    s = self._store_with("job-lora-1", {"base_model": "Qwen/Qwen2.5-0.5B", "fine_tuning_type": "lora"})
    with patch("server.store.get_store", return_value=s):
      self.manager.ensure("job-lora-1", "trainer")
      self.manager.release("job-lora-1")

    self.assertEqual(self.api.deleted, [])

  def test_create_worker_manager_selects_scheduler_mode(self) -> None:
    from server.worker_manager import create_worker_manager

    with (
      patch.dict(os.environ, {"OPEN_RL_WORKER_MANAGER": "scheduler"}, clear=False),
      patch("server.scheduler_worker_manager.SchedulerWorkerManager") as manager_cls,
    ):
      create_worker_manager()
      manager_cls.assert_called_once()


if __name__ == "__main__":
  unittest.main()
