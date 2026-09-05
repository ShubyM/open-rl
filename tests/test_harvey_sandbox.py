"""Sandbox injection and episode lifecycle without Podman, a judge, or GPUs."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import tinker
from harvey_labs import env as lab_env
from harvey_labs.cookbook_compat import recipe_runtime
from harvey_labs.episode import LabEpisodeEnv
from harvey_labs.sandbox import PodmanLabSandbox
from harvey_labs.tasks import LabTask
from harvey_labs.train import RunConfig, build_dataset_builder
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import InitialObservationOverflow, StepResult
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use.types import ToolInput


def fake_sandbox():
  return SimpleNamespace(
    sandbox_id="test",
    run_command=AsyncMock(),
    read_file=AsyncMock(),
    write_file=AsyncMock(),
    send_heartbeat=AsyncMock(),
    cleanup=AsyncMock(),
    tool_definitions=[{"name": "read", "parameters": {"type": "object"}}],
    execute_tool=AsyncMock(return_value="document"),
    tool_metrics=lambda: {},
    collect_outputs=AsyncMock(),
  )


@pytest.fixture
def group(tmp_path, monkeypatch):
  tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()), decode=lambda tokens, **kw: bytes(tokens).decode())
  renderer = SimpleNamespace(tokenizer=tokenizer)
  monkeypatch.setattr(lab_env, "lab_renderer", lambda *args: renderer)
  monkeypatch.setattr(lab_env, "lab_system_prompt", lambda *args: "instructions")
  monkeypatch.setattr(lab_env, "initial_messages", lambda *args: [])
  return lab_env.LabEnvGroupBuilder(
    task=LabTask("area/task", "draft output", tmp_path / "documents", 4),
    lab_root=tmp_path,
    model_name="test",
    renderer_name="test",
    group_size=2,
    max_turns=5,
    command_timeout=60,
    judge_model="test",
    judge_parallel=1,
    max_reward_criteria=None,
    max_trajectory_tokens=1000,
    max_generation_tokens=100,
    max_tool_result_tokens=100,
  )


def test_custom_factory_tools_and_binary_outputs_reach_grader(group, monkeypatch):
  created, built, requests = [], [], []

  async def factory(request):
    sandbox = fake_sandbox()

    async def collect(destination):
      (destination / "answer.pdf").write_bytes(b"%PDF-\x00\xff")

    sandbox.collect_outputs.side_effect = collect
    requests.append(request)
    created.append(sandbox)
    return sandbox

  def build(**kwargs):
    built.append(kwargs)
    return SimpleNamespace()

  group.sandbox_factory = factory
  monkeypatch.setattr(lab_env, "build_agent_tool_env", build)

  async def check():
    envs = await group.make_envs()
    assert len(envs) == 2
    assert len({r.output_dir for r in requests}) == 2
    assert all(isinstance(s, SandboxInterface) for s in created)
    await built[0]["tools"][0].run(ToolInput(arguments={"file_path": "doc"}, call_id="1"))
    created[0].execute_tool.assert_awaited_once_with("read", {"file_path": "doc"})
    reward = built[0]["reward_fn"]

    def score(history):
      assert (reward.run_dir / "output" / "answer.pdf").read_bytes() == b"%PDF-\x00\xff"
      return 1.0, {}

    monkeypatch.setattr(reward, "score", score)
    assert await reward([]) == (1.0, {})
    created[0].collect_outputs.side_effect = OSError("transfer failed")
    with pytest.raises(OSError, match="transfer failed"):
      await reward([])
    await group.cleanup()
    await group.cleanup()
    for sandbox in created:
      sandbox.cleanup.assert_awaited_once()

  asyncio.run(check())


@pytest.mark.parametrize("failure", ["startup", "assembly", "cancel"])
def test_partial_group_is_cleaned_up(group, monkeypatch, failure):
  sandbox = fake_sandbox()

  async def check():
    first = asyncio.Event()
    pending = asyncio.Event()
    calls = 0

    async def factory(request):
      nonlocal calls
      calls += 1
      if calls == 1:
        first.set()
        return sandbox
      await first.wait()
      if failure == "cancel":
        pending.set()
        await asyncio.Event().wait()
      if failure == "startup":
        raise RuntimeError("startup failed")
      return fake_sandbox()

    group.sandbox_factory = factory
    monkeypatch.setattr(group, "_build_envs", Mock(side_effect=RuntimeError("assembly failed")))
    task = asyncio.create_task(group.make_envs())
    if failure == "cancel":
      await pending.wait()
      task.cancel()
      with pytest.raises(asyncio.CancelledError):
        await task
    else:
      with pytest.raises((ExceptionGroup, RuntimeError)):
        await task
    sandbox.cleanup.assert_awaited_once()
    assert group.sandboxes == []

  asyncio.run(check())


def test_episode_metrics_include_initial_overflow_and_keep_cookbook_unpatched():
  inner = SimpleNamespace(
    initial_observation=AsyncMock(return_value=InitialObservationOverflow(reward=-0.1, metrics={"max_tokens_reached": 1.0})),
    step=AsyncMock(
      return_value=StepResult(
        reward=-0.1,
        episode_done=True,
        next_observation=tinker.ModelInput.empty(),
        next_stop_condition=[],
        metrics={"parse_error": 1.0},
        logs={"existing": "kept"},
      )
    ),
  )
  env = LabEpisodeEnv(inner, 4, SimpleNamespace(decode=lambda *a, **kw: "<|tool_call>call:read"))
  original_step = EnvFromMessageEnv.step
  with recipe_runtime(RunConfig()):
    assert EnvFromMessageEnv.step is original_step

  async def check():
    initial = await env.initial_observation()
    assert initial.metrics["lab/criteria_total"] == 4
    assert initial.metrics["lab/failed_before_grading"] == 1
    result = await env.step([1], extra={"stop_reason": "stop"})
    assert result.metrics["parse_error"] == 1
    assert result.metrics["max_tokens_reached"] == 0
    assert result.logs["parse_error_call_names"] == "read"
    assert result.logs["existing"] == "kept"
    assert "lab/criteria_total" not in inner.step.return_value.metrics

  asyncio.run(check())


def test_factory_is_forwarded_to_train_and_eval_groups(tmp_path):
  factory = AsyncMock()
  builder = build_dataset_builder(RunConfig(task="test", lab_root=tmp_path), factory)
  task = LabTask("test", "instructions", tmp_path, 1)
  for size in (builder.group_size, builder.eval_group_size):
    assert builder._env_group(task, tmp_path, size).sandbox_factory is factory


def test_podman_adapter_uses_existing_executor_and_cookbook_results(tmp_path):
  native = SimpleNamespace(
    container_name="test",
    output_dir=tmp_path,
    stop=Mock(),
    read_file=Mock(return_value=b"hello"),
    write_file=Mock(),
    exec=Mock(return_value=SimpleNamespace(stdout="abcdef", stderr="", returncode=None, timed_out=True)),
  )
  executor = SimpleNamespace(execute=Mock(return_value="done"), get_metrics=lambda: {"reads": 1})
  adapter = PodmanLabSandbox(native, executor, [])
  assert isinstance(adapter, SandboxInterface)

  async def check():
    result = await adapter.run_command("sleep 99", max_output_bytes=3)
    assert result.stdout == "abc" and result.exit_code == 124
    assert (await adapter.read_file("/workspace/file", max_bytes=2)).stdout == "he"
    await adapter.write_file("/workspace/file", b"\x00\xff")
    native.write_file.assert_called_once_with("/workspace/file", b"\x00\xff")
    assert await adapter.execute_tool("read", {}) == "done"
    await adapter.collect_outputs(tmp_path)
    await adapter.cleanup()
    native.stop.assert_called_once()

  asyncio.run(check())


def test_podman_factory_cancellation_waits_for_start_before_stop(tmp_path, monkeypatch):
  import sys
  import threading

  from harvey_labs.sandbox import SandboxRequest, podman_sandbox_factory

  started, release = threading.Event(), threading.Event()
  order = []

  def start():
    started.set()
    assert release.wait(5)
    order.append("started")

  native = SimpleNamespace(start=start, stop=lambda: order.append("stopped"))
  monkeypatch.setitem(sys.modules, "harness.tools", SimpleNamespace(ToolExecutor=Mock(), get_all_tool_definitions=lambda: []))
  monkeypatch.setitem(sys.modules, "sandbox.sandbox", SimpleNamespace(DEFAULT_IMAGE="test", Sandbox=lambda **kw: native))
  monkeypatch.setattr("harvey_labs.sandbox.add_lab_to_path", lambda path: None)

  async def check():
    request = SandboxRequest(tmp_path, "test", tmp_path, tmp_path, tmp_path, 60)
    task = asyncio.create_task(podman_sandbox_factory(request))
    try:
      assert await asyncio.to_thread(started.wait, 5)
      task.cancel()
      await asyncio.sleep(0)
    finally:
      release.set()
    with pytest.raises(asyncio.CancelledError):
      await task
    assert order == ["started", "stopped"]

  asyncio.run(check())


def test_full_logging_uses_cookbook_printer():
  from tinker_cookbook.rl import train as cookbook_train

  printer = cookbook_train.print_group
  with recipe_runtime(RunConfig(log_full_rollouts=True)):
    assert cookbook_train.print_group is printer


def test_failure_samples_stay_bounded_and_measure_repetition():
  from harvey_labs.episode import failure_logs

  tokenizer = SimpleNamespace(decode=lambda *a, **kw: "x" * 10000)
  logs = failure_logs(tokenizer, [1] * 1024, "max_tokens")
  assert logs["max_tokens_chars"] == 10000
  assert len(logs["max_tokens_text"]) < 2100
  assert logs["max_tokens_distinct_frac"] == round(1 / 512, 4)
