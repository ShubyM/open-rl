import asyncio
import inspect
import json
import random
import tempfile
import tracemalloc
import uuid
from array import array
from pathlib import Path
from unittest.mock import patch

import httpx
import redis.asyncio as redis
import safetensors.torch
import torch
import uvicorn
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from accel_timeslicer.serve import start_time_slicer
from accel_timeslicer.single_node import SingleNodeTimeSlicer
from accel_timeslicer.time_slicer import SocketTimeSlicerClient
from accel_timeslicer.workload import WorkloadRef
from server.sampling_dispatcher import UNKNOWN_WEIGHTS, Dispatcher, HttpSampler, http_client, redis_client, revision_file, to_result
from server.sampling_store import SamplingStore, future_key
from tests.redis_server import RedisTestCase
from tests.test_exclusive_trainer import tiny_worker
from training.fft_trainer_worker import FFTTrainingWorker
from training.types import Datum, TensorData

SAMPLE = {"type": "sample", "sequences": [{"tokens": [100, 101], "logprobs": [-0.5, -0.25], "stop_reason": "length"}]}
CHOICE = {"token_ids": [100, 101], "finish_reason": "length", "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -0.25}]}}


async def until(condition, timeout: float = 10) -> None:
  async with asyncio.timeout(timeout):
    while not (await result if inspect.isawaitable(result := condition()) else result):
      await asyncio.sleep(0.02)


def fft_worker() -> FFTTrainingWorker:
  """A real FFT trainer on a tiny Llama, resident on the CPU, publishing sparse deltas."""
  worker = tiny_worker()
  worker.cpu_offload = False
  worker.set_weight_sync_strategy("delta")
  worker.prepare_model_for_training()
  return worker


def train_step(worker: FFTTrainingWorker) -> None:
  row = [3, 5, 8, 13, 21, 34]
  datum = Datum(model_input=row[:-1], loss_fn_inputs={"target_tokens": TensorData(data=row[1:]), "weights": TensorData(data=[1.0] * 5)})
  worker.forward_backward([datum], "cross_entropy")
  worker.optim_step({"learning_rate": 1e-2})


def nudge_step(worker: FFTTrainingWorker, index: int) -> None:
  """An optimizer step that moves one weight. In bf16 at a small learning rate each step moves a different few, so a
  delta of the last step misses what the step before moved."""
  with torch.no_grad():
    worker.model.lm_head.weight.view(-1)[index] += 1.0
  worker.optim_step({})


def weights(worker: FFTTrainingWorker) -> dict[str, torch.Tensor]:
  return {name: param.detach().clone() for name, param in worker.model.named_parameters()}


def replayed(start: dict[str, torch.Tensor], revisions: list[str]) -> dict[str, torch.Tensor]:
  """A replica's weights after it loads `revisions` in order, written the way the delta_snapshot engine writes them."""
  out = {name: tensor.clone() for name, tensor in start.items()}
  for revision in revisions:
    metadata = json.loads(Path(revision, "metadata.json").read_text())
    if metadata.get("format") == "sparse_delta":
      tensors = safetensors.torch.load_file(str(Path(revision, "delta.safetensors")))
      for i, name in enumerate(metadata["layer_names"]):
        out[name].view(-1)[tensors[f"{i}.indices"].long()] = tensors[f"{i}.values"]
    else:
      for file in Path(revision).glob("*.safetensors"):
        out.update((name, tensor) for name, tensor in safetensors.torch.load_file(str(file)).items() if name in out)
  return out


def same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
  return a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)


class FakeReplica:
  """vLLM's weight endpoints and generate with a real weight version. `held` pauses both; an update logs what the readiness probe expects.
  Like vLLM, a paused replica queues generate until it is resumed, one weight update is active from start to finish,
  and only a finish that names a version, or /update_weight_version, changes the version. `prompts` keeps the token ids sent."""

  def __init__(self, name: str, log: list, version: str, expected: Path):
    self.name, self.log, self.version, self.expected, self.held, self.url = name, log, version, expected, asyncio.Event(), ""
    self.resumed, self.probes, self.paused, self.updating, self.fail_update, self.prompts = 0, 0, False, False, False, []
    gets = [Route("/weight_info", lambda r: JSONResponse({"weight_version": self.version}))]
    gets.append(Route("/is_paused", lambda r: JSONResponse({"is_paused": self.paused})))
    posts = {"/pause": self.pause, "/resume": self.resume, "/start_weight_update": self.start, "/update_weight_version": self.set_version}
    posts["/reset_prefix_cache"] = lambda r: JSONResponse({"success": True})
    posts |= {"/update_weights": self.update, "/finish_weight_update": self.finish, "/inference/v1/generate": self.generate}
    self.app = Starlette(routes=gets + [Route(path, handler, methods=["POST"]) for path, handler in posts.items()])

  async def start(self, request) -> Response:
    if self.updating:
      return Response("start_weight_update called while a weight update is already active. Call finish_weight_update first.", status_code=500)
    self.updating = True
    return JSONResponse({})

  async def update(self, request) -> Response:
    if not self.updating:
      return Response("start_weight_update must be called before update_weights.", status_code=500)
    self.log.append(("update", self.name, (await request.json())["update_info"]["target_weights_path"], self.expected.read_text()))
    if self.fail_update:  # like vLLM, a failed update_weights ends the update and keeps the version
      self.fail_update, self.updating = False, False
      return Response("Target weights path does not exist", status_code=500)
    await until(lambda: not self.held.is_set())
    return JSONResponse({})

  async def pause(self, request) -> Response:
    self.paused = True
    return JSONResponse({})

  async def resume(self, request) -> Response:
    self.resumed, self.paused = self.resumed + 1, False
    return JSONResponse({})

  async def finish(self, request) -> Response:
    body = await request.body()
    self.probes += not body  # a bare finish, which vLLM refuses with a 500 traceback when no update is open
    if not self.updating:
      return Response("finish_weight_update called without a matching start_weight_update.", status_code=500)
    self.updating = False
    if body:
      self.version = json.loads(body)["weight_version"]
    return JSONResponse({})

  async def set_version(self, request) -> Response:
    self.version = (await request.json())["new_version"]
    return JSONResponse({})

  async def generate(self, request) -> Response:
    self.log.append(("generate", self.name, self.version))
    self.prompts.append((await request.json())["token_ids"])
    await until(lambda: not self.held.is_set() and not self.paused)
    return JSONResponse({"choices": [CHOICE]})


def routed_dispatcher(
  store: SamplingStore, pool: str, http: httpx.AsyncClient, router_url: str | None, urls: list[str], concurrency: int
) -> Dispatcher:
  """The dispatcher of a pool of vllm serve replicas, as sampling_dispatcher.main builds it."""
  return Dispatcher(store, pool, HttpSampler(http, pool, router_url, urls), concurrency)


async def run_routed(dispatcher: Dispatcher) -> None:
  """Runs it as sampling_dispatcher.main does, closing the sampler when it stops."""
  try:
    await dispatcher.run()
  finally:
    await dispatcher.sampler.close()


def router(log: list, replicas: list[FakeReplica], http: httpx.AsyncClient) -> Starlette:
  # Forwards generate to a ready replica, one whose version is the expected one, like the EPP.
  async def route(request) -> Response:
    ready = [r for r in replicas if r.version == r.expected.read_text()]
    if not ready:
      log.append(("unrouted", "", ""))
      return Response(status_code=503)
    upstream = await http.post(f"{ready[0].url}/inference/v1/generate", content=await request.body(), timeout=None)
    return Response(upstream.content, upstream.status_code, headers={"x-openrl-upstream": ready[0].url.removeprefix("http://")})

  return Starlette(routes=[Route("/inference/v1/generate", route, methods=["POST"])])


class SamplingDispatcherTest(RedisTestCase):
  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.store, self.pool, self.log, self.http = SamplingStore(self.redis), f"sampler-{uuid.uuid4()}", [], httpx.AsyncClient()
    self.addAsyncCleanup(self.http.aclose)
    self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
    self.enterContext(patch.dict("os.environ", {"OPEN_RL_TMP_DIR": str(self.tmp)}))

  async def serve(self, app: Starlette) -> str:
    server = uvicorn.Server(uvicorn.Config(app, port=0, log_level="warning", lifespan="off"))
    serving = asyncio.create_task(server.serve())
    self.addAsyncCleanup(lambda: (setattr(server, "should_exit", True), serving)[1])
    await until(lambda: server.started)
    return f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"

  async def replicas(self, *versions: str) -> list[FakeReplica]:
    replicas = [FakeReplica(f"pod-{i}", self.log, version, revision_file(self.pool)) for i, version in enumerate(versions)]
    for replica in replicas:
      replica.url = await self.serve(replica.app)
      self.addAsyncCleanup(replica.held.clear)
    return replicas

  def revision(self, name: str, **metadata) -> str:
    (self.tmp / name).mkdir()
    (self.tmp / name / "metadata.json").write_text(json.dumps(metadata))
    return str(self.tmp / name)

  def loads(self, *names: str) -> dict[str, list[str]]:
    return {name: [entry[2] for entry in self.log if entry[:2] == ("update", name)] for name in names}

  async def accepted(self, revision: str = "", prompt: list | None = None) -> dict:
    identity = {"request_id": str(uuid.uuid4()), "pool": self.pool, "revision": revision, "model": "base", "idempotency_key": None}
    r = {**identity, "prompt_token_ids": prompt or [1, 2, 3], "sampling_params": {"n": 1, "max_tokens": 2}, "include_prompt_logprobs": False}
    await self.store.accept(r)
    return r

  async def dispatcher(self, replicas: list[FakeReplica], concurrency: int = 4) -> asyncio.Task:
    router_url = await self.serve(router(self.log, replicas, self.http))
    dispatcher = routed_dispatcher(self.store, self.pool, self.http, router_url, [r.url for r in replicas], concurrency=concurrency)
    run = asyncio.create_task(run_routed(dispatcher))
    self.addAsyncCleanup(lambda: (run.cancel(), asyncio.gather(run, return_exceptions=True))[1])
    return run

  async def results(self, request_id: str) -> list[dict]:
    await until(lambda: self.redis.llen(future_key(request_id)))
    await asyncio.sleep(0.3)  # a second result would show by now
    return [json.loads(raw) for raw in await self.redis.lrange(future_key(request_id), 0, -1)]

  async def test_each_request_is_generated_once(self):
    replicas = await self.replicas("default", "default")
    records = [await self.accepted() for _ in range(40)]
    await self.dispatcher(replicas, concurrency=256)
    for r in records:
      await until(lambda r=r: self.redis.llen(future_key(r["request_id"])))
    await asyncio.sleep(0.5)  # a duplicate would be generated by now
    self.assertEqual(sum(entry[0] == "generate" for entry in self.log), len(records))

  async def test_a_burst_of_answers_saves_each_once(self):
    replicas = await self.replicas("default", "default")
    records = [await self.accepted() for _ in range(300)]
    # The dispatcher's own clients, as main() builds them: 256 sends answered together commit together.
    self.store = SamplingStore(await self.enterAsyncContext(redis_client(self.url)))
    self.http = await self.enterAsyncContext(http_client())
    await self.dispatcher(replicas, concurrency=256)
    for r in records:
      await until(lambda r=r: self.redis.llen(future_key(r["request_id"])), timeout=60)
    await asyncio.sleep(0.5)  # a resend after a failed commit would generate again by now
    # A send the test router drops is retried, so a few may generate twice. Commits that
    # find no free connection fail and resend about half the burst.
    self.assertLess(sum(entry[0] == "generate" for entry in self.log) - len(records), len(records) // 20)

  async def test_a_burst_over_many_replicas_leaves_at_most_32_idle_connections(self):
    # httpx scans every pooled connection whenever a request starts or ends, so hundreds left idle stall the dispatcher.
    replicas = await self.replicas(*["default"] * 8)
    for replica in replicas:
      replica.held.set()  # every send stays in flight, each on a connection of its own, until all are out
    records = [await self.accepted() for _ in range(100)]
    # The dispatcher's own clients, as main() builds them.
    self.store = SamplingStore(await self.enterAsyncContext(redis_client(self.url)))
    self.http = await self.enterAsyncContext(http_client())
    dispatcher = routed_dispatcher(self.store, self.pool, self.http, None, [r.url for r in replicas], concurrency=256)
    run = asyncio.create_task(run_routed(dispatcher))
    self.addAsyncCleanup(lambda: (run.cancel(), asyncio.gather(run, return_exceptions=True))[1])
    await until(lambda: sum(entry[0] == "generate" for entry in self.log) == len(records), timeout=30)
    for replica in replicas:
      replica.held.clear()
    for r in records:
      await until(lambda r=r: self.redis.llen(future_key(r["request_id"])), timeout=60)
    idle = [connection for connection in self.http._transport._pool.connections if connection.is_idle()]
    self.assertLessEqual(len(idle), 32)

  async def test_a_request_no_replica_answers_fails_at_the_deadline(self):
    replicas = await self.replicas("other-weights")  # never ready for the base weights, so nothing is sent
    self.enterContext(patch("server.sampling_dispatcher.SAMPLE_DEADLINE_SECONDS", 1))
    r = await self.accepted()
    await self.dispatcher(replicas)
    [result] = await self.results(r["request_id"])
    self.assertEqual(result["type"], "RequestFailedResponse")
    self.assertIn("No sampler answered", result["error_message"])
    self.assertEqual(await self.store.pending(self.pool), [])

  async def test_sends_to_a_replica_that_never_answers_give_back_their_slots_when_their_records_fail(self):
    hung, healthy = await self.replicas("default", "default")
    hung.held.set()  # it takes generate and never answers, like a wedged GPU or a node gone without closing TCP
    routed = []

    # Like the EPP, the first two picks land on the hung replica and every later pick on the healthy one.
    async def route(request) -> Response:
      target = hung if len(routed) < 2 else healthy
      routed.append(target.name)
      upstream = await self.http.post(f"{target.url}/inference/v1/generate", content=await request.body(), timeout=None)
      return Response(upstream.content, upstream.status_code, headers={"x-openrl-upstream": target.url.removeprefix("http://")})

    router_url = await self.serve(Starlette(routes=[Route("/inference/v1/generate", route, methods=["POST"])]))
    first = [await self.accepted() for _ in range(2)]
    dispatcher = routed_dispatcher(self.store, self.pool, self.http, router_url, [hung.url, healthy.url], concurrency=2)
    run = asyncio.create_task(run_routed(dispatcher))
    self.addAsyncCleanup(lambda: (run.cancel(), asyncio.gather(run, return_exceptions=True))[1])
    await until(lambda: self.log.count(("generate", "pod-0", "default")) == 2)
    await asyncio.sleep(1.1)
    # The deadline passes while both sends still wait.
    with patch("server.sampling_dispatcher.SAMPLE_DEADLINE_SECONDS", 1):
      for r in first:
        [result] = await self.results(r["request_id"])
        self.assertEqual(result["type"], "RequestFailedResponse")
    # The slots free at once, not when the hung replica lets go some seconds later.
    await until(lambda: not dispatcher.in_flight, timeout=2)
    later = await self.accepted()
    self.assertEqual(await self.results(later["request_id"]), [SAMPLE])
    self.assertEqual(routed, ["pod-0", "pod-0", "pod-1"])

  async def test_a_send_gives_up_at_its_records_deadline(self):
    (hung,) = await self.replicas("default")
    hung.held.set()
    self.enterContext(patch("server.sampling_dispatcher.SAMPLE_DEADLINE_SECONDS", 1))
    await self.accepted()
    [r] = await self.store.pending(self.pool)
    dispatcher = routed_dispatcher(self.store, self.pool, self.http, None, [hung.url], concurrency=1)
    await asyncio.wait_for(dispatcher.send(r), 5)
    self.assertEqual(self.log, [("generate", "pod-0", "default")])

  async def test_the_dispatcher_holds_a_pending_prompt_in_a_few_bytes_a_token(self):
    # It keeps every record until it commits, inside a 2Gi limit, so a long-context backlog must stay small once read
    # and while it is read. A list of ints costs about 36 bytes a token.
    prompt, count = [random.randrange(1000, 150000) for _ in range(20_000)], 50
    for _ in range(count):
      await self.accepted(prompt=prompt)
    dispatcher = routed_dispatcher(self.store, self.pool, self.http, None, ["http://127.0.0.1:9"], concurrency=4)  # nothing answers
    tracemalloc.start()
    self.addCleanup(tracemalloc.stop)
    before, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    await dispatcher.dispatch()
    after, peak = tracemalloc.get_traced_memory()
    held, peak_held = (after - before) / (count * len(prompt)), (peak - before) / (count * len(prompt))
    self.assertEqual(len(dispatcher.records), count)
    self.assertLess(held, 8, f"{held:.1f} bytes a token once read")
    self.assertLess(peak_held, 20, f"{peak_held:.1f} bytes a token while read")

  async def test_a_prompt_kept_compact_is_sent_and_scored_as_the_client_sent_it(self):
    (replica,) = replicas = await self.replicas("default")
    # A negative token does not fit array('I'), so that prompt stays the list the client sent, for vLLM to refuse.
    records = [await self.accepted(), await self.accepted(prompt=[-1, 2])]
    compact, as_sent = await self.store.pending(self.pool)
    self.assertEqual((type(compact["prompt_token_ids"]), as_sent["prompt_token_ids"]), (array, [-1, 2]))
    body = {"choices": [CHOICE], "prompt_logprobs": [None, {"2": {"logprob": -0.5}}, {"3": {"logprob": -0.25}}]}
    self.assertEqual(to_result(body, {**compact, "include_prompt_logprobs": True})["prompt_logprobs"], [None, -0.5, -0.25])
    await self.dispatcher(replicas)
    for r in records:
      self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertEqual(sorted(replica.prompts), [[-1, 2], [1, 2, 3]])

  async def test_a_redis_timeout_does_not_stop_dispatching(self):
    replicas = await self.replicas("default")
    self.enterContext(patch("server.sampling_dispatcher.RETRY_MS", 100))
    impatient = await self.enterAsyncContext(redis.from_url(self.url, decode_responses=True, socket_timeout=0.3, retry=Retry(NoBackoff(), 0)))
    patient, self.store = self.store, SamplingStore(impatient)
    run = await self.dispatcher(replicas)
    self.store = patient
    await self.redis.execute_command("CLIENT", "PAUSE", "1000", "ALL")
    r = await self.accepted()  # waits out the pause; the dispatcher's reads time out meanwhile
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertFalse(run.done())

  async def test_a_replica_that_is_still_starting_fails_nothing(self):
    (replica,) = replicas = await self.replicas("default")
    starting = "http://127.0.0.1:9"  # nothing listens here yet
    r = await self.accepted()
    router_url = await self.serve(router(self.log, replicas, self.http))
    dispatcher = routed_dispatcher(self.store, self.pool, self.http, router_url, [starting, replica.url], concurrency=4)
    run = asyncio.create_task(run_routed(dispatcher))
    self.addAsyncCleanup(lambda: (run.cancel(), asyncio.gather(run, return_exceptions=True))[1])
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])

  async def test_crash_mid_request_saves_exactly_one_result(self):
    (replica,) = replicas = await self.replicas("default")
    replica.held.set()
    r = await self.accepted()
    first = await self.dispatcher(replicas)
    await until(lambda: self.log.count(("generate", "pod-0", "default")) == 1)
    first.cancel()
    await self.dispatcher(replicas)
    replica.held.clear()
    self.assertEqual((await self.results(r["request_id"]), await self.store.pending(self.pool)), ([SAMPLE], []))

  async def test_an_answer_from_the_wrong_weights_is_not_saved_and_the_pod_is_prepared_again(self):
    (replica,) = replicas = await self.replicas(w1 := self.revision("w1"))
    replica.held.set()
    r = await self.accepted(w1)
    await self.dispatcher(replicas)
    await until(lambda: ("generate", "pod-0", w1) in self.log)
    replica.version = "default"  # restarted in place while the request ran
    replica.held.clear()
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertEqual(self.log, [("generate", "pod-0", w1), ("update", "pod-0", w1, w1), ("generate", "pod-0", w1)])

  async def test_fft_switch_expects_the_new_version_before_any_weight_update_and_sends_only_to_a_ready_pod(self):
    w1, w2 = self.revision("w1"), self.revision("w2")
    r = await self.accepted(w2)
    await self.dispatcher(await self.replicas(w1, w1))
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertEqual(sorted(self.log[:2]), [("update", "pod-0", w2, w2), ("update", "pod-1", w2, w2)])
    self.assertEqual([(entry[0], entry[2]) for entry in self.log[2:]], [("generate", w2)])

  async def test_a_deleted_revision_fails_its_requests_and_the_pool_moves_on(self):
    (replica,) = replicas = await self.replicas("default")
    gone, later = await self.accepted(str(self.tmp / "gone")), await self.accepted(self.revision("w2", format="sparse_delta", parent=""))
    await self.dispatcher(replicas)
    failed = {"type": "RequestFailedResponse", "error_message": f"Sampler weights {self.tmp / 'gone'} no longer exist"}
    self.assertEqual(await self.results(gone["request_id"]), [failed])
    self.assertEqual((await self.results(later["request_id"]), replica.version), ([SAMPLE], later["revision"]))

  async def test_a_replica_left_paused_at_the_pool_revision_is_resumed_on_startup_and_serves(self):
    (replica,) = replicas = await self.replicas(w1 := self.revision("w1"))
    replica.paused = True  # the previous dispatcher died after this replica loaded w1 and before its /resume
    r = await self.accepted(w1)
    await self.dispatcher(replicas)
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertEqual((replica.resumed, replica.paused, self.loads("pod-0")), (1, False, {"pod-0": []}))

  async def test_a_replica_left_paused_at_an_older_revision_is_resumed_and_then_loaded(self):
    w1 = self.revision("w1", format="sparse_delta", parent="")
    w2 = self.revision("w2", format="sparse_delta", parent=w1)
    (replica,) = replicas = await self.replicas(w1)
    replica.paused = True
    r = await self.accepted(w2)
    await self.dispatcher(replicas)
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertEqual((replica.resumed, replica.paused, replica.version, self.loads("pod-0")), (2, False, w2, {"pod-0": [w2]}))

  async def test_a_replica_left_inside_a_weight_update_takes_a_full_load_before_it_serves(self):
    w1, w2 = self.revision("w1"), self.revision("w2")
    (replica,) = replicas = await self.replicas(w1)
    # The previous dispatcher was killed after /start_weight_update for w2, so the replica may hold w1, w2 or part of w2.
    replica.paused, replica.updating = True, True
    r = await self.accepted(w2)
    await self.dispatcher(replicas)
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertEqual((replica.version, replica.updating, replica.paused), (w2, False, False))
    self.assertEqual([entry[:3] for entry in self.log], [("update", "pod-0", w2), ("generate", "pod-0", w2)])

  async def test_a_replica_left_inside_a_weight_update_at_the_pool_revision_does_not_serve_until_reloaded(self):
    w1 = self.revision("w1")
    self.revision("w2")
    (replica,) = replicas = await self.replicas(w1)
    # The pool still names w1 while the replica, cut off loading w2, may hold w2's changes under w1's version.
    replica.paused, replica.updating = True, True
    r = await self.accepted(w1)
    await self.dispatcher(replicas)
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertEqual((replica.version, replica.updating), (w1, False))
    self.assertEqual([entry[:3] for entry in self.log], [("update", "pod-0", w1), ("generate", "pod-0", w1)])

  async def test_a_replica_resumed_inside_a_weight_update_has_it_closed_before_any_load(self):
    w1, w2 = self.revision("w1"), self.revision("w2")
    (replica,) = replicas = await self.replicas(w1)
    replica.updating = True  # a dispatcher cut off between start and finish resumed the replica before it died, so nothing marks it paused
    r = await self.accepted(w2)
    with self.assertLogs("server.sampling_dispatcher", "WARNING") as logs:
      await self.dispatcher(replicas)
      self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertNotIn("start_weight_update", "\n".join(logs.output))  # no load starts into the open update and is refused
    self.assertEqual((self.loads("pod-0"), replica.version, replica.updating), ({"pod-0": [w2]}, w2, False))

  async def test_a_replica_resumed_inside_a_weight_update_at_the_pool_revision_does_not_serve_until_reloaded(self):
    w1 = self.revision("w1")
    self.revision("w2")
    (replica,) = replicas = await self.replicas(w1)
    # A dispatcher stopped while loading w2 resumed the replica, which may hold part of w2 under w1's version.
    replica.updating = True
    r = await self.accepted(w1)
    await self.dispatcher(replicas)
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertEqual([entry[:3] for entry in self.log], [("update", "pod-0", w1), ("generate", "pod-0", w1)])
    self.assertEqual((replica.version, replica.updating), (w1, False))

  async def test_a_load_cut_off_inside_its_update_stays_paused_until_the_next_dispatcher_reloads_the_replica(self):
    w1, w2 = self.revision("w1"), self.revision("w2")
    (replica,) = replicas = await self.replicas(w1)
    replica.held.set()  # update_weights for w2 outlasts the shutdown grace, as a dense reload over NFS can
    pending_w2 = await self.accepted(w2)
    first = await self.dispatcher(replicas)
    await until(lambda: ("update", "pod-0", w2, w2) in self.log)
    first.cancel()  # SIGTERM, from a pod deletion, a rollout or a pool release
    await asyncio.gather(first, return_exceptions=True)
    await asyncio.sleep(0.3)  # a resume sent by the cut-off load would land by now
    self.assertEqual((replica.paused, replica.updating, replica.version), (True, True, w1))
    replica.held.clear()  # vLLM finishes writing w2, but nothing finishes the update, so the replica still says w1
    # The w2 request is answered elsewhere, and a late request for w1 arrives.
    [r] = [r for r in await self.store.pending(self.pool) if r["request_id"] == pending_w2["request_id"]]
    await self.store.set_result(r, SAMPLE)
    late = await self.accepted(w1)
    await self.dispatcher(replicas)
    self.assertEqual(await self.results(late["request_id"]), [SAMPLE])
    # The next dispatcher closed the update, which left the replica unknown, and loaded w1 in full before serving from it.
    self.assertEqual(self.log, [("update", "pod-0", w2, w2), ("update", "pod-0", w1, w1), ("generate", "pod-0", w1)])
    self.assertEqual((replica.version, replica.updating, replica.paused), (w1, False, False))

  async def test_a_failed_update_leaves_the_replica_reporting_unknown_weights_until_a_full_load(self):
    w1, w2 = self.revision("w1"), self.revision("w2")
    (replica,) = replicas = await self.replicas(w1)
    replica.fail_update = True  # the first update_weights fails inside vLLM, which may have written part of w2
    replica.held.set()
    r = await self.accepted(w2)
    await self.dispatcher(replicas)
    await until(lambda: self.loads("pod-0") == {"pod-0": [w2, w2]})
    self.assertEqual((replica.version, replica.updating, replica.resumed), (UNKNOWN_WEIGHTS, True, 1))
    replica.held.clear()
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertEqual((self.loads("pod-0"), replica.version), ({"pod-0": [w2, w2]}, w2))

  async def test_a_replica_is_probed_for_an_open_update_on_first_sight_and_after_a_load_that_breaks_off_only(self):
    # The probe is a bare finish_weight_update, and vLLM logs a 500 traceback for each one that finds no update open.
    (replica,) = await self.replicas("default")
    sampler = HttpSampler(self.http, self.pool, None, [replica.url])

    async def load(revision: str) -> None:
      sampler.revision = revision
      sampler.write_revision()
      sampler.loading[replica.url] = task = asyncio.create_task(sampler.load_weights(replica.url, revision))
      await task

    await sampler.inspect(replica.url)
    w1, w2 = self.revision("w1"), self.revision("w2")
    await load(w1)
    await sampler.inspect(replica.url)
    self.assertEqual((replica.probes, replica.version), (1, w1))
    replica.fail_update = True
    await load(w2)
    probes = replica.probes  # the broken load closes its own update with a bare finish
    await sampler.inspect(replica.url)
    self.assertEqual((replica.probes, replica.version), (probes + 1, UNKNOWN_WEIGHTS))

  async def test_under_the_delta_strategy_a_replica_with_unknown_weights_stays_out_of_service(self):
    # No full revision is kept beside the deltas yet, so a replica that lost its place in the chain cannot be brought back.
    w1 = self.revision("w1", format="sparse_delta", parent="")
    self.enterContext(patch("server.sampling_dispatcher.SAMPLE_DEADLINE_SECONDS", 1))
    (replica,) = replicas = await self.replicas("default")
    replica.paused, replica.updating = True, True  # cut off loading w1
    r = await self.accepted(w1)
    await self.dispatcher(replicas)
    [result] = await self.results(r["request_id"])
    self.assertEqual(result["type"], "RequestFailedResponse")
    self.assertEqual((replica.version, replica.paused, self.loads("pod-0")), (UNKNOWN_WEIGHTS, False, {"pod-0": []}))

  async def test_a_stopping_dispatcher_lets_a_load_in_flight_finish_and_resume(self):
    (replica,) = replicas = await self.replicas("default")
    replica.held.set()
    await self.accepted(w1 := self.revision("w1"))
    run = await self.dispatcher(replicas)
    await until(lambda: ("update", "pod-0", w1, w1) in self.log)
    run.cancel()
    asyncio.get_running_loop().call_later(0.2, replica.held.clear)
    await asyncio.gather(run, return_exceptions=True)
    self.assertEqual((replica.version, replica.resumed, replica.paused), (w1, 1, False))

  async def test_a_delta_loads_on_the_save_its_optimizer_step_started_from_past_a_save_the_pool_never_sampled(self):
    self.enterContext(patch("server.sampling_dispatcher.SAMPLE_DEADLINE_SECONDS", 5))
    worker = fft_worker()
    train_step(worker)
    w1, at_w1 = worker.save_for_sampler("job", None, "tinker://job/sampler_weights/sampler-0"), weights(worker)
    train_step(worker)
    worker.save_for_sampler("job", "ckpt", "tinker://job/sampler_weights/ckpt")  # a named save nobody samples
    w2 = worker.save_for_sampler("job", None, "tinker://job/sampler_weights/sampler-1")
    (replica,) = replicas = await self.replicas(w1)
    r = await self.accepted(w2)
    await self.dispatcher(replicas)
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertEqual((self.loads("pod-0"), replica.version), ({"pod-0": [w2]}, w2))
    self.assertTrue(same(replayed(at_w1, [w2]), weights(worker)))
    self.assertEqual(json.loads(Path(w2, "metadata.json").read_text())["parent"], w1)

  async def test_a_sampler_save_two_optimizer_steps_after_the_last_is_written_in_full_and_loads_the_trainers_weights(self):
    self.enterContext(patch("server.sampling_dispatcher.SAMPLE_DEADLINE_SECONDS", 5))
    worker = fft_worker()
    train_step(worker)
    w1, at_w1 = worker.save_for_sampler("job", None, "tinker://job/sampler_weights/sampler-0"), weights(worker)
    nudge_step(worker, 0)
    nudge_step(worker, 1)  # a delta holds only this step, so no replica on w1 could apply it
    w2 = worker.save_for_sampler("job", None, "tinker://job/sampler_weights/sampler-1")
    (replica,) = replicas = await self.replicas(w1)
    r = await self.accepted(w2)
    await self.dispatcher(replicas)
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    self.assertEqual((self.loads("pod-0"), replica.version), ({"pod-0": [w2]}, w2))
    self.assertTrue(same(replayed(at_w1, [w2]), weights(worker)))
    metadata = json.loads(Path(w2, "metadata.json").read_text())
    self.assertEqual((metadata.get("format"), metadata.get("parent")), (None, None))

  async def test_a_replica_on_any_save_at_the_weights_a_delta_starts_from_or_reaches_applies_it(self):
    worker = fft_worker()
    train_step(worker)
    # An RL step with nothing to learn from takes no optimizer step, so its rollouts sample a second save at the same weights.
    first, at_first = worker.save_for_sampler("job", None, "tinker://job/sampler_weights/sampler-0"), weights(worker)
    second = worker.save_for_sampler("job", None, "tinker://job/sampler_weights/sampler-1")
    # The late replica started on the first save and never loads the second.
    early, late = await self.replicas(first, first)
    r = await self.accepted(second)
    run = await self.dispatcher([early])
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)
    train_step(worker)
    third = worker.save_for_sampler("job", None, "tinker://job/sampler_weights/sampler-2")
    r = await self.accepted(third)
    await self.dispatcher([early, late])
    self.assertEqual(await self.results(r["request_id"]), [SAMPLE])
    await until(lambda: self.loads(late.name) == {late.name: [third]})
    self.assertEqual(self.loads(early.name), {early.name: [second, third]})
    self.assertTrue(same(replayed(at_first, [second, third]), weights(worker)))
    self.assertTrue(same(replayed(at_first, [third]), weights(worker)))


class RecordingRestorer:
  """Parks and restores nothing, and logs each park and restore the time slicer asks for."""

  def __init__(self, log: list) -> None:
    self.log = log

  def checkpoint(self, workload: WorkloadRef) -> None:
    self.log.append(("park", workload.name))

  def restore(self, workload: WorkloadRef) -> None:
    self.log.append(("restore", workload.name))


class GpuSampler:
  """Stands in for vllm_sampler.Sampler, whose vLLM engine needs a GPU. It logs each call and fails any that reaches
  it off the GPU, or while the trainer holds it."""

  def __init__(self, log: list, gpu: set) -> None:
    self.log, self.gpu, self.revision = log, gpu, ""

  async def wake(self) -> None:
    self.gpu.add("sampler")
    self.log.append(("wake", "sampler"))

  async def sleep(self) -> None:
    self.gpu.discard("sampler")
    self.log.append(("sleep", "sampler"))

  async def ensure_weights(self, revision: str) -> bool:
    assert self.gpu == {"sampler"}, self.gpu
    if revision != self.revision:
      self.log.append(("load", revision))
      self.revision = revision
    return True

  async def generate(self, r: dict) -> dict:
    assert self.gpu == {"sampler"}, self.gpu
    self.log.append(("generate", r["revision"]))
    await asyncio.sleep(0.01)
    return SAMPLE


class TimeSlicedPoolTest(RedisTestCase):
  """A pool of one sampler on its trainer's GPU. The dispatcher runs each batch inside one turn from the real time
  slicer, over its socket, and the trainer steps between batches."""

  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.store, self.pool, self.log, self.gpu = SamplingStore(self.redis), f"sampler-{uuid.uuid4()}", [], set()
    self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
    socket_path = str(self.tmp / "slicer.sock")
    server = await start_time_slicer(SingleNodeTimeSlicer(RecordingRestorer(self.log)), socket_path)
    self.addAsyncCleanup(server.wait_closed)
    self.addCleanup(server.close)
    self.sampler_slicer, self.trainer_slicer = SocketTimeSlicerClient(socket_path=socket_path), SocketTimeSlicerClient(socket_path=socket_path)
    self.addAsyncCleanup(self.sampler_slicer.close)
    self.addAsyncCleanup(self.trainer_slicer.close)
    self.sampler, self.trainer = WorkloadRef("sampler", "gpu-0"), WorkloadRef("trainer", "gpu-0")
    await self.sampler_slicer.register(self.sampler)
    await self.trainer_slicer.register(self.trainer)

  async def accepted(self, revision: str = "") -> dict:
    r = {"request_id": str(uuid.uuid4()), "pool": self.pool, "revision": revision, "model": "base", "idempotency_key": None}
    r |= {"prompt_token_ids": [1, 2, 3], "sampling_params": {"n": 1, "max_tokens": 2}, "include_prompt_logprobs": False}
    await self.store.accept(r)
    return r

  async def saved(self, *records: dict) -> None:
    for r in records:
      await until(lambda r=r: self.redis.llen(future_key(r["request_id"])))

  async def test_each_batch_runs_inside_one_turn_and_the_trainer_steps_between_batches(self):
    w1 = str(self.tmp / "w1")
    Path(w1).mkdir()
    Path(w1, "metadata.json").write_text("{}")
    first = [await self.accepted() for _ in range(3)]
    dispatcher = Dispatcher(self.store, self.pool, GpuSampler(self.log, self.gpu), 4, self.sampler_slicer, self.sampler)
    run = asyncio.create_task(dispatcher.run())
    self.addAsyncCleanup(lambda: (run.cancel(), asyncio.gather(run, return_exceptions=True))[1])
    await self.saved(*first)
    # The trainer steps on the GPU, and the next batch waits for its turn.
    async with self.trainer_slicer.acquire(self.trainer):
      self.log.append(("step", "trainer"))
      second = [await self.accepted(w1) for _ in range(2)]
      await asyncio.sleep(0.3)
      self.assertNotIn(("load", w1), self.log)
    await self.saved(*second)
    await until(lambda: self.log[-1] == ("park", "sampler"))
    batch = [("generate", "")] * 3
    turn = [("restore", "sampler"), ("wake", "sampler"), ("load", w1), ("generate", w1), ("generate", w1), ("sleep", "sampler"), ("park", "sampler")]
    expected = [("wake", "sampler"), *batch, ("sleep", "sampler"), ("park", "sampler"), ("step", "trainer"), ("park", "trainer"), *turn]
    self.assertEqual(self.log, expected)
    self.assertEqual([json.loads(raw) for r in first + second for raw in await self.redis.lrange(future_key(r["request_id"]), 0, -1)], [SAMPLE] * 5)
