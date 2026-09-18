import asyncio
import os
import subprocess
import sys
import unittest
from unittest.mock import AsyncMock, patch

import torch
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from server import api_server
from server.store import InMemoryStore
from tests.test_trainer_optimizer_correctness import training_requests_processor_module as processor
from training import commands, telemetry


class TracingSetupTest(unittest.TestCase):
  def test_import_is_side_effect_free_and_startup_keeps_one_provider(self) -> None:
    result = subprocess.run(
      [
        sys.executable,
        "-c",
        """
from opentelemetry import trace
from server import telemetry
before = trace.get_tracer_provider()
assert telemetry._provider is None
telemetry.initialize_tracing('open-rl-trainer')
provider = trace.get_tracer_provider()
assert provider is not before
assert provider.resource.attributes['service.name'] == 'open-rl-trainer'
telemetry.initialize_tracing('do-not-replace')
assert trace.get_tracer_provider() is provider
telemetry.shutdown_tracing()
""",
      ],
      env={**os.environ, "ENABLE_GCP_TRACE": "0"},
      capture_output=True,
      text=True,
      check=False,
    )
    self.assertEqual(result.returncode, 0, result.stderr)


class TracePropagationTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.exporter = InMemorySpanExporter()
    self.provider = TracerProvider()
    self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
    self.addCleanup(self.provider.shutdown)
    self.tracer = self.provider.get_tracer("test")
    self.enterContext(patch.object(processor, "tracer", self.tracer))
    self.enterContext(patch.object(telemetry, "tracer", self.tracer))
    self.store = InMemoryStore()
    self.enterContext(patch.object(api_server, "store", self.store))

  async def test_enqueued_trace_reaches_worker_thread_and_next_request_is_isolated(self) -> None:
    with self.tracer.start_as_current_span("api.request") as parent:
      await api_server.enqueue(commands.OptimStep(request_id="first", model_id="model"))
      parent_context = parent.get_span_context()
    self.assertEqual(self.store.futures_store["first"], {"status": "pending"})
    request = (await self.store.get_requests())[0]

    def work():
      with telemetry.TrainingTimings(torch.device("cpu")).phase("forward"):
        pass
      return {"status": "ok"}

    async def dispatch(*_args):
      return await asyncio.to_thread(work)

    handler = AsyncMock()
    handler.dispatch_operation.side_effect = dispatch
    _, result = await processor.TrainingRequestsProcessor.handle_request(handler, request)
    self.assertEqual(result, {"status": "ok"})
    self.assertFalse(trace.get_current_span().get_span_context().is_valid)
    with self.tracer.start_as_current_span("unrelated"):
      await processor.TrainingRequestsProcessor.handle_request(handler, {"op": "optim_step", "model_id": "model", "request_id": "second"})

    spans = self.exporter.get_finished_spans()
    first, second = [span for span in spans if span.name == "training.optim_step"]
    phase = next(span for span in spans if span.name == "training.forward")
    self.assertEqual(first.context.trace_id, parent_context.trace_id)
    self.assertEqual(first.parent.span_id, parent_context.span_id)
    self.assertEqual(phase.parent.span_id, first.context.span_id)
    self.assertIsNone(second.parent)
    self.assertNotEqual(second.context.trace_id, first.context.trace_id)

  async def test_worker_failure_is_recorded_before_conversion_to_response(self) -> None:
    handler = AsyncMock()
    handler.dispatch_operation.side_effect = ValueError("bad training input")
    with patch.object(processor.traceback, "print_exc"):
      _, result = await processor.TrainingRequestsProcessor.handle_request(handler, {"op": "optim_step", "model_id": "model", "request_id": "failed"})
    self.assertEqual(result["type"], "RequestFailedResponse")
    span = self.exporter.get_finished_spans()[0]
    self.assertEqual(span.status.status_code, StatusCode.ERROR)
    self.assertEqual(span.events[0].name, "exception")
    self.assertFalse(trace.get_current_span().get_span_context().is_valid)

  async def test_sampling_submission_propagates_and_failed_result_marks_span(self) -> None:
    from server import vllm_sampler

    with self.tracer.start_as_current_span("api.sample") as parent:
      await api_server.enqueue_sampling({"request_id": "sample", "model_id": "model"})
      parent_context = parent.get_span_context()
    request = (await self.store.get_sampling_requests_for_model("model"))[0]
    with (
      patch.object(vllm_sampler, "tracer", self.tracer),
      patch.object(vllm_sampler, "is_fft_enabled", return_value=False),
      patch.object(vllm_sampler, "run_generation_backend", new=AsyncMock(return_value={"type": "RequestFailedResponse", "error_message": "failed"})),
    ):
      await vllm_sampler.process_sampling_request(request, self.store)
    span = next(span for span in self.exporter.get_finished_spans() if span.name == "sampling.request")
    self.assertEqual(span.parent.span_id, parent_context.span_id)
    self.assertEqual(span.status.status_code, StatusCode.ERROR)
    self.assertEqual((await self.store.get_future("sample", timeout=0))["type"], "RequestFailedResponse")


class TrainingTimingsTest(unittest.TestCase):
  @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
  def test_gpu_times_are_collected_after_existing_scalar_read(self) -> None:
    timings = telemetry.TrainingTimings(torch.device("cuda"))
    tensor = torch.ones(256, device="cuda", requires_grad=True)
    with timings.phase("forward"):
      loss = tensor.square().sum()
    with timings.phase("backward"):
      loss.backward()
    loss.item()
    timings.collect_gpu()
    self.assertGreater(timings.metrics["time/forward_gpu:sum"], 0)
    self.assertGreater(timings.metrics["time/backward_gpu:sum"], 0)


if __name__ == "__main__":
  unittest.main()
