"""The HTTP side of a routed sampler: what its set's llm-d router calls
instead of the queue, served from the sampler's own engine."""

import asyncio
import time
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

ENGINE_POLL_SECONDS = 5
# A request the sampler can never serve. Anything else is worth another attempt.
PERMANENT_ERRORS = (ValueError, TypeError, KeyError, FileNotFoundError)


class Engine(Protocol):
  errored: bool


class Generator(Protocol):
  engine: Engine

  async def generate(self, request: dict[str, Any]) -> dict[str, Any]: ...


class EngineDead(RuntimeError):
  pass


def failed(message: str, category: str = "server") -> dict[str, Any]:
  return {"type": "RequestFailedResponse", "error_message": message, "category": category}


def http_app(sampler: Generator) -> FastAPI:
  """The body is OpenAI shaped so the router can read the prompt and model;
  its `openrl` field is the request the queue would have carried, and the
  answer is the same result."""
  app = FastAPI()

  @app.post("/v1/completions")
  async def completions(request: Request) -> JSONResponse:
    """200 with the sample; 400 when the request can never succeed here; 503 or
    500 when another attempt may. Leaving the call cancels the engine request,
    and so does the attempt's deadline."""
    if sampler.engine.errored:
      return JSONResponse(failed("vLLM engine is dead"), status_code=503)
    task = None
    status = 200
    try:
      body = await request.json()
      openrl = body["openrl"]
      deadline = openrl.get("deadline")
      task = asyncio.create_task(sampler.generate(openrl))
      while not task.done():
        remaining = deadline - time.time() if deadline else ENGINE_POLL_SECONDS
        if remaining <= 0:
          raise TimeoutError("attempt deadline passed during generation")
        await asyncio.wait({task}, timeout=min(ENGINE_POLL_SECONDS, remaining))
        if sampler.engine.errored:
          raise EngineDead("vLLM engine is dead")
        if await request.is_disconnected():
          raise RuntimeError("Sampling client disconnected")
      result = task.result()
      result["type"] = "sample"
    except EngineDead as exc:
      status, result = 503, failed(str(exc))
    except TimeoutError as exc:
      status, result = 504, failed(str(exc))
    except PERMANENT_ERRORS as exc:
      status, result = 400, failed(f"Invalid sampling request: {type(exc).__name__}: {exc}", "user")
    except Exception as exc:
      status, result = 500, failed(f"vLLM Worker Error: {exc}")
    finally:
      if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return JSONResponse(result, status_code=status)

  @app.get("/health")
  async def health() -> Response:
    return Response(status_code=503 if sampler.engine.errored else 200)

  @app.get("/metrics")
  async def metrics() -> Response:
    # vLLM's metrics; prometheus_client comes with vLLM, which only samplers install.
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

  return app


async def serve_http(sampler: Generator, port: int) -> None:
  server = uvicorn.Server(uvicorn.Config(http_app(sampler), host="0.0.0.0", port=port, log_level="warning"))
  await server.serve()
