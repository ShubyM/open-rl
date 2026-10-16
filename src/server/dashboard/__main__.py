"""The dashboard as its own process: `python -m server.dashboard`.

It serves fleet inspection separately from the client API, without its queues
or worker manager, and records placement history while it runs.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager, suppress

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from server.dashboard import history
from server.dashboard.router import router
from server.dashboard.snapshot import snapshot
from server.store import get_state_store


@asynccontextmanager
async def lifespan(_: FastAPI):
  recorder = asyncio.create_task(history.recorder(get_state_store(), snapshot))
  yield
  recorder.cancel()
  with suppress(asyncio.CancelledError):
    await recorder


app = FastAPI(title="Open-RL Inspection API", lifespan=lifespan)
app.include_router(router)


@app.get("/", include_in_schema=False)
async def index():
  return RedirectResponse("/dashboard")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  # Flags only: Kubernetes injects OPEN_RL_DASHBOARD_PORT=tcp://... for a
  # Service named open-rl-dashboard, so an env default here would collide.
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8080)
  args = parser.parse_args()
  uvicorn.run(app, host=args.host, port=args.port)
