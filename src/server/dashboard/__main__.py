"""The dashboard as its own read-only process: `python -m server.dashboard`.

It serves the same pages and /api/v1/dashboard/* as the gateway does, and
records placement history, without the gateway's queues or worker manager.
A gateway restart no longer takes the dashboard with it, and two dashboards
(different ports or namespaces) can run side by side.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from server.dashboard import history
from server.dashboard.router import router
from server.dashboard.snapshot import snapshot
from server.store import get_store


@asynccontextmanager
async def lifespan(_: FastAPI):
  store = get_store()
  recorder = asyncio.create_task(history.recorder(store, lambda: snapshot.placements(store)))
  try:
    yield
  finally:
    recorder.cancel()
    await asyncio.gather(recorder, return_exceptions=True)


app = FastAPI(title="Open-RL Dashboard", lifespan=lifespan)
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
