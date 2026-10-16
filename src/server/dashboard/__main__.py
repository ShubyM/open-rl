"""The dashboard as its own process: `python -m server.dashboard`.

It serves fleet inspection separately from the client API, without its queues
or worker manager.
"""

import argparse

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from server.dashboard.router import router

app = FastAPI(title="Open-RL Inspection API")
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
