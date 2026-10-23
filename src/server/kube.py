"""One lightkube watch per Kubernetes kind for the life of the process, shared by every reader in it. The cluster extra
installs lightkube."""

import asyncio
import os
import threading
import time

RESYNC_SECONDS = 300
watched: dict[str, dict[str, dict] | None] = {}  # pods, nodes and workloads by name; None when the kind is not served
watch_errors: dict[str, str | None] = {kind: f"{kind} read failed: no answer yet" for kind in ("pods", "nodes", "workloads")}
attempted = {kind: threading.Event() for kind in watch_errors}  # set once the kind's first list answered
watchers: dict[str, asyncio.Task] = {}  # the loop holds tasks weakly
client = None  # one AsyncClient for every watch


def namespace() -> str:
  return os.getenv("OPEN_RL_WORKER_NAMESPACE") or os.getenv("POD_NAMESPACE") or "openrl-system"


async def follow(api, kind: str, resource, namespace: str | None) -> None:
  """List, then apply each change from the list's resourceVersion. An expired version (410), a failure and every
  RESYNC_SECONDS list again; a silently dropped connection costs at most that long. The last list serves meanwhile."""
  from httpx2 import HTTPStatusError  # lightkube's HTTP client

  while True:
    try:
      async with asyncio.timeout(RESYNC_SECONDS):
        listed = api.list(resource, namespace=namespace)
        objects = {o.metadata.name: o.to_dict() async for o in listed}
        watched[kind], watch_errors[kind] = objects, None
        attempted[kind].set()
        async for change, obj in api.watch(resource, namespace=namespace, resource_version=listed.resourceVersion):
          objects = dict(objects)  # readers on other threads keep the copy they took
          if change == "DELETED":
            objects.pop(obj.metadata.name, None)
          else:
            objects[obj.metadata.name] = obj.to_dict()
          watched[kind] = objects
    except TimeoutError:
      continue
    except HTTPStatusError as exc:  # lightkube's ApiError too, which carries no response when a watch stream sent it
      code = exc.status.code if exc.response is None else exc.response.status_code
      if code == 410:
        continue
      if code == 404:  # no scheduler CRD on a local stack; it is not about to appear
        watched[kind], watch_errors[kind] = None, None
        attempted[kind].set()
        await asyncio.sleep(RESYNC_SECONDS)
        continue
      watch_errors[kind] = f"{kind} read failed: {exc}"
    except Exception as exc:
      watch_errors[kind] = f"{kind} read failed: {exc}"
    attempted[kind].set()
    await asyncio.sleep(5)


def watch(kinds=tuple(watch_errors)) -> None:
  """Start each kind's watch on the running loop, unless it is running, from the in-cluster service account or else the
  kubeconfig. A watch that ended, with a lifespan or a loop that closed, starts again."""
  from lightkube import AsyncClient
  from lightkube.resources.core_v1 import Node, Pod

  from server.scheduler_worker_manager import Workload

  global client
  client = client or AsyncClient()
  scopes = {"pods": (Pod, namespace()), "nodes": (Node, None), "workloads": (Workload, namespace())}
  for kind in (k for k in set(kinds) if k not in watchers or watchers[k].done()):
    watchers[kind] = asyncio.create_task(follow(client, kind, *scopes[kind]))


def wait(kinds, timeout: float = 6) -> None:
  """Block until each kind's first list answered, up to timeout in all."""
  deadline = time.monotonic() + timeout
  for kind in kinds:
    attempted[kind].wait(max(0.0, deadline - time.monotonic()))
