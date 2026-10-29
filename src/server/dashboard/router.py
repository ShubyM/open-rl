from fastapi import APIRouter

from server.dashboard import fleet

router = APIRouter(tags=["inspection"])


@router.get("/api/v1/inspect/inventory")
async def inventory():
  """Runs with their lifecycle (ended, active) and an error per unavailable source or unreadable run record. client_status is
  what the client last reported, active until it deletes the model, so a crashed run stays active there; lifecycle is ended
  once none of the run's sessions is live."""
  return await fleet.snapshot()
