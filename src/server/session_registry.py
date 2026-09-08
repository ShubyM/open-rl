"""Which client sessions are using which owners.

An owner is the scheduler's ownerID, the trainer and sampler pair that
serves one FFT job or one LoRA base model. The tinker client heartbeats its
session every ten seconds for as long as it runs. A session that stops is
dead, and an owner whose sessions are all dead is abandoned, so a job that
exits without calling delete_model still gives its GPUs back.

Everything lives in the store, so a gateway restart keeps it:

  open_rl:session:<id>    present while the session is live. Each heartbeat
                          resets its expiry, so a silent session vanishes
                          on its own.
  open_rl:owner:<owner>   the sessions using this owner's workers.
  open_rl:owners          every owner that has workers.
"""

from server.store import RequestStore


class SessionRegistry:
  def __init__(self, store: RequestStore, ttl_seconds: float = 120.0):
    self.store = store
    self.ttl_seconds = ttl_seconds

  async def heartbeat(self, session_id: str) -> None:
    await self.store.set_value(f"open_rl:session:{session_id}", "1", ttl_seconds=self.ttl_seconds)

  async def live(self, session_id: str) -> bool:
    return await self.store.get_value(f"open_rl:session:{session_id}") is not None

  async def attach(self, session_id: str, owner: str) -> None:
    """The session is using this owner's workers from now on."""
    await self.heartbeat(session_id)
    await self.store.add_to_set("open_rl:owners", owner)
    await self.store.add_to_set(f"open_rl:owner:{owner}", session_id)

  async def abandoned(self) -> list[str]:
    """The owners none of whose sessions are live anymore."""
    result = []
    for owner in sorted(await self.store.set_members("open_rl:owners")):
      for session_id in await self.store.set_members(f"open_rl:owner:{owner}"):
        if not await self.live(session_id):
          await self.store.remove_from_set(f"open_rl:owner:{owner}", session_id)
      if not await self.store.set_members(f"open_rl:owner:{owner}"):
        result.append(owner)
    return result

  async def forget(self, owner: str) -> None:
    """The owner's workers are gone. A session that attached since abandoned() keeps it."""
    if not await self.store.set_members(f"open_rl:owner:{owner}"):
      await self.store.remove_from_set("open_rl:owners", owner)
