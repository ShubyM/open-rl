import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any


def timed(fn: Callable[..., Any]) -> Callable[..., Any]:
  fn_logger = logging.getLogger(fn.__module__)

  if asyncio.iscoroutinefunction(fn):

    @functools.wraps(fn)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
      t0 = time.perf_counter()
      try:
        return await fn(*args, **kwargs)
      finally:
        fn_logger.info("%s took %.0f ms", fn.__qualname__, (time.perf_counter() - t0) * 1000)

    return async_wrapper

  @functools.wraps(fn)
  def wrapper(*args: Any, **kwargs: Any) -> Any:
    t0 = time.perf_counter()
    try:
      return fn(*args, **kwargs)
    finally:
      fn_logger.info("%s took %.0f ms", fn.__qualname__, (time.perf_counter() - t0) * 1000)

  return wrapper
