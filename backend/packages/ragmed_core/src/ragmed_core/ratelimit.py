"""Fixed-window rate limits: at most `limit` hits per key per window.

Valkey (an open-source Redis) keeps the counters when it is configured, so every instance of a
service shares them; otherwise they are kept in memory, which suits tests and local runs. If
Valkey cannot be reached, requests are let through and the failure is logged: an outage of the
limiter should not take the clinic offline, and the web app's bot check still applies.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import structlog

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    count: int
    retry_after: int  # seconds until the window resets


class RateLimiter(Protocol):
    async def hit(self, key: str, limit: int, window: int) -> Decision:
        """Count one hit for `key` in the current `window` (seconds) and judge it."""
        ...


def _window(now: float, window: int) -> tuple[int, int]:
    index = int(now // window)
    return index, int((index + 1) * window - now) + 1


class MemoryRateLimiter:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._counts: dict[tuple[str, int], int] = {}

    async def hit(self, key: str, limit: int, window: int) -> Decision:
        index, retry_after = _window(self._clock(), window)
        self._counts = {k: v for k, v in self._counts.items() if k[1] >= index - 1 or k[0] != key}
        count = self._counts.get((key, index), 0) + 1
        self._counts[(key, index)] = count
        return Decision(count <= limit, count, retry_after)


class ValkeyRateLimiter:
    def __init__(
        self, url: str, *, prefix: str = "ragmed:rl", clock: Callable[[], float] = time.time
    ) -> None:
        from redis.asyncio import Redis  # noqa: PLC0415 - optional dependency

        self._client = Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
        self._prefix = prefix
        self._clock = clock

    async def hit(self, key: str, limit: int, window: int) -> Decision:
        index, retry_after = _window(self._clock(), window)
        name = f"{self._prefix}:{key}:{index}"
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.incr(name)
                pipe.expire(name, window + 1, nx=True)
                count, _ = await pipe.execute()
        except Exception as error:
            log.error("rate limiter unavailable", error=str(error))
            return Decision(True, 0, 0)
        return Decision(int(count) <= limit, int(count), retry_after)

    async def close(self) -> None:
        await self._client.aclose()


def make_rate_limiter(valkey_url: str | None) -> RateLimiter:
    return ValkeyRateLimiter(valkey_url) if valkey_url else MemoryRateLimiter()
