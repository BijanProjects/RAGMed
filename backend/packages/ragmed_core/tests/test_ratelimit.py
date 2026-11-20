import os

import pytest

from ragmed_core.ratelimit import MemoryRateLimiter, ValkeyRateLimiter

pytestmark = pytest.mark.anyio
VALKEY_URL = os.environ.get("RAGMED_TEST_VALKEY_URL")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


async def test_hits_are_allowed_up_to_the_limit() -> None:
    limiter = MemoryRateLimiter(Clock())

    decisions = [await limiter.hit("ip:1", 3, 60) for _ in range(4)]

    assert [d.allowed for d in decisions] == [True, True, True, False]
    assert decisions[-1].count == 4
    assert 0 < decisions[-1].retry_after <= 61


async def test_a_new_window_starts_fresh_and_keys_are_separate() -> None:
    clock = Clock(1_000_020.0)
    limiter = MemoryRateLimiter(clock)
    await limiter.hit("ip:1", 1, 60)

    blocked = await limiter.hit("ip:1", 1, 60)
    other = await limiter.hit("ip:2", 1, 60)
    clock.now += 60
    later = await limiter.hit("ip:1", 1, 60)

    assert not blocked.allowed
    assert other.allowed
    assert later.allowed


async def test_an_unreachable_valkey_lets_requests_through() -> None:
    limiter = ValkeyRateLimiter("redis://127.0.0.1:9/0")

    decision = await limiter.hit("ip:1", 1, 60)

    assert decision.allowed
    await limiter.close()


@pytest.mark.skipif(not VALKEY_URL, reason="set RAGMED_TEST_VALKEY_URL to test a Valkey server")
async def test_valkey_counts_across_instances() -> None:
    clock = Clock()
    first = ValkeyRateLimiter(VALKEY_URL or "", prefix="test-rl", clock=clock)
    second = ValkeyRateLimiter(VALKEY_URL or "", prefix="test-rl", clock=clock)
    key = f"ip:{os.getpid()}:{clock.now}"

    results = [
        await first.hit(key, 2, 60),
        await second.hit(key, 2, 60),
        await first.hit(key, 2, 60),
    ]

    assert [r.allowed for r in results] == [True, True, False]
    await first.close()
    await second.close()
