from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from coordinator_agent.app import CoordinatorSettings, create_coordinator_app
from coordinator_agent.limits import Limits, booking_limit, client_ip, message_limit, wait_text
from coordinator_agent.testing import FakeDiagnosis, FakeInterviewer, FakeRouter, FakeScheduling
from ragmed_a2a.client import RemoteAgent, Reply
from ragmed_core.ratelimit import MemoryRateLimiter

pytestmark = pytest.mark.anyio
BASE = "http://coordinator.test"


def context_with(headers: dict[str, str] | None) -> Any:
    call = SimpleNamespace(state={"headers": headers}) if headers is not None else None
    return SimpleNamespace(call_context=call)


def test_the_client_ip_comes_from_the_web_apps_header() -> None:
    assert client_ip(context_with({"x-ragmed-client-ip": " 203.0.113.9 "})) == "203.0.113.9"
    assert client_ip(context_with({})) == "unknown"
    assert client_ip(context_with(None)) == "unknown"


@pytest.mark.parametrize(
    ("seconds", "text"), [(30, "1 minute"), (600, "10 minutes"), (7200, "2 hours")]
)
def test_waits_are_readable(seconds: int, text: str) -> None:
    assert wait_text(seconds) == text


async def test_messages_are_limited_per_ip_and_per_conversation() -> None:
    limiter, limits = (
        MemoryRateLimiter(),
        Limits(messages_per_ip_per_hour=2, messages_per_conversation_per_day=3),
    )

    by_ip = [await message_limit(limiter, limits, "1.1.1.1", f"c{n}") for n in range(3)]
    by_conversation = [await message_limit(limiter, limits, f"ip{n}", "same") for n in range(4)]

    assert by_ip[:2] == [None, None]
    assert by_ip[2] is not None
    assert "message limit" in by_ip[2]
    assert by_conversation[:3] == [None, None, None]
    assert by_conversation[3] is not None


async def test_bookings_are_capped_per_ip_and_for_the_clinic() -> None:
    limiter = MemoryRateLimiter()
    limits = Limits(bookings_per_ip_per_day=1, bookings_per_clinic_per_day=2)

    first = await booking_limit(limiter, limits, "1.1.1.1")
    same_ip = await booking_limit(limiter, limits, "1.1.1.1")
    other_ip = await booking_limit(limiter, limits, "2.2.2.2")
    third_ip = await booking_limit(limiter, limits, "3.3.3.3")

    assert first is None
    assert same_ip is not None
    assert "booking limit" in same_ip
    assert other_ip is None
    assert third_ip is not None
    assert "full for today" in third_ip


async def test_the_agent_refuses_politely_once_the_limit_is_reached() -> None:
    app = create_coordinator_app(
        CoordinatorSettings(public_url=BASE),
        router=FakeRouter(intent="greeting"),
        interviewer=FakeInterviewer(),
        diagnosis=FakeDiagnosis(),
        scheduling=FakeScheduling(),
        limiter=MemoryRateLimiter(),
        limits=Limits(messages_per_ip_per_hour=1),
    )
    agent = RemoteAgent(BASE, transport=httpx.ASGITransport(app=app))

    replies = []
    for _ in range(2):
        events = [e async for e in agent.stream("hello")]
        replies.append(events[-1])

    first, second = replies
    assert isinstance(first, Reply)
    assert isinstance(second, Reply)
    assert first.state == "completed"
    assert second.state == "rejected"
    assert "emergency number" in second.text
