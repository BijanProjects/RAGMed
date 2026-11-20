import json
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest

from mcp_scheduling.providers import (
    Attendee,
    CalComProvider,
    InMemoryProvider,
    ProviderError,
    discover_event_types,
)
from mcp_scheduling.roster import GENERAL, ROSTER, normalise, physician_for

pytestmark = pytest.mark.anyio
ZONE = ZoneInfo("America/St_Johns")
MONDAY = datetime(2026, 11, 23, 0, 0, tzinfo=ZONE)
PATIENT = Attendee("Patient RC-TEST01", "patient-rc-test01@example.com", "America/St_Johns")


def test_clinician_types_are_matched_loosely() -> None:
    assert normalise("general PHYSICIAN") == GENERAL
    assert normalise("neuro") == "Neurology"
    with pytest.raises(ValueError, match="list_clinician_types"):
        normalise("astrologer")


def test_physician_choice_is_stable_and_on_the_team() -> None:
    first = physician_for("Neurology", "2026-11-23T09:00")
    assert first == physician_for("Neurology", "2026-11-23T09:00")
    assert first in ROSTER["Neurology"]


async def test_memory_calendar_offers_weekday_office_hours() -> None:
    calendar = InMemoryProvider(now=MONDAY - timedelta(days=3))

    slots = await calendar.find_slots(30, MONDAY - timedelta(days=2), MONDAY + timedelta(days=1))

    assert slots  # Saturday and Sunday give nothing, Monday gives the day
    assert all(s.start.weekday() == 0 for s in slots)
    assert slots[0].start.hour == 9
    assert slots[-1].end.hour == 17
    assert all(s.end - s.start == timedelta(minutes=30) for s in slots)


async def test_a_booked_time_is_no_longer_offered_until_cancelled() -> None:
    calendar = InMemoryProvider(now=MONDAY - timedelta(days=1))
    nine = MONDAY.replace(hour=9)
    window = (MONDAY, MONDAY + timedelta(days=1))

    booking = await calendar.book(30, nine, PATIENT, {}, "")
    after_booking = await calendar.find_slots(15, *window)
    await calendar.cancel(booking.uid, "test")
    after_cancel = await calendar.find_slots(15, *window)

    assert nine not in {s.start for s in after_booking}
    assert nine.replace(minute=15) not in {s.start for s in after_booking}
    assert nine in {s.start for s in after_cancel}


async def test_memory_calendar_refuses_a_taken_or_invalid_time() -> None:
    calendar = InMemoryProvider(now=MONDAY - timedelta(days=1))
    nine = MONDAY.replace(hour=9)
    await calendar.book(15, nine, PATIENT, {}, "")

    with pytest.raises(ProviderError, match="no longer available"):
        await calendar.book(15, nine, PATIENT, {}, "")
    with pytest.raises(ProviderError):
        await calendar.cancel("missing", "test")


def cal(handler: Any) -> CalComProvider:
    return CalComProvider(
        "cal_live_test", {15: 1501, 30: 3001}, "America/St_Johns",
        transport=httpx.MockTransport(handler),
    )  # fmt: skip


async def test_calcom_slots_use_the_pinned_version_and_parse_both_shapes() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"status": "success", "data": {
            "2026-11-23": [
                {"start": "2026-11-23T09:30:00.000-03:30", "end": "2026-11-23T10:00:00.000-03:30"}
            ],
            "2026-11-24": ["2026-11-24T10:00:00.000-03:30"],
        }})  # fmt: skip

    slots = await cal(handler).find_slots(30, MONDAY, MONDAY + timedelta(days=2))

    request = seen[0]
    assert request.headers["cal-api-version"] == "2024-09-04"
    assert request.headers["authorization"] == "Bearer cal_live_test"
    assert request.url.params["eventTypeId"] == "3001"
    assert request.url.params["format"] == "range"
    assert [s.start.hour for s in slots] == [9, 10]
    assert slots[1].end - slots[1].start == timedelta(minutes=30)


async def test_calcom_booking_sends_anonymous_attendee_and_trimmed_metadata() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        assert request.headers["cal-api-version"] == "2026-02-25"
        return httpx.Response(201, json={"status": "success", "data": {
            "uid": "abc", "start": "2026-11-23T12:30:00.000Z",
            "end": "2026-11-23T12:45:00.000Z", "status": "accepted",
        }})  # fmt: skip

    booking = await cal(handler).book(
        15, MONDAY.replace(hour=9), PATIENT, {"note": "x" * 900, "k" * 60: "v"}, "summary"
    )

    body = bodies[0]
    assert body["start"] == "2026-11-23T12:30:00Z"  # sent in UTC
    assert body["eventTypeId"] == 1501
    assert body["attendee"]["email"].endswith("@example.com")
    assert len(body["metadata"]["note"]) == 500
    assert all(len(key) <= 40 for key in body["metadata"])
    assert body["bookingFieldsResponses"]["notes"] == "summary"
    assert booking.uid == "abc"
    assert booking.start.tzinfo is not None


async def test_calcom_errors_become_readable_provider_errors() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"status": "error", "error": {"message": "slot taken"}})

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    with pytest.raises(ProviderError, match="slot taken"):
        await cal(refuse).book(15, MONDAY, PATIENT, {}, "")
    with pytest.raises(ProviderError, match="not reachable"):
        await cal(down).cancel("abc", "test")
    with pytest.raises(ProviderError, match="45-minute"):
        await cal(refuse).find_slots(45, MONDAY, MONDAY)


async def test_event_types_are_discovered_from_visible_15_and_30_minute_types() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "success", "data": [
            {"id": 9, "lengthInMinutes": 15, "hidden": True},
            {"id": 10, "lengthInMinutes": 15, "hidden": False},
            {"id": 11, "lengthInMinutes": 30, "hidden": False},
            {"id": 12, "lengthInMinutes": 60, "hidden": False},
        ]})  # fmt: skip

    assert await discover_event_types("k", transport=httpx.MockTransport(handler)) == {
        15: 10,
        30: 11,
    }


async def test_missing_event_types_are_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": 1, "lengthInMinutes": 15}]})

    with pytest.raises(ProviderError, match=r"\[30\]"):
        await discover_event_types("k", transport=httpx.MockTransport(handler))
