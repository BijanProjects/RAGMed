from typing import Any

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mcp_scheduling.app import SchedulingSettings, create_server, make_provider
from mcp_scheduling.providers import InMemoryProvider
from mcp_scheduling.server import MAX_BOOKINGS_PER_DAY, attendee_email, build_server
from mcp_scheduling.store import AppointmentStore
from ragmed_core.settings import Environment

pytestmark = pytest.mark.anyio
CLINICIAN = "k" * 40
SESSION = "session-123456"
REPORT = {"summary": "Headache for three days", "urgency": "routine"}


@pytest.fixture
def calendar() -> InMemoryProvider:
    return InMemoryProvider()


@pytest.fixture
def server(calendar: InMemoryProvider) -> Any:
    return build_server(
        calendar,
        AppointmentStore(":memory:"),
        time_zone="America/St_Johns",
        clinician_tokens={"clinic": CLINICIAN},
    )


async def call(server: Any, tool: str, **arguments: Any) -> Any:
    async with Client(server) as client:
        result = await client.call_tool(tool, arguments)
    content = result.structured_content or {}
    return content.get("result", content)


async def first_slot(server: Any, kind: str = "Neurology", minutes: int = 30) -> dict[str, Any]:
    slots = await call(server, "find_slots", clinician_type=kind, visit_minutes=minutes)
    assert slots
    first: dict[str, Any] = slots[0]
    return first


async def book(server: Any, slot: dict[str, Any], session: str = SESSION) -> dict[str, Any]:
    booked: dict[str, Any] = await call(
        server, "book_visit", clinician_type=slot["clinician_type"],
        visit_minutes=slot["visit_minutes"], start=slot["start"], session_id=session,
        urgency="routine", report=REPORT,
    )  # fmt: skip
    return booked


async def test_lists_every_clinician_type(server: Any) -> None:
    types = await call(server, "list_clinician_types")

    names = [t["clinician_type"] for t in types]
    assert names[0] == "General physician"
    assert "Neurology" in names
    assert len(names) == 20


async def test_slots_are_spread_over_days_with_a_physician(server: Any) -> None:
    slots = await call(server, "find_slots", clinician_type="neurology", visit_minutes=15)

    days = {s["start"][:10] for s in slots}
    assert all(sum(s["start"].startswith(d) for s in slots) <= 4 for d in days)
    assert all(s["physician"] in ("Dr. Isabel Cohen", "Dr. Daniel Osei") for s in slots)
    assert all(s["start"].endswith("-03:30") or s["start"].endswith("-02:30") for s in slots)


@pytest.mark.parametrize(
    "arguments",
    [
        {"clinician_type": "astrology", "visit_minutes": 15},
        {"clinician_type": "Neurology", "visit_minutes": 45},
        {"clinician_type": "Neurology", "visit_minutes": 15, "days_ahead": 30},
    ],
)
async def test_find_slots_validates_its_input(server: Any, arguments: dict[str, Any]) -> None:
    with pytest.raises(ToolError):
        await call(server, "find_slots", **arguments)


async def test_booking_stores_the_report_and_takes_the_slot(server: Any) -> None:
    slot = await first_slot(server)

    booked = await book(server, slot)
    again = await call(server, "get_appointment", reference=booked["reference"])
    later = await call(server, "find_slots", clinician_type="Neurology", visit_minutes=30)

    assert booked["reference"].startswith("RC-")
    assert booked["status"] == "booked"
    assert booked["physician"] == slot["physician"]
    assert "session_id" not in booked  # internal ids stay inside the server
    assert again["report"] == REPORT
    assert slot["start"] not in {s["start"] for s in later}


async def test_the_calendar_only_ever_sees_an_anonymous_patient(
    server: Any, calendar: InMemoryProvider
) -> None:
    booked = await book(server, await first_slot(server))

    (attendee, metadata, notes) = calendar.received[0]
    reference = booked["reference"]
    assert attendee.name == f"Patient {reference}"
    assert attendee.email == f"patient-{reference.lower()}@example.com"
    assert metadata["reference"] == reference
    assert "not a diagnosis" in metadata["source"]
    assert reference in notes


async def test_a_repeated_request_returns_the_first_booking(server: Any) -> None:
    slot = await first_slot(server)

    first = await book(server, slot)
    second = await book(server, slot)

    assert first["reference"] == second["reference"]


async def test_a_chat_can_book_only_two_visits_a_day(server: Any) -> None:
    slots = await call(server, "find_slots", clinician_type="General physician", visit_minutes=15)
    for slot in slots[:MAX_BOOKINGS_PER_DAY]:
        await book(server, slot)

    with pytest.raises(ToolError, match="at most 2"):
        await book(server, slots[MAX_BOOKINGS_PER_DAY])


async def test_only_the_booking_session_can_cancel(server: Any) -> None:
    booked = await book(server, await first_slot(server))
    reference = booked["reference"]

    with pytest.raises(ToolError, match="for this conversation"):
        await call(server, "cancel_appointment", reference=reference, session_id="someone-else")
    cancelled = await call(server, "cancel_appointment", reference=reference, session_id=SESSION)
    repeat = await call(server, "cancel_appointment", reference=reference, session_id=SESSION)

    assert cancelled["status"] == "cancelled"
    assert repeat["status"] == "cancelled"


async def test_booking_validates_its_input(server: Any) -> None:
    slot = await first_slot(server)

    with pytest.raises(ToolError, match="time zone"):
        await book(server, {**slot, "start": slot["start"][:19]})
    with pytest.raises(ToolError):
        await book(server, {**slot, "clinician_type": "astrology"})
    with pytest.raises(ToolError, match="no longer available"):
        await book(server, {**slot, "start": "2020-01-06T09:00:00-03:30"})


async def get_clinician(server: Any, token: str | None) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    transport = httpx.ASGITransport(app=server.http_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://s") as http:
        return await http.get("/clinician/appointments", headers=headers)


async def test_clinician_page_needs_its_token_and_lists_urgent_first(server: Any) -> None:
    slots = await call(server, "find_slots", clinician_type="General physician", visit_minutes=15)
    for slot, urgency, session in [
        (slots[0], "routine", "session-aaaaaaa"),
        (slots[1], "urgent", "session-bbbbbbb"),
    ]:
        await call(
            server, "book_visit", clinician_type=slot["clinician_type"], visit_minutes=15,
            start=slot["start"], session_id=session, urgency=urgency, report=REPORT,
        )  # fmt: skip

    refused = await get_clinician(server, None)
    wrong = await get_clinician(server, "x" * 40)
    allowed = await get_clinician(server, CLINICIAN)

    assert refused.status_code == 401
    assert wrong.status_code == 401
    assert [a["urgency"] for a in allowed.json()] == ["urgent", "routine"]
    assert allowed.json()[0]["report"] == REPORT


def test_production_requires_service_tokens(tmp_path: Any) -> None:
    settings = SchedulingSettings(
        environment=Environment.PRODUCTION, appointments_db=tmp_path / "a.db"
    )

    with pytest.raises(RuntimeError, match="RAGMED_SERVICE_TOKENS"):
        create_server(settings, InMemoryProvider())


def test_bookings_use_a_tagged_clinic_address_never_the_patients() -> None:
    assert attendee_email("RC-7K2Q9F", None) == "patient-rc-7k2q9f@example.com"
    assert attendee_email("RC-7K2Q9F", "clinic@gmail.com") == "clinic+rc-7k2q9f@gmail.com"
    assert attendee_email("RC-7K2Q9F", "clinic+old@gmail.com") == "clinic+rc-7k2q9f@gmail.com"


async def test_calcom_mode_needs_a_clinic_inbox() -> None:
    with pytest.raises(RuntimeError, match="RAGMED_CLINIC_EMAIL"):
        await make_provider(SchedulingSettings(calendar="calcom", clinic_email=None))
