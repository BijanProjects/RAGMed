"""The Scheduling MCP server: find visit slots and book them, with the clinic's safeguards.

Safeguards: only the clinic's clinician types and the two visit lengths; at most
`MAX_BOOKINGS_PER_DAY` bookings per chat session per day; a repeated booking request returns
the first booking instead of creating another; only the session that booked can cancel.
Patients are anonymous: a booking carries a reference code, never a name or email.
"""

import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import TokenVerifier
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_scheduling.providers import Attendee, ProviderError, SchedulingProvider
from mcp_scheduling.roster import CLINICIAN_TYPES, ROSTER, normalise, physician_for
from mcp_scheduling.store import Appointment, AppointmentStore
from ragmed_core.tokens import match_caller

MAX_BOOKINGS_PER_DAY = 2
SLOTS_PER_DAY = 4
ATTENDEE_DOMAIN = "example.com"  # reserved for examples (RFC 2606): nothing is ever delivered

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)
WRITES = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)

Minutes = Annotated[
    Literal[15, 30], Field(description="15 for straightforward visits, 30 for complex ones.")
]
ClinicianType = Annotated[str, Field(description="A clinician type from list_clinician_types.")]


def new_reference() -> str:
    """A short, unguessable visit reference such as RC-7K2Q9F."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I
    return "RC-" + "".join(secrets.choice(alphabet) for _ in range(6))


def _spread(slots: list[Any], per_day: int) -> list[Any]:
    """Up to `per_day` slots per day, spaced across the day rather than the first few."""
    by_day: dict[str, list[Any]] = {}
    for slot in slots:
        by_day.setdefault(slot.start.date().isoformat(), []).append(slot)
    chosen = []
    for day_slots in by_day.values():
        step = max(1, len(day_slots) // per_day)
        chosen.extend(day_slots[::step][:per_day])
    return chosen


def build_server(  # noqa: PLR0915 - one closure per tool, next to the data it uses
    provider: SchedulingProvider,
    store: AppointmentStore,
    *,
    time_zone: str,
    auth: TokenVerifier | None = None,
    clinician_tokens: dict[str, str] | None = None,
) -> FastMCP:
    zone = ZoneInfo(time_zone)
    mcp = FastMCP(
        "ragmed-scheduling",
        instructions="Find and book visits at RAGMed Clinic. Patients are anonymous.",
        auth=auth,
    )

    @mcp.tool(annotations=READ_ONLY)
    def list_clinician_types() -> list[dict[str, Any]]:
        """The clinic's clinician types and how many physicians each has."""
        return [{"clinician_type": t, "physicians": len(ROSTER[t])} for t in CLINICIAN_TYPES]

    @mcp.tool(annotations=READ_ONLY)
    async def find_slots(
        clinician_type: ClinicianType,
        visit_minutes: Minutes,
        days_ahead: Annotated[int, Field(ge=1, le=14)] = 7,
    ) -> list[dict[str, Any]]:
        """Open visit times in the coming days, a few per day, each with its physician."""
        try:
            kind = normalise(clinician_type)
        except ValueError as error:
            raise ToolError(str(error)) from error
        now = datetime.now(UTC)
        try:
            slots = await provider.find_slots(visit_minutes, now, now + timedelta(days=days_ahead))
        except ProviderError as error:
            raise ToolError(str(error)) from error
        return [
            {
                "start": s.start.astimezone(zone).isoformat(),
                "end": s.end.astimezone(zone).isoformat(),
                "physician": physician_for(kind, s.start.isoformat()),
                "clinician_type": kind,
                "visit_minutes": visit_minutes,
            }
            for s in _spread(slots, SLOTS_PER_DAY)
        ]

    @mcp.tool(annotations=WRITES)
    async def book_visit(  # noqa: PLR0917 - MCP passes tool arguments by name
        clinician_type: ClinicianType,
        visit_minutes: Minutes,
        start: Annotated[str, Field(description="A slot start exactly as find_slots gave it.")],
        session_id: Annotated[str, Field(min_length=8, max_length=100)],
        urgency: Literal["self_care", "routine", "urgent"],
        report: Annotated[dict[str, Any], Field(description="The pre-visit report.")],
    ) -> dict[str, Any]:
        """Book a visit and attach the patient's pre-visit report for the physician."""
        try:
            kind = normalise(clinician_type)
            begins = datetime.fromisoformat(start)
        except ValueError as error:
            raise ToolError(str(error)) from error
        if begins.tzinfo is None:
            raise ToolError("start must include a time zone, as find_slots returns it")

        for existing in await store.upcoming():  # a retried request returns the first booking
            same_time = datetime.fromisoformat(existing.start) == begins  # compare instants
            if existing.session_id == session_id and same_time:
                return existing.public()
        if await store.bookings_today(session_id) >= MAX_BOOKINGS_PER_DAY:
            raise ToolError(f"A chat can book at most {MAX_BOOKINGS_PER_DAY} visits a day.")

        reference = new_reference()
        physician = physician_for(kind, begins.isoformat())
        attendee = Attendee(
            name=f"Patient {reference}",
            email=f"patient-{reference.lower()}@{ATTENDEE_DOMAIN}",
            time_zone=time_zone,
        )
        summary = str(report.get("summary", ""))[:900]
        try:
            booking = await provider.book(
                visit_minutes,
                begins,
                attendee,
                {
                    "reference": reference,
                    "clinician_type": kind,
                    "physician": physician,
                    "urgency": urgency,
                    "source": "RAGMed Clinic intake (draft report, not a diagnosis)",
                },
                f"RAGMed {reference} | {kind} | {physician} | urgency: {urgency}\n{summary}",
            )
        except ProviderError as error:
            raise ToolError(str(error)) from error
        appointment = Appointment(
            reference=reference,
            booking_uid=booking.uid,
            session_id=session_id,
            start=booking.start.astimezone(zone).isoformat(),
            end=booking.end.astimezone(zone).isoformat(),
            minutes=visit_minutes,
            clinician_type=kind,
            physician=physician,
            urgency=urgency,
            status="booked",
            report=report,
            created_at=datetime.now(UTC).isoformat(),
        )
        await store.add(appointment)
        return appointment.public()

    @mcp.tool(annotations=READ_ONLY)
    async def get_appointment(reference: str) -> dict[str, Any]:
        """One appointment by its reference."""
        appointment = await store.get(reference)
        if appointment is None:
            raise ToolError(f"no appointment {reference!r}")
        return appointment.public()

    @mcp.tool(annotations=WRITES)
    async def cancel_appointment(
        reference: str, session_id: str, reason: str = "Cancelled by the patient"
    ) -> dict[str, Any]:
        """Cancel a visit. Only the chat session that booked it can cancel it."""
        appointment = await store.get(reference)
        if appointment is None or appointment.session_id != session_id:
            raise ToolError(f"no appointment {reference!r} for this conversation")
        if appointment.status == "cancelled":
            return appointment.public()
        try:
            await provider.cancel(appointment.booking_uid, reason)
        except ProviderError as error:
            raise ToolError(str(error)) from error
        await store.set_status(reference, "cancelled")
        updated = await store.get(reference)
        assert updated is not None  # noqa: S101 - it was just read
        return updated.public()

    @mcp.custom_route("/clinician/appointments", methods=["GET"])
    async def clinician_appointments(request: Request) -> JSONResponse:
        """Upcoming visits with their reports, most urgent first. Clinician token required."""
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if (
            not clinician_tokens
            or scheme.lower() != "bearer"
            or not match_caller(token.strip(), clinician_tokens)
        ):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return JSONResponse([a.public() for a in await store.upcoming()])

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return mcp
