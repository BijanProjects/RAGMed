"""Where appointment slots come from and bookings go: Cal.com, or an in-memory calendar."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import httpx

FRIDAY = 4  # datetime.weekday(): Monday is 0


@dataclass(frozen=True, slots=True)
class Slot:
    start: datetime  # timezone-aware
    end: datetime


@dataclass(frozen=True, slots=True)
class Booking:
    uid: str
    start: datetime
    end: datetime
    status: str


@dataclass(frozen=True, slots=True)
class Attendee:
    name: str
    email: str
    time_zone: str


class ProviderError(RuntimeError):
    """The calendar refused or failed; the message is safe to show."""


class SchedulingProvider(Protocol):
    async def find_slots(self, minutes: int, start: datetime, end: datetime) -> list[Slot]: ...

    async def book(
        self,
        minutes: int,
        start: datetime,
        attendee: Attendee,
        metadata: Mapping[str, str],
        notes: str,
    ) -> Booking: ...

    async def cancel(self, uid: str, reason: str) -> None: ...


def _utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class InMemoryProvider:
    """A clinic calendar kept in memory: weekdays 09:00 to 17:00 in the clinic's time zone."""

    time_zone: str = "America/St_Johns"
    opens: time = time(9)
    closes: time = time(17)
    minimum_notice: timedelta = timedelta(hours=1)
    bookings: dict[str, Booking] = field(default_factory=dict)
    now: datetime | None = None  # fixed in tests
    # What callers sent, so tests can check exactly what a calendar would receive.
    received: list[tuple[Attendee, dict[str, str], str]] = field(default_factory=list)
    cancellations: list[tuple[str, str]] = field(default_factory=list)

    def _taken(self, start: datetime, end: datetime) -> bool:
        return any(
            b.status == "accepted" and b.start < end and start < b.end
            for b in self.bookings.values()
        )

    async def find_slots(self, minutes: int, start: datetime, end: datetime) -> list[Slot]:
        zone = ZoneInfo(self.time_zone)
        earliest = (self.now or datetime.now(UTC)) + self.minimum_notice
        slots: list[Slot] = []
        day: date = start.astimezone(zone).date()
        while datetime.combine(day, time(0), zone) < end:
            if day.weekday() <= FRIDAY:
                cursor = datetime.combine(day, self.opens, zone)
                closing = datetime.combine(day, self.closes, zone)
                while cursor + timedelta(minutes=minutes) <= closing:
                    slot_end = cursor + timedelta(minutes=minutes)
                    if (
                        start <= cursor < end
                        and cursor >= earliest
                        and not self._taken(cursor, slot_end)
                    ):
                        slots.append(Slot(cursor, slot_end))
                    cursor += timedelta(minutes=15)
            day += timedelta(days=1)
        return slots

    async def book(
        self,
        minutes: int,
        start: datetime,
        attendee: Attendee,
        metadata: Mapping[str, str],
        notes: str,
    ) -> Booking:
        end = start + timedelta(minutes=minutes)
        offered = await self.find_slots(minutes, start, end + timedelta(seconds=1))
        if not any(s.start == start for s in offered):
            raise ProviderError("That time is no longer available. Please pick another slot.")
        booking = Booking(f"mem-{len(self.bookings) + 1}", start, end, "accepted")
        self.bookings[booking.uid] = booking
        self.received.append((attendee, dict(metadata), notes))
        return booking

    async def cancel(self, uid: str, reason: str) -> None:
        booking = self.bookings.get(uid)
        if booking is None:
            raise ProviderError("No such booking.")
        self.bookings[uid] = Booking(booking.uid, booking.start, booking.end, "cancelled")
        self.cancellations.append((uid, reason))


class CalComProvider:
    """Cal.com API v2. One event type per visit length (15 and 30 minutes).

    API versions are pinned per endpoint, as Cal.com requires: without the header an endpoint
    silently falls back to an older version.
    """

    BASE_URL = "https://api.cal.com/v2"
    SLOTS_VERSION = "2024-09-04"
    BOOKINGS_VERSION = "2026-02-25"

    def __init__(
        self,
        api_key: str,
        event_type_ids: Mapping[int, int],
        time_zone: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._event_types = dict(event_type_ids)
        self._time_zone = time_zone
        self._http = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20,
            transport=transport,
        )

    def _event_type(self, minutes: int) -> int:
        if minutes not in self._event_types:
            raise ProviderError(f"No {minutes}-minute visit type is configured.")
        return self._event_types[minutes]

    async def _call(self, method: str, path: str, version: str, **kwargs: Any) -> Any:
        try:
            response = await self._http.request(
                method, path, headers={"cal-api-version": version}, **kwargs
            )
        except httpx.HTTPError as error:
            raise ProviderError("The calendar service is not reachable right now.") from error
        if response.status_code >= 400:  # noqa: PLR2004 - HTTP client and server errors
            detail = response.json().get("error", {}) if response.content else {}
            message = detail.get("message") if isinstance(detail, dict) else None
            raise ProviderError(message or f"The calendar returned {response.status_code}.")
        return response.json().get("data")

    async def find_slots(self, minutes: int, start: datetime, end: datetime) -> list[Slot]:
        data = await self._call(
            "GET",
            "/slots",
            self.SLOTS_VERSION,
            params={
                "eventTypeId": self._event_type(minutes),
                "start": _utc(start),
                "end": _utc(end),
                "timeZone": self._time_zone,
                "format": "range",
            },
        )
        slots: list[Slot] = []
        for day_slots in (data or {}).values():
            for item in day_slots:
                begin = datetime.fromisoformat(item["start"] if isinstance(item, dict) else item)
                finish = (
                    datetime.fromisoformat(item["end"])
                    if isinstance(item, dict) and "end" in item
                    else begin + timedelta(minutes=minutes)
                )
                slots.append(Slot(begin, finish))
        return sorted(slots, key=lambda s: s.start)

    async def book(
        self,
        minutes: int,
        start: datetime,
        attendee: Attendee,
        metadata: Mapping[str, str],
        notes: str,
    ) -> Booking:
        body = {
            "start": _utc(start),
            "eventTypeId": self._event_type(minutes),
            "attendee": {
                "name": attendee.name,
                "email": attendee.email,
                "timeZone": attendee.time_zone,
                "language": "en",
            },
            # Cal.com limits: 50 keys, 40-character keys, 500-character values.
            "metadata": {k[:40]: v[:500] for k, v in list(metadata.items())[:50]},
            "bookingFieldsResponses": {"notes": notes[:1000]},
        }
        data = await self._call("POST", "/bookings", self.BOOKINGS_VERSION, json=body)
        return Booking(
            uid=str(data["uid"]),
            start=datetime.fromisoformat(data["start"]),
            end=datetime.fromisoformat(data["end"]),
            status=str(data.get("status", "accepted")),
        )

    async def cancel(self, uid: str, reason: str) -> None:
        await self._call(
            "POST",
            f"/bookings/{uid}/cancel",
            self.BOOKINGS_VERSION,
            json={"cancellationReason": reason[:500]},
        )

    async def close(self) -> None:
        await self._http.aclose()


async def discover_event_types(
    api_key: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> dict[int, int]:
    """The account's visible 15- and 30-minute event types, as {minutes: event type id}."""
    async with httpx.AsyncClient(
        base_url=CalComProvider.BASE_URL,
        headers={"Authorization": f"Bearer {api_key}", "cal-api-version": "2024-06-14"},
        timeout=20,
        transport=transport,
    ) as http:
        response = await http.get("/event-types")
    if response.status_code != 200:  # noqa: PLR2004 - HTTP OK
        raise ProviderError(f"Could not list Cal.com event types ({response.status_code}).")
    found: dict[int, int] = {}
    for event in response.json().get("data") or []:
        minutes = int(event.get("lengthInMinutes") or event.get("length") or 0)
        if minutes in (15, 30) and not event.get("hidden") and minutes not in found:
            found[minutes] = int(event["id"])
    missing = {15, 30} - set(found)
    if missing:
        raise ProviderError(f"Cal.com has no visible event type of {sorted(missing)} minutes.")
    return found
