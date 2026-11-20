"""Appointments and their pre-visit reports, kept for the clinician page (SQLite)."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio.to_thread

_SCHEMA = """
CREATE TABLE IF NOT EXISTS appointments (
    reference TEXT PRIMARY KEY,
    booking_uid TEXT NOT NULL,
    session_id TEXT NOT NULL,
    start TEXT NOT NULL,
    end TEXT NOT NULL,
    minutes INTEGER NOT NULL,
    clinician_type TEXT NOT NULL,
    physician TEXT NOT NULL,
    urgency TEXT NOT NULL,
    status TEXT NOT NULL,
    report TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


@dataclass(frozen=True, slots=True)
class Appointment:
    reference: str
    booking_uid: str
    session_id: str
    start: str
    end: str
    minutes: int
    clinician_type: str
    physician: str
    urgency: str
    status: str
    report: dict[str, Any]
    created_at: str

    def public(self) -> dict[str, Any]:
        """Everything except internal ids, for tools and the clinician page."""
        data = asdict(self)
        data.pop("session_id")
        return data


class AppointmentStore:
    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._memory = (
            sqlite3.connect(":memory:", check_same_thread=False)
            if (self._path == ":memory:")
            else None
        )
        with self._connect() as db:
            db.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = self._memory or sqlite3.connect(self._path)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            if self._memory is None:
                db.close()

    def _row(self, row: sqlite3.Row) -> Appointment:
        values = dict(row)
        values["report"] = json.loads(values["report"])
        return Appointment(**values)

    def _add(self, appointment: Appointment) -> None:
        values = asdict(appointment)
        values["report"] = json.dumps(appointment.report)
        with self._connect() as db:
            db.execute(
                f"INSERT INTO appointments ({', '.join(values)}) "  # noqa: S608 - fixed columns
                f"VALUES ({', '.join('?' for _ in values)})",
                list(values.values()),
            )

    async def add(self, appointment: Appointment) -> None:
        await anyio.to_thread.run_sync(self._add, appointment)

    def _get(self, reference: str) -> Appointment | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM appointments WHERE reference = ?", (reference,)
            ).fetchone()
        return self._row(row) if row else None

    async def get(self, reference: str) -> Appointment | None:
        return await anyio.to_thread.run_sync(self._get, reference)

    def _set_status(self, reference: str, status: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE appointments SET status = ? WHERE reference = ?", (status, reference)
            )

    async def set_status(self, reference: str, status: str) -> None:
        await anyio.to_thread.run_sync(self._set_status, reference, status)

    def _count_today(self, session_id: str) -> int:
        today = datetime.now(UTC).date().isoformat()
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM appointments WHERE session_id = ? AND status = 'booked' "
                "AND substr(created_at, 1, 10) = ?",
                (session_id, today),
            ).fetchone()
        return int(row[0])

    async def bookings_today(self, session_id: str) -> int:
        return await anyio.to_thread.run_sync(self._count_today, session_id)

    def _upcoming(self) -> list[Appointment]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM appointments WHERE status = 'booked' ORDER BY start"
            ).fetchall()
        return [self._row(r) for r in rows]

    async def upcoming(self) -> list[Appointment]:
        """Booked visits, most urgent first, then by time."""
        order = {"emergency": 0, "urgent": 1, "routine": 2, "self_care": 3}
        visits = await anyio.to_thread.run_sync(self._upcoming)
        return sorted(visits, key=lambda a: (order.get(a.urgency, 9), a.start))
