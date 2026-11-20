import sqlite3
from pathlib import Path

from mcp_scheduling.store import AppointmentStore


def test_an_older_database_gains_the_patient_name_column(tmp_path: Path) -> None:
    path = tmp_path / "appointments.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE appointments (reference TEXT PRIMARY KEY, booking_uid TEXT NOT NULL, "
            "session_id TEXT NOT NULL, start TEXT NOT NULL, end TEXT NOT NULL, minutes INTEGER "
            "NOT NULL, clinician_type TEXT NOT NULL, physician TEXT NOT NULL, urgency TEXT NOT "
            "NULL, status TEXT NOT NULL, report TEXT NOT NULL, created_at TEXT NOT NULL)"
        )

    AppointmentStore(path)

    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(appointments)")}
    assert "patient_name" in columns
