"""RAGMed Clinic: a hypothetical clinic with general physicians and every major specialty.

Every physician has a well-known placeholder name (John Doe, Joe Bloggs, Max Mustermann, ...)
so nobody mistakes them for real people.
"""

import hashlib
from typing import Literal

from ragmed_core.clinic import CLINICIAN_TYPES, GENERAL_PHYSICIAN

GENERAL = GENERAL_PHYSICIAN

VisitMinutes = Literal[15, 30]

__all__ = ["CLINICIAN_TYPES", "GENERAL", "ROSTER", "VisitMinutes", "normalise", "physician_for"]

ROSTER: dict[str, tuple[str, ...]] = {
    GENERAL: ("Dr. John Doe", "Dr. Jane Doe", "Dr. Richard Roe", "Dr. Jane Roe"),
    "Allergy and immunology": ("Dr. Joe Bloggs",),
    "Cardiology": ("Dr. Max Mustermann", "Dr. Erika Mustermann"),
    "Dermatology": ("Dr. Jean Dupont",),
    "Endocrinology": ("Dr. Mario Rossi",),
    "Gastroenterology": ("Dr. Jan Kowalski",),
    "Hematology": ("Dr. Ivan Ivanov",),
    "Infectious disease": ("Dr. Taro Yamada",),
    "Nephrology": ("Dr. Ola Nordmann",),
    "Neurology": ("Dr. Fred Nurk", "Dr. Joe Public"),
    "Obstetrics and gynecology": ("Dr. Kari Nordmann",),
    "Oncology": ("Dr. Juan Pérez",),
    "Ophthalmology": ("Dr. Hanako Yamada",),
    "Orthopedics": ("Dr. Tommy Atkins",),
    "Otolaryngology (ENT)": ("Dr. Marie Dupont",),
    "Pediatrics": ("Dr. Jan Jansen", "Dr. Sven Svensson"),
    "Psychiatry": ("Dr. A. N. Other",),
    "Pulmonology": ("Dr. Juan dela Cruz",),
    "Rheumatology": ("Dr. Anna Kowalska",),
    "Urology": ("Dr. Pat Doe",),
}


def normalise(clinician_type: str) -> str:
    """The roster's name for a clinician type, matching case and spacing loosely."""
    wanted = " ".join(clinician_type.casefold().split())
    for name in CLINICIAN_TYPES:
        if name.casefold() == wanted or name.casefold().startswith(wanted):
            return name
    raise ValueError(f"unknown clinician type {clinician_type!r}; see list_clinician_types")


def physician_for(clinician_type: str, start: str) -> str:
    """A stable choice of physician for a slot, spreading visits across the team."""
    team = ROSTER[clinician_type]
    digest = hashlib.sha256(f"{clinician_type}|{start}".encode()).digest()
    return team[digest[0] % len(team)]
