"""RAGMed Clinic: a hypothetical clinic with general physicians and every major specialty.

All names are fictional.
"""

import hashlib
from typing import Literal

from ragmed_core.clinic import CLINICIAN_TYPES, GENERAL_PHYSICIAN

GENERAL = GENERAL_PHYSICIAN

VisitMinutes = Literal[15, 30]

__all__ = ["CLINICIAN_TYPES", "GENERAL", "ROSTER", "VisitMinutes", "normalise", "physician_for"]

ROSTER: dict[str, tuple[str, ...]] = {
    GENERAL: ("Dr. Amara Okafor", "Dr. Liam Fournier", "Dr. Priya Raman", "Dr. Sofia Lindqvist"),
    "Allergy and immunology": ("Dr. Kenji Watanabe",),
    "Cardiology": ("Dr. Elena Marquez", "Dr. Samuel Achebe"),
    "Dermatology": ("Dr. Hannah Brooks",),
    "Endocrinology": ("Dr. Omar Haddad",),
    "Gastroenterology": ("Dr. Chloe Martin",),
    "Hematology": ("Dr. Ravi Menon",),
    "Infectious disease": ("Dr. Grace Mwangi",),
    "Nephrology": ("Dr. Lucas Petrov",),
    "Neurology": ("Dr. Isabel Cohen", "Dr. Daniel Osei"),
    "Obstetrics and gynecology": ("Dr. Fatima Noor",),
    "Oncology": ("Dr. Thomas Berg",),
    "Ophthalmology": ("Dr. Mei Lin",),
    "Orthopedics": ("Dr. Jonah Fraser",),
    "Otolaryngology (ENT)": ("Dr. Aisha Bello",),
    "Pediatrics": ("Dr. Noah Tremblay", "Dr. Leila Karimi"),
    "Psychiatry": ("Dr. Martin Novak",),
    "Pulmonology": ("Dr. Yara Haddad",),
    "Rheumatology": ("Dr. Clara Rossi",),
    "Urology": ("Dr. Ethan Walsh",),
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
