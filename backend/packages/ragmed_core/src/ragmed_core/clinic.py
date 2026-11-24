"""RAGMed Clinic: the facts every service shares about the (hypothetical) clinic."""

from typing import Literal

CLINIC_NAME = "RAGMed Clinic"

# What the Diagnosis agent may recommend and the Scheduling server can book.
ClinicianType = Literal[
    "General physician",
    "Allergy and immunology",
    "Cardiology",
    "Dermatology",
    "Endocrinology",
    "Gastroenterology",
    "Hematology",
    "Infectious disease",
    "Nephrology",
    "Neurology",
    "Obstetrics and gynecology",
    "Oncology",
    "Ophthalmology",
    "Orthopedics",
    "Otolaryngology (ENT)",
    "Pediatrics",
    "Psychiatry",
    "Pulmonology",
    "Rheumatology",
    "Urology",
]
CLINICIAN_TYPES: tuple[str, ...] = ClinicianType.__args__  # type: ignore[attr-defined]
GENERAL_PHYSICIAN: ClinicianType = "General physician"

# 15 minutes: straightforward problems, notes, prescriptions. 30: complex or several problems.
VisitMinutes = Literal[15, 30]

DRAFT_NOTICE = (
    "This is not a diagnosis. It is information gathered before your visit, as a draft for "
    "your clinician, and it may be incomplete or wrong. Your clinician will assess you."
)

# The care team, as progress updates name them: who is acting at each step of a conversation.
COORDINATOR_AGENT = "Care Coordinator"
DIAGNOSIS_AGENT = "Diagnosis agent"
KNOWLEDGE_BASE = "Knowledge base"
SCHEDULING_AGENT = "Scheduling"
