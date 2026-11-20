"""What the agent reads and returns. `DiagnosisAnswer` is the A2A DataPart other agents use."""

from typing import Literal

from pydantic import BaseModel, Field

from ragmed_core.clinic import ClinicianType, VisitMinutes

Urgency = Literal["self_care", "routine", "urgent", "emergency"]


class Passage(BaseModel):
    """One search result from the Medical Knowledge MCP server."""

    answer_id: str
    question: str
    qtype: str
    text: str
    score: float


class GroundingCheck(BaseModel):
    unsupported: list[str] = Field(
        description="Sentences copied exactly from the answer that state a medical fact the "
        "passages do not support. Advice to see a clinician is always allowed; do not list it."
    )


class SearchQuery(BaseModel):
    query: str = Field(description="A self-contained medical search query.")


class DiagnosisAnswer(BaseModel):
    answerable: bool = Field(
        description="True only if the passages contain enough to answer the question. Be "
        "strict: passages about a different condition, or only loosely related, are not enough."
    )
    answer: str = Field(
        description="The answer in plain language, citing passages as [1], [2], ..."
    )
    citations: list[int] = Field(description="Numbers of the passages the answer relies on.")
    possible_conditions: list[str] = Field(
        description="Conditions the passages connect to the symptoms; empty if none."
    )
    urgency: Urgency = Field(
        description="self_care, routine (see a doctor in days or weeks), urgent (today), "
        "or emergency (call emergency services now)."
    )
    recommend_visit: bool = Field(description="True if the person should see a clinician.")
    clinician_type: ClinicianType = Field(
        description="Who should see them at RAGMed Clinic. Prefer 'General physician' for "
        "common or unclear problems, prescriptions, renewals and notes; choose a specialty only "
        "when the problem clearly belongs to one (e.g. Neurology for recurring migraines with "
        "aura, Pediatrics for a young child)."
    )
    visit_minutes: VisitMinutes = Field(
        description="15 for a straightforward problem, a note or a prescription; 30 for a "
        "complex, severe, long-standing or multi-symptom problem."
    )
    visit_reason: str = Field(description="One sentence: why this clinician and visit length.")


class Citation(BaseModel):
    number: int
    answer_id: str
    question: str


class DiagnosisResult(BaseModel):
    """The agent's final output: the answer, its sources, and whether evidence was enough."""

    answer: str
    citations: list[Citation]
    possible_conditions: list[str]
    urgency: Urgency
    recommend_visit: bool
    clinician_type: ClinicianType
    visit_minutes: VisitMinutes
    visit_reason: str
    sufficient_evidence: bool
    search_query: str
