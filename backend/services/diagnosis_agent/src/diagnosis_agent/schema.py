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
    url: str = ""  # the page the answer was published on, if known


class GroundingCheck(BaseModel):
    unsupported: list[str] = Field(
        description="Sentences copied exactly from the answer that state a medical fact the "
        "passages do not support. Advice to see a clinician is always allowed; do not list it."
    )


class SearchQuery(BaseModel):
    query: str = Field(description="A self-contained medical search query.")
    condition: str = Field(
        description="The single most likely common condition, by its plain name (for example "
        "'migraine'), or empty if unclear."
    )


class Triage(BaseModel):
    """What the answer means for the person: read from the finished answer, not the passages."""

    possible_conditions: list[str] = Field(
        description="Conditions the answer connects to the symptoms; empty if none."
    )
    urgency: Urgency = Field(
        description="self_care, routine (see a doctor in days or weeks), urgent (today), "
        "or emergency (call emergency services now). Judge from the question too, even when "
        "the answer could not help."
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


class Lead(BaseModel):
    condition: str = Field(description="A possible condition, by its plain name.")
    fits: str = Field(description="Under 15 words: what in the case points to it.")
    citations: list[int] = Field(description="Numbers of the passages that describe it.")


class ConsultDraft(BaseModel):
    """The model's working picture of an intake that is still under way."""

    leads: list[Lead] = Field(
        description="Up to 3 conditions the passages connect to the case, most likely first; "
        "empty if none fit."
    )
    ask_next: list[str] = Field(
        description="Up to 3 facts not yet known that would best tell the leads apart or "
        "settle the urgency, most useful first, as short plain topics (for example 'whether "
        "light makes it worse'), not questions."
    )
    warning_signs: list[str] = Field(
        description="Up to 3 warning signs not yet asked about that would make this urgent, "
        "each in plain words and under 10 words (for example 'sudden, worst-ever headache')."
    )
    urgency: Urgency = Field(description="The working urgency from what is known so far.")
    enough: bool = Field(
        description="True if more questions would not change the leads, the urgency or who "
        "should see them."
    )


class Citation(BaseModel):
    number: int
    answer_id: str
    question: str
    excerpt: str = ""  # the passage text the answer relied on, so readers can check it
    url: str = ""  # the page it comes from, so readers can read it in full


class ConsultLead(BaseModel):
    condition: str
    fits: str
    sources: list[Citation]


class ConsultResult(BaseModel):
    """The `consult` skill's output: grounded leads that guide the next intake question."""

    leads: list[ConsultLead]
    ask_next: list[str]
    warning_signs: list[str]
    urgency: Urgency
    enough: bool
    search_query: str
    passages_read: int


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
