"""The intake interview: one adaptive question at a time, like a nurse before a visit."""

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from ragmed_llm import ask_structured

MAX_QUESTIONS = 10
MAX_REPLIES = 4
MAX_REPLY_CHARS = 40
FINAL_QUESTION = (
    "Is there anything else you would like the clinician to know, such as worries, other "
    "symptoms, or what you hope the visit will cover?"
)

CaseField = Literal[
    "main_problem",
    "onset",
    "duration",
    "location",
    "character",
    "severity",
    "pattern",
    "worse_with",
    "better_with",
    "other_symptoms",
    "age",
    "sex",
    "pregnancy",
    "conditions",
    "medications",
    "allergies",
    "exposures",
    "visit_goal",
    "other_notes",
]

LABELS: dict[str, str] = {
    "main_problem": "Main problem",
    "onset": "Started",
    "duration": "Duration",
    "location": "Where",
    "character": "What it feels like",
    "severity": "Severity (0-10)",
    "pattern": "Pattern",
    "worse_with": "Worse with",
    "better_with": "Better with",
    "other_symptoms": "Other symptoms",
    "age": "Age",
    "sex": "Sex",
    "pregnancy": "Pregnancy",
    "conditions": "Existing conditions",
    "medications": "Medications",
    "allergies": "Allergies",
    "exposures": "Exposures or travel",
    "visit_goal": "Hopes for the visit",
    "other_notes": "Other notes",
}

INTERVIEW = """You are the intake assistant at RAGMed Clinic. Before a visit, you gather what the
clinician needs, the way a nurse would. You do NOT diagnose or advise during intake.

Ask ONE short, clear question at a time, in plain language, adapted to the person's problem.
Over the interview, cover what is relevant: the main problem; when it started and how long it
has lasted; where it is; what it feels like; how severe it is from 0 to 10; whether it comes and
goes; what makes it worse or better; other symptoms; age; sex, and pregnancy only if relevant;
existing conditions; medications; allergies; relevant exposures or travel. For a prescription
renewal or a note, ask only what that needs. Skip anything already answered or irrelevant.

From the person's latest answer, record every fact you learned in `updates` (one field each,
in their words, short). Set `complete` to true when a clinician would have enough for a
pre-visit note, usually after 4 to 8 questions; then leave `next_question` empty. Offer up to 4
short `quick_replies` when natural (for example "Yes", "No", "Less than a day")."""


class CaseUpdate(BaseModel):
    field: CaseField
    value: str = Field(description="What the person said, briefly.")


class IntakeTurn(BaseModel):
    updates: list[CaseUpdate]
    complete: bool
    next_question: str = Field(description="The one question to ask next; empty if complete.")
    quick_replies: list[str] = Field(description="Up to 4 short suggested answers.")


class Interviewer(Protocol):
    async def next_turn(self, case: Mapping[str, str], transcript: str) -> IntakeTurn: ...


class LlmInterviewer:
    def __init__(self, models: Sequence[BaseChatModel]) -> None:
        self._models = list(models)

    async def next_turn(self, case: Mapping[str, str], transcript: str) -> IntakeTurn:
        known = case_lines(case) or "(nothing yet)"
        user = f"Known so far:\n{known}\n\nInterview so far:\n{transcript}"
        return await ask_structured(self._models, IntakeTurn, INTERVIEW, user)


def merge(case: Mapping[str, str], updates: Sequence[CaseUpdate]) -> dict[str, str]:
    """Add new facts; a later answer about the same thing is appended, never lost."""
    merged = dict(case)
    for update in updates:
        value = update.value.strip()
        if not value:
            continue
        previous = merged.get(update.field)
        merged[update.field] = (
            value if not previous or value in previous else f"{previous}; {value}"
        )
    return merged


def case_lines(case: Mapping[str, str]) -> str:
    return "\n".join(f"- {LABELS[k]}: {v}" for k, v in case.items() if k in LABELS and v)


def case_summary(case: Mapping[str, str]) -> str:
    """The case as text for the Diagnosis agent."""
    return (
        "Patient-reported information gathered before a clinic visit (not verified):\n"
        f"{case_lines(case)}\n\n"
        "Using only the passages, describe which conditions could fit this presentation, how "
        "urgent it is, which clinician at the clinic should see them and for how long."
    )


def quick_replies(replies: Sequence[str]) -> list[str]:
    cleaned = [r.strip() for r in replies if r.strip() and len(r.strip()) <= MAX_REPLY_CHARS]
    return cleaned[:MAX_REPLIES]
