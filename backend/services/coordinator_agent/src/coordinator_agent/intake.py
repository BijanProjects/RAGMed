"""The intake interview: one adaptive question at a time, like a nurse before a visit."""

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
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
renewal or a note, ask only what that needs. Skip anything already answered or irrelevant:
the person's first message is their main problem, so never ask for it again, and read it for
details they already gave (for example "for two days" answers when it started).
Never repeat a question from "Questions already asked". If an answer did not address your
question, you may rephrase it once; after that, move on to something else.

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
        asked = [
            line[len("assistant: ") :]
            for line in transcript.splitlines()
            if line.startswith("assistant: ")
        ]
        asked_text = "\n".join(f"- {q}" for q in asked) or "(none)"
        user = (
            f"Known so far:\n{known}\n\nQuestions already asked:\n{asked_text}\n\n"
            f"Interview so far:\n{transcript}"
        )
        return await ask_structured(self._models, IntakeTurn, INTERVIEW, user)


NON_ANSWERS = {
    "unknown", "not sure", "n/a", "na", "none given", "not provided", "unclear",
    "none mentioned", "not mentioned", "not asked", "not specified", "not stated",
}  # fmt: skip
SINGLE_VALUE: set[str] = {"main_problem", "age", "sex"}  # the first answer stands
MAX_REPEATS = 2
SIMILAR = 0.65  # rewordings measured 0.62-0.92 apart; distinct questions at most 0.61


def merge(case: Mapping[str, str], updates: Sequence[CaseUpdate]) -> dict[str, str]:
    """Add new facts; a later answer about the same thing is appended, never lost."""
    merged = dict(case)
    for update in updates:
        value = update.value.strip()
        meaningless = not any(c.isalnum() for c in value)  # "?", "-", "..."
        if not value or meaningless or value.casefold().strip(" .") in NON_ANSWERS:
            continue
        previous = merged.get(update.field)
        if previous and update.field in SINGLE_VALUE:
            continue
        if previous and value.casefold() in previous.casefold():
            continue  # already recorded: keep everything we have
        merged[update.field] = f"{previous}; {value}" if previous else value
    return merged


def case_lines(case: Mapping[str, str]) -> str:
    return "\n".join(f"- {LABELS[k]}: {v}" for k, v in case.items() if k in LABELS and v)


def case_summary(case: Mapping[str, str]) -> str:
    """The case as text for the Diagnosis agent."""
    return (
        "Patient-reported information gathered before a clinic visit (not verified):\n"
        f"{case_lines(case)}\n\n"
        "Using only the passages, describe which conditions could fit this presentation and how "
        "urgent it is. Prefer common conditions over rare ones unless the details point to a "
        "rare one. Then choose the visit:\n"
        "- clinician_type: 'Pediatrics' for a child under 16; otherwise 'General physician' "
        "for common or unclear problems, prescription renewals, notes and forms, and a "
        "specialty only when the problem clearly belongs to one. "
        "It must match the clinician your answer recommends.\n"
        "- visit_minutes: 15 for one straightforward problem, a prescription renewal or a note; "
        "30 for complex, severe, long-lasting or several problems."
    )


def quick_replies(replies: Sequence[str]) -> list[str]:
    cleaned = [r.strip() for r in replies if r.strip() and len(r.strip()) <= MAX_REPLY_CHARS]
    return cleaned[:MAX_REPLIES]


def _normal(question: str) -> str:
    return " ".join("".join(c for c in question.casefold() if c.isalnum() or c == " ").split())


def is_repeat(question: str, asked: Sequence[str]) -> bool:
    """True if a question like this one was already asked `MAX_REPEATS` times.

    Similar wording counts as the same question ("How long does each episode usually last?"
    and "How long does each episode last, from start to finish?").
    """
    wanted = _normal(question)
    similar = sum(
        SequenceMatcher(None, wanted, _normal(q)).ratio() >= SIMILAR for q in asked if q.strip()
    )
    return similar >= MAX_REPEATS


ADMIN_WORDS = ("renew", "refill", "repeat prescription", "prescription", "sick note", "note for",
               "doctor's note", "certificate", "form")  # fmt: skip


def is_straightforward_request(case: Mapping[str, str]) -> bool:
    """A prescription renewal, a note or a form: the clinic books these as short GP visits."""
    text = f"{case.get('main_problem', '')} {case.get('visit_goal', '')}".casefold()
    return any(word in text for word in ADMIN_WORDS)
