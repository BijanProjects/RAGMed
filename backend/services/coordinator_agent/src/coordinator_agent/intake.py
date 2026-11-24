"""The intake: a triage coordinator who asks only what matters, then hands over to the clinician.

Each turn, the Diagnosis agent's working picture (from the knowledge base) says what is being
considered, what would tell those possibilities apart, and which warning signs to check. The
coordinator uses it to choose the one question worth asking next, or to stop.
"""

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any, Literal, Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from ragmed_llm import ask_structured

MAX_QUESTIONS = 10
MAX_REPLIES = 4
MAX_CHOICES = 8  # options for a question where several answers can apply
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
    "negatives",
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
    "negatives": "Does not have",
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

INTERVIEW = """You are the Care Coordinator at RAGMed Clinic: the experienced person who talks
with each patient before their visit, decides what the clinician needs to know, and hands the
case over. You work with the Diagnosis agent, which searches the clinic's medical knowledge base
as you talk; its latest working picture is below. You do not diagnose or advise during intake.

Deciding what to ask
- Ask only what would change the picture: something that tells the leading possibilities
  apart, rules a warning sign in or out, or settles how urgent this is or who should see them.
  If no answer would change anything, do not ask it.
- Lean on the working picture: its "would help to know" and "warning signs" lists are your best
  next questions. If the working urgency is urgent or emergency, check the warning signs first
  and keep everything else short. Ask about warning signs plainly, without naming frightening
  conditions.
- Read everything the person has said. Never ask for something they already told you, even in
  passing ("for two days" says when it started). Their first message is their main problem.
- Background (age, medicines, conditions, allergies, pregnancy) only when it matters for this
  problem. Age usually does, for urgency and for children; medicines matter for renewals, side
  effects and bleeding. Two short, related questions may share a turn ("How old are you, and do
  you take any regular medicines?").
- For a prescription renewal or a note, ask only what that needs.
- Never repeat a question from "Questions already asked". If an answer missed your question,
  rephrase it once at most, then move on.

How to talk
- Sound like a warm, competent person, not a form. In `acknowledgement`, when it fits, react
  briefly and specifically to what they just said ("A month of that sounds exhausting.", "Good,
  that helps."). Vary it, keep it to one short sentence, leave it empty when a plain question
  is better, and never write "Thank you for sharing".
- `next_question`: one clear question in plain words. No jargon, no lists.
- `reason`: in under 15 plain words the patient could read, why this question matters (for
  example "it helps tell two kinds of headache apart"). Never name a serious condition.

When to stop
Set `complete` to true once the clinician would have enough to see this person well: the main
problem is clear, the warning signs are checked, the urgency is settled, and more questions
would not change who should see them or how soon. That is often 3 to 6 questions; a simple
request may need 1 or 2. Then leave `next_question` and `reason` empty.

Record every fact from the person's latest answer in `updates` (one field each, short, in their
words), including what they say they do NOT have, under `negatives`. Offer up to 4 short
`quick_replies` when natural (for example "Yes", "No", "Less than a day").

When several answers could apply at once (other symptoms, medicines, existing conditions,
allergies, triggers), set `multi_select` to true and offer up to 8 short options the person can
tick, chosen for this problem: for symptoms, the ones that would tell the leads apart or that
are warning signs. Do not include "None" or "Other"; the screen adds those, and the person can
always type their own answer."""

HANDOFF = """You are the Care Coordinator at RAGMed Clinic, handing a patient over to the
clinician who will see them. From the intake, the working picture and the Diagnosis agent's
assessment, write:

- impression: 2 or 3 sentences for the clinician: the presenting problem and its key features,
  the important negatives, and the possibilities the assessment raised. Not a diagnosis.
- urgency_score from 1 to 5: 1 emergency, now; 2 today; 3 within a few days; 4 routine, within a
  week or two; 5 self-care is likely enough. Move it up for severe or worsening symptoms, for
  warning signs that were not ruled out, and for higher-risk patients (infants, older adults,
  pregnancy, serious existing conditions).
- score_reason: one sentence, in plain words the patient can read too.
- open_questions: up to 3 things the clinician should still ask or examine."""


class CaseUpdate(BaseModel):
    field: CaseField
    value: str = Field(description="What the person said, briefly.")


class IntakeTurn(BaseModel):
    updates: list[CaseUpdate]
    complete: bool
    acknowledgement: str = Field(
        default="", description="One short, specific reaction to their answer; may be empty."
    )
    next_question: str = Field(description="The one question to ask next; empty if complete.")
    reason: str = Field(default="", description="Why this question matters, in plain words.")
    quick_replies: list[str] = Field(
        description="Up to 4 short suggested answers, or up to 8 options if multi_select."
    )
    multi_select: bool = Field(
        default=False, description="True if the person may pick several of the options."
    )


class Handoff(BaseModel):
    impression: str = Field(description="2 or 3 sentences for the clinician.")
    urgency_score: int = Field(description="1 (emergency, now) to 5 (self-care likely enough).")
    score_reason: str = Field(description="One plain sentence: why this score.")
    open_questions: list[str] = Field(description="Up to 3 things still to ask or examine.")


class Interviewer(Protocol):
    async def next_turn(
        self, case: Mapping[str, str], transcript: str, briefing: str = ""
    ) -> IntakeTurn: ...

    async def handoff(self, case: Mapping[str, str], briefing: str, assessment: str) -> Handoff: ...


class LlmInterviewer:
    def __init__(self, models: Sequence[BaseChatModel]) -> None:
        self._models = list(models)

    async def next_turn(
        self, case: Mapping[str, str], transcript: str, briefing: str = ""
    ) -> IntakeTurn:
        known = case_lines(case) or "(nothing yet)"
        asked = [
            line[len("assistant: ") :]
            for line in transcript.splitlines()
            if line.startswith("assistant: ")
        ]
        asked_text = "\n".join(f"- {q}" for q in asked) or "(none)"
        user = (
            f"{briefing or NO_PICTURE}\n\nKnown so far:\n{known}\n\n"
            f"Questions already asked:\n{asked_text}\n\nInterview so far:\n{transcript}"
        )
        return await ask_structured(self._models, IntakeTurn, INTERVIEW, user)

    async def handoff(self, case: Mapping[str, str], briefing: str, assessment: str) -> Handoff:
        user = (
            f"Intake:\n{case_lines(case)}\n\n{briefing or NO_PICTURE}\n\n"
            f"Diagnosis agent's assessment:\n{assessment}"
        )
        return await ask_structured(self._models, Handoff, HANDOFF, user)


NO_PICTURE = "Working picture: none yet (the knowledge base has not been searched)."


def briefing(consult: Mapping[str, Any] | None) -> str:
    """The Diagnosis agent's working picture, as the coordinator reads it."""
    if not consult:
        return NO_PICTURE
    leads = consult.get("leads") or []
    lines = ["Working picture from the Diagnosis agent (knowledge base; not a diagnosis):"]
    lines.append(
        "- Considering: "
        + ("; ".join(f"{lead['condition']} ({lead.get('fits', '')})" for lead in leads)
           if leads else "nothing specific yet")
    )  # fmt: skip
    if consult.get("ask_next"):
        lines.append("- Would help to know: " + "; ".join(consult["ask_next"]))
    if consult.get("warning_signs"):
        lines.append("- Warning signs to check: " + "; ".join(consult["warning_signs"]))
    lines.append(f"- Working urgency: {consult.get('urgency', 'routine')}")
    lines.append(
        f"- More questions would change little: {'yes' if consult.get('enough') else 'no'}"
    )
    return "\n".join(lines)


def lead_names(consult: Mapping[str, Any] | None) -> list[str]:
    return [str(lead["condition"]) for lead in (consult or {}).get("leads", [])]


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


def case_summary(case: Mapping[str, str], considered: Sequence[str] = ()) -> str:
    """The case as text for the Diagnosis agent, with the leads its consults raised."""
    leads = f"Possibilities considered during the intake: {', '.join(considered)}\n\n"
    return (
        "Patient-reported information gathered before a clinic visit (not verified):\n"
        f"{case_lines(case)}\n\n"
        f"{leads if considered else ''}"
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


def quick_replies(replies: Sequence[str], multi_select: bool = False) -> list[str]:
    cleaned = [r.strip() for r in replies if r.strip() and len(r.strip()) <= MAX_REPLY_CHARS]
    unique = list(dict.fromkeys(cleaned))
    return unique[: MAX_CHOICES if multi_select else MAX_REPLIES]


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
