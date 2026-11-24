"""What the Coordinator sends back: readable text plus structured data for the web app."""

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel

from coordinator_agent.intake import LABELS, is_straightforward_request
from ragmed_core.clinic import CLINIC_NAME, DRAFT_NOTICE, GENERAL_PHYSICIAN

ReplyKind = Literal[
    "emergency",
    "question",
    "report",
    "slots",
    "verify_email",
    "booked",
    "answer",
    "no_answer",
    "greeting",
    "out_of_scope",
]

DISCLAIMER = (
    "RAGMed is an educational project, not a medical device. This information is not a "
    "diagnosis and does not replace a clinician. In an emergency, call your local emergency "
    "number."
)
GREETING = (
    f"Hi, I'm the Care Coordinator at {CLINIC_NAME}. Tell me what's going on, in your own words. "
    "While we talk, our Diagnosis agent checks the clinic's medical knowledge base, so I only ask "
    "what matters. Then I'll give your clinician a clear summary with how urgent it looks, and "
    "help you book a visit. You can also just ask a general health question."
)
OUT_OF_SCOPE = (
    "I can only help with health concerns and booking a visit at the clinic. What would you "
    "like to be seen for?"
)
BOOKING_START = "Happy to book a visit. First, what would you like to be seen for?"
URGENCY_TEXT = {
    "self_care": "Self-care may be enough, but a visit can confirm it",
    "routine": "Routine: see a clinician in the coming days",
    "urgent": "Urgent: see a clinician today",
    "emergency": "Emergency: call your local emergency number now",
}


# The urgency score (1 most urgent), loosely modelled on five-level emergency triage scales. It
# is not a validated clinical scale.
SCORE_TEXT = {
    1: "Emergency: call your local emergency number now",
    2: "Urgent: see a clinician today",
    3: "Soon: see a clinician within a few days",
    4: "Routine: see a clinician in the next week or two",
    5: "Self-care is likely enough; a visit can confirm it",
}
_LEAST_URGENT_SCORE = {"emergency": 1, "urgent": 2, "routine": 4, "self_care": 5}
_URGENCY_FOR_SCORE = {1: "emergency", 2: "urgent", 3: "routine", 4: "routine", 5: "self_care"}


URGENT_SIGNS = (
    "**Some of what you ticked can be serious.** If it is happening now or getting worse, call "
    "your local emergency number. Otherwise, please be seen today."
)
_ORDER = ["self_care", "routine", "urgent", "emergency"]


def at_least(urgency: str, floor: str) -> str:
    """The more urgent of the two."""
    return max(urgency, floor, key=lambda u: _ORDER.index(u) if u in _ORDER else 1)


def triage(evidence_urgency: str, proposed: int | None) -> tuple[int, str]:
    """The urgency score and the urgency the clinic acts on.

    The Care Coordinator proposes the score; it may raise the urgency the Diagnosis agent's
    evidence supports (a worrying history), but never lower it.
    """
    limit = _LEAST_URGENT_SCORE.get(evidence_urgency, 4)
    score = min(proposed, limit) if proposed in SCORE_TEXT else limit  # ignore a malformed score
    return score, _URGENCY_FOR_SCORE[score]


class Source(BaseModel):
    number: int
    question: str
    answer_id: str
    excerpt: str = ""  # the passage text, shown when the reader opens the source


class Offer(BaseModel):
    """The visit the report recommends, which the patient can book from the chat."""

    clinician_type: str
    visit_minutes: int
    urgency: str
    alternative: str | None = None  # a general physician, when a specialist is recommended


class Reply(BaseModel):
    kind: ReplyKind
    text: str
    urgency: str | None = None
    recommend_visit: bool = False
    possible_conditions: list[str] = []
    sources: list[Source] = []
    red_flag: str | None = None
    quick_replies: list[str] = []
    multi_select: bool = False  # the patient may tick several quick replies
    report: dict[str, Any] | None = None
    offer: Offer | None = None
    slots: list[dict[str, Any]] = []
    appointment: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None  # the booking waiting for its email code


def _sources(data: dict[str, Any]) -> list[Source]:
    return [
        Source(
            number=int(c["number"]),
            question=str(c["question"]),
            answer_id=str(c["answer_id"]),
            excerpt=str(c.get("excerpt", "")),
        )
        for c in data.get("citations", [])
    ]


def from_diagnosis(data: dict[str, Any], urgent_guidance: str) -> Reply:
    """A general health question's answer, as the person reads it."""
    sources = _sources(data)
    urgency = str(data.get("urgency", "routine"))
    parts: list[str] = []
    if urgency == "emergency":
        parts.append(urgent_guidance)
    parts.append(str(data.get("answer", "")).strip())
    if sources:
        parts.append(
            "**Sources** (MedQuAD):\n" + "\n".join(f"{s.number}. {s.question}" for s in sources)
        )
    if data.get("recommend_visit") and urgency != "emergency":
        who = str(data.get("clinician_type") or "a clinician")
        when = "today" if urgency == "urgent" else "soon"
        parts.append(f"**Suggested next step:** see {who} {when}.")
    parts.append(f"_{DISCLAIMER}_")
    return Reply(
        kind="answer" if data.get("sufficient_evidence") else "no_answer",
        text="\n\n".join(p for p in parts if p),
        urgency=urgency,
        recommend_visit=bool(data.get("recommend_visit")),
        possible_conditions=[str(c) for c in data.get("possible_conditions", [])],
        sources=sources,
    )


def _without_disclaimer(text: str) -> str:
    """The report shows its own notice, so drop a disclaimer the model copied into its answer."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [
        s for s in sentences if "educational project" not in s and "local emergency number" not in s
        and "does not replace a clinician" not in s
    ]  # fmt: skip
    return " ".join(kept).strip()


def build_report(
    case: dict[str, str], data: dict[str, Any], handoff: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The draft pre-visit report the physician receives with the booking."""
    sources = _sources(data)
    handoff = handoff or {}
    proposed = handoff.get("urgency_score")
    score, urgency = triage(str(data.get("urgency", "routine")), proposed)
    score_reason = str(handoff.get("score_reason", "")).strip()
    if proposed in SCORE_TEXT and score != proposed:  # the reason was written for another score
        score_reason = (
            "Raised because a warning sign was reported."
            if handoff.get("warning_signs")
            else "Raised to match the urgency the medical sources support."
        )
    clinician = str(data.get("clinician_type") or GENERAL_PHYSICIAN)
    minutes = int(data.get("visit_minutes") or 30)
    reason = str(data.get("visit_reason", ""))
    if is_straightforward_request(case) and urgency in ("self_care", "routine"):
        # The clinic's rule, whatever the model chose: renewals, notes, forms are short GP visits.
        clinician, minutes = GENERAL_PHYSICIAN, 15
        reason = "A straightforward request such as a renewal or a note fits a short visit."
    facts = [f"{LABELS[k]}: {v}" for k, v in case.items() if k in LABELS and v]
    return {
        "title": "Draft pre-visit report",
        "notice": DRAFT_NOTICE,
        "summary": "; ".join(facts),
        "case": {LABELS[k]: v for k, v in case.items() if k in LABELS and v},
        "impression": str(handoff.get("impression", "")).strip(),
        "open_questions": [str(q) for q in handoff.get("open_questions", []) if str(q).strip()][:3],
        "urgency_score": score,
        "score_reason": score_reason,
        "assessment": _without_disclaimer(str(data.get("answer", ""))),
        "possible_conditions": [str(c) for c in data.get("possible_conditions", [])],
        "warning_signs": str(handoff.get("warning_signs", "")),
        "urgency": urgency,
        "clinician_type": clinician,
        "visit_minutes": minutes,
        "visit_reason": reason,
        "sources": [s.model_dump() for s in sources],
        "sufficient_evidence": bool(data.get("sufficient_evidence")),
        "created_at": datetime.now(UTC).isoformat(),
    }


def report_reply(report: dict[str, Any], urgent_guidance: str) -> Reply:
    """The report as the patient sees it, with the visit it recommends."""
    urgency = report["urgency"]
    lines = [f"## {report['title']}", f"_{report['notice']}_"]
    if urgency == "emergency":
        lines.insert(0, urgent_guidance)
    elif report.get("warning_signs"):
        lines.insert(0, URGENT_SIGNS)
    score = report.get("urgency_score")
    if score in SCORE_TEXT:
        lines.append(
            f"**Urgency score: {score} of 5** (1 is most urgent). {SCORE_TEXT[score]}. "
            f"{report.get('score_reason', '')}".strip()
        )
    else:
        lines.append(f"**Urgency:** {URGENCY_TEXT.get(urgency, urgency)}")
    lines.append(
        "**What you told us**\n" + "\n".join(f"- {k}: {v}" for k, v in report["case"].items())
    )
    if report.get("impression"):
        lines.append("**Handoff note for your clinician**\n" + report["impression"])
    if report.get("open_questions"):
        lines.append(
            "**Still to check at your visit**\n"
            + "\n".join(f"- {q}" for q in report["open_questions"])
        )
    if report["assessment"]:
        lines.append("**What the medical sources suggest**\n" + report["assessment"])
    if report["possible_conditions"]:
        lines.append(
            "**Possibilities to discuss with your clinician:** "
            + ", ".join(report["possible_conditions"])
        )
    offer = None
    if urgency != "emergency":
        lines.append(
            f"**Suggested visit:** {report['visit_minutes']} minutes with "
            f"{report['clinician_type']}. {report['visit_reason']}".strip()
        )
        specialist = report["clinician_type"] != GENERAL_PHYSICIAN
        offer = Offer(
            clinician_type=report["clinician_type"],
            visit_minutes=report["visit_minutes"],
            urgency=urgency,
            alternative=GENERAL_PHYSICIAN if specialist else None,
        )
    if report["sources"]:
        lines.append(
            "**Sources** (MedQuAD):\n"
            + "\n".join(f"{s['number']}. {s['question']}" for s in report["sources"])
        )
    return Reply(
        kind="report",
        text="\n\n".join(lines),
        urgency=urgency,
        recommend_visit=True,
        possible_conditions=report["possible_conditions"],
        sources=[Source(**s) for s in report["sources"]],
        report=report,
        offer=offer,
    )
