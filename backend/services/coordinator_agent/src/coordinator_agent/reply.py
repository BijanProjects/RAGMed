"""What the Coordinator sends back: readable text plus structured data for the web app."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel

from coordinator_agent.intake import LABELS
from ragmed_core.clinic import CLINIC_NAME, DRAFT_NOTICE, GENERAL_PHYSICIAN

ReplyKind = Literal[
    "emergency",
    "question",
    "report",
    "slots",
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
    f"Hi, I'm the intake assistant at {CLINIC_NAME}. Tell me what's bothering you and I'll ask "
    "a few questions, prepare a draft report for your clinician, and help you book a visit. You "
    "can also ask a general health question."
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


class Source(BaseModel):
    number: int
    question: str
    answer_id: str


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
    report: dict[str, Any] | None = None
    offer: Offer | None = None
    slots: list[dict[str, Any]] = []
    appointment: dict[str, Any] | None = None


def _sources(data: dict[str, Any]) -> list[Source]:
    return [
        Source(number=int(c["number"]), question=str(c["question"]), answer_id=str(c["answer_id"]))
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


def build_report(case: dict[str, str], data: dict[str, Any]) -> dict[str, Any]:
    """The draft pre-visit report the physician receives with the booking."""
    sources = _sources(data)
    facts = [f"{LABELS[k]}: {v}" for k, v in case.items() if k in LABELS and v]
    return {
        "title": "Draft pre-visit report",
        "notice": DRAFT_NOTICE,
        "summary": "; ".join(facts),
        "case": {LABELS[k]: v for k, v in case.items() if k in LABELS and v},
        "assessment": str(data.get("answer", "")).strip(),
        "possible_conditions": [str(c) for c in data.get("possible_conditions", [])],
        "urgency": str(data.get("urgency", "routine")),
        "clinician_type": str(data.get("clinician_type") or GENERAL_PHYSICIAN),
        "visit_minutes": int(data.get("visit_minutes") or 30),
        "visit_reason": str(data.get("visit_reason", "")),
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
    lines.append(
        "**What you told us**\n" + "\n".join(f"- {k}: {v}" for k, v in report["case"].items())
    )
    if report["assessment"]:
        lines.append("**What the medical sources suggest**\n" + report["assessment"])
    if report["possible_conditions"]:
        lines.append(
            "**Possibilities to discuss with your clinician:** "
            + ", ".join(report["possible_conditions"])
        )
    lines.append(f"**Urgency:** {URGENCY_TEXT.get(urgency, urgency)}")
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
