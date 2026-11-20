"""What the Coordinator sends back: readable text plus structured data for the web app."""

from typing import Any, Literal

from pydantic import BaseModel

ReplyKind = Literal["emergency", "answer", "no_answer", "greeting", "booking", "out_of_scope"]

DISCLAIMER = (
    "RAGMed is an educational project, not a medical device. This information is not a "
    "diagnosis and does not replace a clinician. In an emergency, call your local emergency "
    "number."
)
GREETING = (
    "Hi, I'm RAGMed. Describe a symptom or ask a health question, and I will answer from a "
    "curated medical knowledge base (MedQuAD, from NIH sources), with the sources I used."
)
OUT_OF_SCOPE = (
    "I can only help with health questions, such as symptoms, conditions and treatments. "
    "What would you like to know?"
)
BOOKING = (
    "Booking appointments is coming soon. For now, please contact your clinic or family doctor "
    "directly."
)


class Source(BaseModel):
    number: int
    question: str
    answer_id: str


class Reply(BaseModel):
    kind: ReplyKind
    text: str
    urgency: str | None = None
    recommend_visit: bool = False
    specialty: str | None = None
    possible_conditions: list[str] = []
    sources: list[Source] = []
    red_flag: str | None = None


def from_diagnosis(data: dict[str, Any], urgent_guidance: str) -> Reply:
    """Turn the Diagnosis agent's result into the reply the person reads."""
    sources = [
        Source(number=int(c["number"]), question=str(c["question"]), answer_id=str(c["answer_id"]))
        for c in data.get("citations", [])
    ]
    urgency = str(data.get("urgency", "routine"))
    parts: list[str] = []
    if urgency == "emergency":
        parts.append(urgent_guidance)
    parts.append(str(data.get("answer", "")).strip())
    if sources:
        parts.append(
            "**Sources** (MedQuAD):\n" + "\n".join(f"{s.number}. {s.question}" for s in sources)
        )
    specialty = data.get("specialty")
    if data.get("recommend_visit") and urgency != "emergency":
        who = f"a {specialty}" if specialty else "a clinician"
        when = "today" if urgency == "urgent" else "soon"
        parts.append(f"**Suggested next step:** see {who} {when}.")
    parts.append(f"_{DISCLAIMER}_")
    return Reply(
        kind="answer" if data.get("sufficient_evidence") else "no_answer",
        text="\n\n".join(p for p in parts if p),
        urgency=urgency,
        recommend_visit=bool(data.get("recommend_visit")),
        specialty=str(specialty) if specialty else None,
        possible_conditions=[str(c) for c in data.get("possible_conditions", [])],
        sources=sources,
    )
