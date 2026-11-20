"""Fakes for tests: router, interviewer, Diagnosis agent and scheduling, with no network."""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from coordinator_agent.intake import CaseUpdate, IntakeTurn
from coordinator_agent.router import Intent, Route
from coordinator_agent.scheduling import SchedulingError
from ragmed_a2a.client import AgentEvent, Progress, Reply

ANSWER: dict[str, Any] = {
    "answer": "Recurring one-sided headaches can be migraines [1].",
    "citations": [{"number": 1.0, "answer_id": "m1", "question": "What are migraines?"}],
    "possible_conditions": ["migraine"],
    "urgency": "routine",
    "recommend_visit": True,
    "clinician_type": "Neurology",
    "visit_minutes": 30.0,
    "visit_reason": "Recurring headaches need a full neurological history.",
    "sufficient_evidence": True,
    "search_query": "recurring one-sided headache",
}

SLOTS: list[dict[str, Any]] = [
    {
        "start": "2026-11-23T09:00:00-03:30",
        "end": "2026-11-23T09:30:00-03:30",
        "physician": "Dr. Fred Nurk",
        "clinician_type": "Neurology",
        "visit_minutes": 30,
    },
    {
        "start": "2026-11-23T10:00:00-03:30",
        "end": "2026-11-23T10:30:00-03:30",
        "physician": "Dr. Joe Public",
        "clinician_type": "Neurology",
        "visit_minutes": 30,
    },
]


@dataclass
class FakeRouter:
    intent: Intent = "symptoms"
    rewrite: str = "What causes migraines?"
    seen: list[tuple[str, str]] = field(default_factory=list)
    error: Exception | None = None

    async def route(self, message: str, history: str) -> Route:
        self.seen.append((message, history))
        if self.error:
            raise self.error
        question = self.rewrite if self.intent == "medical_question" else ""
        return Route(intent=self.intent, question=question)


@dataclass
class FakeInterviewer:
    """Asks `questions` in order, recording one fact per answer, then says it is complete."""

    questions: list[str] = field(
        default_factory=lambda: ["How long have you had it?", "How bad is it, from 0 to 10?"]
    )
    calls: list[tuple[dict[str, str], str]] = field(default_factory=list)

    async def next_turn(self, case: Mapping[str, str], transcript: str) -> IntakeTurn:
        self.calls.append((dict(case), transcript))
        asked = len(self.calls) - 1
        updates = [CaseUpdate(field="other_symptoms", value=f"answer {asked}")] if asked else []
        if asked < len(self.questions):
            return IntakeTurn(
                updates=updates,
                complete=False,
                next_question=self.questions[asked],
                quick_replies=["A day", "A week", "x" * 60],
            )
        return IntakeTurn(updates=updates, complete=True, next_question="", quick_replies=[])


@dataclass
class FakeDiagnosis:
    result: dict[str, Any] = field(default_factory=lambda: dict(ANSWER))
    state: str = "completed"
    text: str = ""
    questions: list[str] = field(default_factory=list)

    async def stream(self, text: str) -> AsyncIterator[AgentEvent]:
        self.questions.append(text)
        yield Progress("Searching the medical knowledge base")
        data = [self.result] if self.state == "completed" else []
        yield Reply(self.state, self.text or str(self.result.get("answer", "")), data)


@dataclass
class FakeScheduling:
    slots: list[dict[str, Any]] = field(default_factory=lambda: [dict(s) for s in SLOTS])
    searches: list[tuple[str, int]] = field(default_factory=list)
    bookings: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    async def find_slots(self, clinician_type: str, visit_minutes: int) -> list[dict[str, Any]]:
        self.searches.append((clinician_type, visit_minutes))
        if self.error:
            raise SchedulingError(self.error)
        return [{**s, "clinician_type": clinician_type} for s in self.slots]

    async def book(self, **request: Any) -> dict[str, Any]:
        if self.error:
            raise SchedulingError(self.error)
        self.bookings.append(request)
        return {
            "reference": "RC-TEST42",
            "start": request["start"],
            "minutes": request["visit_minutes"],
            "clinician_type": request["clinician_type"],
            "physician": "Dr. Fred Nurk",
            "patient_name": request["patient_name"],
            "urgency": request["urgency"],
            "status": "booked",
        }
