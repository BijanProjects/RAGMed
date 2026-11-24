"""Fakes for tests: router, interviewer, Diagnosis agent and scheduling, with no network."""

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from coordinator_agent.intake import CaseUpdate, FollowUp, Handoff, IntakeTurn, Support
from coordinator_agent.router import Intent, Route
from coordinator_agent.scheduling import SchedulingError
from ragmed_a2a.client import AgentEvent, Draft, Progress, Reply

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

CONSULT: dict[str, Any] = {
    "leads": [
        {
            "condition": "migraine",
            "fits": "one-sided, recurring",
            "sources": [{"number": 1, "answer_id": "m1", "question": "What are migraines?"}],
        }
    ],
    "ask_next": ["whether light or noise makes it worse"],
    "warning_signs": [],
    "urgency": "routine",
    "enough": False,
    "search_query": "one-sided headache",
    "passages_read": 4,
}

WITH_SIGNS: dict[str, Any] = {
    **CONSULT,
    "warning_signs": ["Sudden, worst-ever headache", "Fever with a stiff neck"],
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
    acknowledgement: str = ""
    choices: list[str] | None = None  # if set, every question is multiple choice with these
    score: int = 4
    calls: list[tuple[dict[str, str], str]] = field(default_factory=list)
    briefings: list[str] = field(default_factory=list)
    handoffs: list[str] = field(default_factory=list)
    supported: list[tuple[str, str]] = field(default_factory=list)

    async def next_turn(
        self, case: Mapping[str, str], transcript: str, briefing: str = ""
    ) -> IntakeTurn:
        self.calls.append((dict(case), transcript))
        self.briefings.append(briefing)
        asked = len(self.calls) - 1
        updates = [CaseUpdate(field="other_symptoms", value=f"answer {asked}")] if asked else []
        if asked < len(self.questions):
            return IntakeTurn(
                updates=updates,
                complete=False,
                acknowledgement=self.acknowledgement if asked else "",
                next_question=self.questions[asked],
                reason="it helps tell possibilities apart",
                quick_replies=self.choices or ["A day", "A week", "x" * 60],
                multi_select=self.choices is not None,
            )
        return IntakeTurn(updates=updates, complete=True, next_question="", quick_replies=[])

    async def follow_up(
        self, report: str, transcript: str, write: Callable[[str], None] | None = None
    ) -> FollowUp:
        latest = transcript.splitlines()[-1].removeprefix("patient: ")
        if latest.endswith("?"):
            answer = FollowUp(reply="Your report suggests a visit soon.", adds="")
        else:
            answer = FollowUp(reply="Thanks, I've added that for your clinician.", adds=latest)
        if write:
            write(answer.reply)
        return answer

    async def support(
        self, category: str, transcript: str, write: Callable[[str], None] | None = None
    ) -> Support:
        """Resumes when the patient says it is not happening; otherwise stays with them."""
        self.supported.append((category, transcript))
        latest = transcript.splitlines()[-1].casefold()
        if "not happening" in latest or "passed" in latest:
            return Support(reply="Good, I'm glad it has passed.", resume=True)
        if write:
            for word in ("Emergency departments need no appointment.", " Call now."):
                write(word)
        return Support(reply="Emergency departments need no appointment. Call now.", resume=False)

    async def handoff(self, case: Mapping[str, str], briefing: str, assessment: str) -> Handoff:
        self.handoffs.append(assessment)
        return Handoff(
            impression="One-sided recurring headaches for a month; migraine is the main lead.",
            urgency_score=self.score,
            score_reason="Recurring but stable, with no warning signs.",
            open_questions=["Neurological examination"],
        )


@dataclass
class FakeDiagnosis:
    """Answers questions with `result`, and consults with `consult` (or fails them)."""

    result: dict[str, Any] = field(default_factory=lambda: dict(ANSWER))
    state: str = "completed"
    text: str = ""
    consult: dict[str, Any] | None = field(default_factory=lambda: dict(CONSULT))
    questions: list[str] = field(default_factory=list)
    consults: list[dict[str, Any]] = field(default_factory=list)

    async def stream(
        self, text: str, *, data: dict[str, Any] | None = None
    ) -> AsyncIterator[AgentEvent]:
        if data and "consult" in data:
            self.consults.append(data["consult"])
            yield Progress("Searched for “one-sided headache”: 4 passages", "Knowledge base")
            if self.consult is None:
                yield Reply("failed", "The diagnosis service had a problem.")
            else:
                yield Reply("completed", "Working picture", [self.consult])
            return
        self.questions.append(text)
        yield Progress("Searching the medical knowledge base")
        yield Draft("", reset=True, label="Draft · checking against sources")
        yield Draft("Recurring headaches ", label="Draft · checking against sources")
        found = [self.result] if self.state == "completed" else []
        yield Reply(self.state, self.text or str(self.result.get("answer", "")), found)


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
