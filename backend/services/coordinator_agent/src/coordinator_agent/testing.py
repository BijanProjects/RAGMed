"""Fakes for tests: a router and a Diagnosis agent that need no network or model."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from coordinator_agent.router import Intent, Route
from ragmed_a2a.client import AgentEvent, Progress, Reply

ANSWER: dict[str, Any] = {
    "answer": "Asthma is treated with inhalers [1].",
    "citations": [{"number": 1.0, "answer_id": "a1", "question": "How is asthma treated?"}],
    "possible_conditions": ["asthma"],
    "urgency": "routine",
    "recommend_visit": True,
    "specialty": "pulmonologist",
    "sufficient_evidence": True,
    "search_query": "asthma treatment",
}


@dataclass
class FakeRouter:
    intent: Intent = "medical_question"
    rewrite: str = "How is asthma treated?"
    seen: list[tuple[str, str]] = field(default_factory=list)
    error: Exception | None = None

    async def route(self, message: str, history: str) -> Route:
        self.seen.append((message, history))
        if self.error:
            raise self.error
        question = self.rewrite if self.intent == "medical_question" else ""
        return Route(intent=self.intent, question=question)


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
