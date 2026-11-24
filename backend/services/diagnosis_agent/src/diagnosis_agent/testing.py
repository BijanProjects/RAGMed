"""Fakes for tests: a knowledge base and a reasoner that need no network or model."""

from collections.abc import Callable
from dataclasses import dataclass, field

from diagnosis_agent.schema import (
    ConsultDraft,
    DiagnosisAnswer,
    GroundingCheck,
    Lead,
    Passage,
    SearchQuery,
)

PASSAGES = [
    Passage(answer_id="a1", question="What causes asthma?", qtype="causes",
            text="Asthma is caused by inflamed airways.", score=0.8),
    Passage(answer_id="a2", question="How is asthma treated?", qtype="treatment",
            text="Asthma is treated with inhalers.", score=0.7,
            url="https://medlineplus.gov/asthma.html"),
]  # fmt: skip


@dataclass
class FakeKnowledge:
    results: list[list[Passage]]
    queries: list[str] = field(default_factory=list)

    async def search(self, query: str, top_k: int) -> list[Passage]:
        self.queries.append(query)
        return self.results[min(len(self.queries), len(self.results)) - 1]

    async def answer_instructions(self, question: str, passages: str) -> str:
        return f"INSTRUCTIONS\n{question}\n{passages}"


@dataclass
class FakeReasoner:
    answerable: list[bool]
    answer_given: DiagnosisAnswer = field(
        default_factory=lambda: DiagnosisAnswer(
            answerable=True,
            answer="Inhalers help [2].",
            citations=[2],
            possible_conditions=["asthma"],
            urgency="routine",
            recommend_visit=True,
            clinician_type="Pulmonology",
            visit_minutes=30,
            visit_reason="Breathing problems need a specialist assessment.",
        )
    )
    consult_given: ConsultDraft = field(
        default_factory=lambda: ConsultDraft(
            leads=[
                Lead(condition="asthma", fits="wheeze at night", citations=[1, 2]),
                Lead(condition="a guess", fits="no source", citations=[9]),
            ],
            ask_next=["whether exercise brings it on", " "],
            warning_signs=["lips turning blue"],
            urgency="routine",
            enough=False,
        )
    )
    unsupported: list[str] = field(default_factory=list)
    condition: str = ""
    calls: list[str] = field(default_factory=list)
    previous_queries: list[str | None] = field(default_factory=list)

    async def search_query(self, question: str, previous: str | None) -> SearchQuery:
        self.calls.append("rewrite")
        self.previous_queries.append(previous)
        return SearchQuery(query=f"query {len(self.previous_queries)}", condition=self.condition)

    async def answer(
        self,
        question: str,
        instructions: str,
        write: Callable[[str], None] | None = None,
    ) -> DiagnosisAnswer:
        self.calls.append("answer")
        answerable = self.answerable.pop(0)
        if write and answerable:
            for word in self.answer_given.answer.split(" "):
                write(word + " ")
        return self.answer_given.model_copy(update={"answerable": answerable})

    async def check(self, answer: str, passages: str) -> GroundingCheck:
        self.calls.append("check")
        return GroundingCheck(unsupported=list(self.unsupported))

    async def consult(self, case: str, passages: str) -> ConsultDraft:
        self.calls.append("consult")
        return self.consult_given
