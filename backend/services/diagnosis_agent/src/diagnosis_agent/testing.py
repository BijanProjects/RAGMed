"""Fakes for tests: a knowledge base and a reasoner that need no network or model."""

from dataclasses import dataclass, field

from diagnosis_agent.schema import DiagnosisAnswer, GroundingCheck, Passage

PASSAGES = [
    Passage(answer_id="a1", question="What causes asthma?", qtype="causes",
            text="Asthma is caused by inflamed airways.", score=0.8),
    Passage(answer_id="a2", question="How is asthma treated?", qtype="treatment",
            text="Asthma is treated with inhalers.", score=0.7),
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
            specialty="pulmonology",
        )
    )
    unsupported: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    previous_queries: list[str | None] = field(default_factory=list)

    async def search_query(self, question: str, previous: str | None) -> str:
        self.calls.append("rewrite")
        self.previous_queries.append(previous)
        return f"query {len(self.previous_queries)}"

    async def answer(self, instructions: str) -> DiagnosisAnswer:
        self.calls.append("answer")
        return self.answer_given.model_copy(update={"answerable": self.answerable.pop(0)})

    async def check(self, answer: str, passages: str) -> GroundingCheck:
        self.calls.append("check")
        return GroundingCheck(unsupported=list(self.unsupported))
