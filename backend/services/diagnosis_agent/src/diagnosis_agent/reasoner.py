"""The four language-model steps of the Diagnosis agent, each with a typed result."""

from collections.abc import Sequence
from typing import Protocol

from langchain_core.language_models import BaseChatModel

from diagnosis_agent.schema import DiagnosisAnswer, GroundingCheck, SearchQuery
from ragmed_llm import ask_structured

REWRITE = """Turn the patient's message into one self-contained search query for a medical
knowledge base of consumer health questions and answers (MedQuAD). Name the condition or
symptoms explicitly. If a previous query found nothing useful, try different wording, for example
the likely condition instead of the symptoms."""

CHECK = """You are checking an answer against its numbered sources. List every sentence of the
answer that states a medical fact the passages do not support, copied exactly. Do not list
sentences that are supported, and never list advice to see a clinician or to seek care. If every
fact is supported, return an empty list."""


class Reasoner(Protocol):
    async def search_query(self, question: str, previous: str | None) -> str: ...

    async def answer(self, instructions: str) -> DiagnosisAnswer: ...

    async def check(self, answer: str, passages: str) -> GroundingCheck: ...


class LlmReasoner:
    """A small fast model rewrites and checks; the large one judges the evidence and answers.

    Judging and answering share one call because Groq's free tier allows 8,000 tokens a minute
    per model, and every call that carries the passages costs about 2,500 of them.
    """

    def __init__(self, small: Sequence[BaseChatModel], large: Sequence[BaseChatModel]) -> None:
        """Each role takes models in order of preference; later ones are fallbacks."""
        if not small or not large:
            raise ValueError("each role needs at least one model")
        self._small = list(small)
        self._large = list(large)

    async def search_query(self, question: str, previous: str | None) -> str:
        user = f"Patient's message:\n{question}"
        if previous:
            user += f"\n\nThis query did not find enough: {previous}"
        return (await ask_structured(self._small, SearchQuery, REWRITE, user)).query

    async def answer(self, instructions: str) -> DiagnosisAnswer:
        system = "Return the answer in the requested structure."
        return await ask_structured(self._large, DiagnosisAnswer, system, instructions)

    async def check(self, answer: str, passages: str) -> GroundingCheck:
        user = f"Passages:\n{passages}\n\nAnswer to check:\n{answer}"
        return await ask_structured(self._small, GroundingCheck, CHECK, user)
