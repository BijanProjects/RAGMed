"""The four language-model steps of the Diagnosis agent, each with a typed result."""

from collections.abc import Sequence
from typing import Protocol

from langchain_core.language_models import BaseChatModel

from diagnosis_agent.schema import ConsultDraft, DiagnosisAnswer, GroundingCheck, SearchQuery
from ragmed_llm import ask_structured

REWRITE = """Turn the patient's message into one self-contained search query for a medical
knowledge base of consumer health questions and answers (MedQuAD). If the symptoms point to a
common condition, start the query with its name (for example "migraine" for one-sided throbbing
headaches with nausea), then the key symptoms; the knowledge base files answers under condition
names. If a previous query found nothing useful, try different wording."""

CHECK = """You are checking an answer against its numbered sources. List every sentence of the
answer that states a medical fact the passages do not support, copied exactly. Do not list
sentences that are supported, and never list advice to see a clinician or to seek care. If every
fact is supported, return an empty list."""


CONSULT = """You support a clinic's triage coordinator while they talk with a patient. From the
intake so far and the numbered passages from the clinic's medical knowledge base, give your
working picture, using only what the passages say:

- leads: the conditions the passages connect to this case, most likely first, each citing the
  passages that describe it. Prefer common conditions unless the details point elsewhere.
- ask_next: the facts not yet known that would best tell the leads apart or settle the urgency.
  Never list something the intake already answers.
- warning_signs: the signs that would make this urgent and that nobody has asked about yet.
- urgency, from what is known so far, and whether more questions would change anything.

This guides the questions; it is never shown to the patient as a diagnosis."""


class Reasoner(Protocol):
    async def search_query(self, question: str, previous: str | None) -> SearchQuery: ...

    async def answer(self, instructions: str) -> DiagnosisAnswer: ...

    async def check(self, answer: str, passages: str) -> GroundingCheck: ...

    async def consult(self, case: str, passages: str) -> ConsultDraft: ...


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

    async def search_query(self, question: str, previous: str | None) -> SearchQuery:
        user = f"Patient's message:\n{question}"
        if previous:
            user += f"\n\nThis query did not find enough: {previous}"
        return await ask_structured(self._small, SearchQuery, REWRITE, user)

    async def answer(self, instructions: str) -> DiagnosisAnswer:
        system = "Return the answer in the requested structure."
        return await ask_structured(self._large, DiagnosisAnswer, system, instructions)

    async def check(self, answer: str, passages: str) -> GroundingCheck:
        user = f"Passages:\n{passages}\n\nAnswer to check:\n{answer}"
        return await ask_structured(self._small, GroundingCheck, CHECK, user)

    async def consult(self, case: str, passages: str) -> ConsultDraft:
        """Runs on every intake turn, on shortened passages. It uses the large model: weighing a
        differential needs it, and its token budget is separate from the Coordinator's, which
        talks to the patient on the small model at the same time."""
        user = f"Intake so far:\n{case}\n\nPassages:\n{passages}"
        return await ask_structured(self._large, ConsultDraft, CONSULT, user)
