"""The four language-model steps of the Diagnosis agent, each with a typed result."""

import re
from collections.abc import Callable, Sequence
from typing import Protocol

from langchain_core.language_models import BaseChatModel

from diagnosis_agent.schema import (
    ConsultDraft,
    DiagnosisAnswer,
    GroundingCheck,
    SearchQuery,
    Triage,
)
from ragmed_llm import ask_structured, stream_text

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


NOT_ENOUGH = "NOT_ENOUGH"
WRITE_ANSWER = f"""Answer as plain text for the person, following the instructions you are given,
with one change: do not add a disclaimer at the end; the clinic adds its own. Cite every claim
with the number of its passage in square brackets, like [2] or [1, 3], right after the claim:
an answer without citations is thrown away. Be strict about
the evidence: if the passages do not contain enough to answer (passages about a different
condition, or only loosely related, are not enough), reply with exactly {NOT_ENOUGH} and nothing
else."""

TRIAGE = """Read a medical question and the answer written for it from the clinic's sources,
and say what it means for the person: possible conditions, urgency, whether to see a clinician,
who, and for how long. If the answer is NOT_ENOUGH, judge from the question alone."""

# How a model says the passages are not enough: the agreed word, or the source prompt's wording.
REFUSALS = (NOT_ENOUGH, "I do not have enough information")


def might_refuse(text: str) -> bool:
    """True while `text` is, or could still become, a refusal: such a draft is never shown."""
    start = text.lstrip()
    return any(start.startswith(r) or r.startswith(start) for r in REFUSALS)


_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
Write = Callable[[str], None]


def citations_in(answer: str) -> list[int]:
    """The passage numbers an answer cites, as [2] or [1, 3]."""
    numbers = {int(n) for group in _CITATION.findall(answer) for n in group.split(",")}
    return sorted(numbers)


class Reasoner(Protocol):
    async def search_query(self, question: str, previous: str | None) -> SearchQuery: ...

    async def answer(
        self, question: str, instructions: str, write: Write | None = None
    ) -> DiagnosisAnswer: ...

    async def check(self, answer: str, passages: str) -> GroundingCheck: ...

    async def consult(self, case: str, passages: str) -> ConsultDraft: ...


class LlmReasoner:
    """A small fast model rewrites and checks; the large one judges the evidence and answers.

    Judging and answering share one call because Groq's free tier allows 8,000 tokens a minute
    per model, and every call that carries the passages costs about 2,500 of them.
    """

    def __init__(
        self,
        small: Sequence[BaseChatModel],
        large: Sequence[BaseChatModel],
        consult: Sequence[BaseChatModel] = (),
    ) -> None:
        """Each role takes models in order of preference; later ones are fallbacks. Consults
        use the large models unless given their own."""
        if not small or not large:
            raise ValueError("each role needs at least one model")
        self._small = list(small)
        self._large = list(large)
        self._consult = list(consult) or self._large

    async def search_query(self, question: str, previous: str | None) -> SearchQuery:
        user = f"Patient's message:\n{question}"
        if previous:
            user += f"\n\nThis query did not find enough: {previous}"
        return await ask_structured(self._small, SearchQuery, REWRITE, user)

    async def answer(
        self, question: str, instructions: str, write: Write | None = None
    ) -> DiagnosisAnswer:
        """Write the answer as streamed plain text (pieces go to `write` as they arrive), then
        read its triage from the finished text with the small model, without the passages."""
        text, held = "", ""
        async for piece in stream_text(self._large, WRITE_ANSWER, instructions):
            text += piece
            if write is None or might_refuse(text):
                held += piece  # might still become NOT_ENOUGH: show nothing yet
                continue
            write(held + piece)
            held = ""
        answer = text.strip()
        enough = not might_refuse(answer)
        if write and held and enough:
            write(held)
        user = f"Question:\n{question}\n\nAnswer:\n{answer}"
        triage = await ask_structured(self._small, Triage, TRIAGE, user)
        cited = citations_in(answer) if enough else []
        return DiagnosisAnswer(
            answerable=enough and bool(cited),
            answer=answer if enough else "",
            citations=cited,
            **triage.model_dump(),
        )

    async def check(self, answer: str, passages: str) -> GroundingCheck:
        user = f"Passages:\n{passages}\n\nAnswer to check:\n{answer}"
        return await ask_structured(self._small, GroundingCheck, CHECK, user)

    async def consult(self, case: str, passages: str) -> ConsultDraft:
        """Runs on every intake turn, on shortened passages, on models whose token budget is
        not shared with the Coordinator's or the final assessment's."""
        user = f"Intake so far:\n{case}\n\nPassages:\n{passages}"
        return await ask_structured(self._consult, ConsultDraft, CONSULT, user)
