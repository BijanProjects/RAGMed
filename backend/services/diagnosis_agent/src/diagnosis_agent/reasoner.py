"""The four language-model steps of the Diagnosis agent, each with a typed result."""

import re
from collections.abc import Sequence
from typing import Protocol, TypeVar

import anyio
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from diagnosis_agent.schema import DiagnosisAnswer, GroundingCheck, SearchQuery

T = TypeVar("T", bound=BaseModel)

# If every model is rate limited but the provider says the budget refills within this many
# seconds, wait once instead of failing.
MAX_RATE_LIMIT_WAIT = 10.0
_TRY_AGAIN = re.compile(r"try again in (\d+(?:\.\d+)?)s", re.IGNORECASE)


class AtCapacityError(RuntimeError):
    """Every model's free-tier budget is used up for now."""


def is_rate_limit(error: BaseException) -> bool:
    return getattr(error, "status_code", None) == 429  # noqa: PLR2004 - HTTP Too Many Requests


def retry_after(error: BaseException) -> float | None:
    """Seconds until the provider's budget refills, if it said."""
    response = getattr(error, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = _TRY_AGAIN.search(str(error))
    return float(match.group(1)) if match else None


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

    @staticmethod
    async def _ask(models: list[BaseChatModel], schema: type[T], system: str, user: str) -> T:
        # json_schema mode: Groq's gpt-oss-20b rejects the default function-calling mode.
        first, *rest = [m.with_structured_output(schema, method="json_schema") for m in models]
        # A rate limit or unparseable output moves straight to the next model (each has its
        # own free-tier token budget); then the whole chain is tried once more.
        chain = first.with_fallbacks(rest).with_retry(stop_after_attempt=2)
        messages = [SystemMessage(system), HumanMessage(user)]
        try:
            result = await chain.ainvoke(messages)
        except Exception as error:
            if not is_rate_limit(error):
                raise
            wait = retry_after(error)
            if wait is None or wait > MAX_RATE_LIMIT_WAIT:
                raise AtCapacityError(str(error)) from error
            await anyio.sleep(wait)
            try:
                result = await chain.ainvoke(messages)
            except Exception as again:
                if is_rate_limit(again):
                    raise AtCapacityError(str(again)) from again
                raise
        return schema.model_validate(result)

    async def search_query(self, question: str, previous: str | None) -> str:
        user = f"Patient's message:\n{question}"
        if previous:
            user += f"\n\nThis query did not find enough: {previous}"
        return (await self._ask(self._small, SearchQuery, REWRITE, user)).query

    async def answer(self, instructions: str) -> DiagnosisAnswer:
        system = "Return the answer in the requested structure."
        return await self._ask(self._large, DiagnosisAnswer, system, instructions)

    async def check(self, answer: str, passages: str) -> GroundingCheck:
        user = f"Passages:\n{passages}\n\nAnswer to check:\n{answer}"
        return await self._ask(self._small, GroundingCheck, CHECK, user)
