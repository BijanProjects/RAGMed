from types import SimpleNamespace
from typing import Any, cast

import anyio
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, SecretStr

from ragmed_llm import AtCapacityError, ask_structured, is_rate_limit, retry_after
from ragmed_llm.gemini import gemini_models

pytestmark = pytest.mark.anyio


class SearchQuery(BaseModel):
    query: str


async def ask(*models: BaseChatModel) -> str:
    return (await ask_structured(list(models), SearchQuery, "system", "user")).query


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class RateLimitedError(Exception):
    status_code = 429


class FakeModel:
    """Answers structured-output calls from a queue; queued exceptions are raised."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        def respond(_: Any) -> Any:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return RunnableLambda(respond)


def model(*outcomes: Any) -> BaseChatModel:
    return cast(BaseChatModel, FakeModel(list(outcomes)))


def test_retry_after_reads_the_providers_hint() -> None:
    assert retry_after(RateLimitedError("Please try again in 4.71s.")) == pytest.approx(4.71)
    assert retry_after(RateLimitedError("no hint")) is None


async def test_the_next_model_takes_over_when_the_first_is_rate_limited() -> None:
    first = model(RateLimitedError("busy"), RateLimitedError("busy"))
    second = model({"query": "asthma"})
    assert await ask(first, second) == "asthma"


async def test_a_short_wait_is_absorbed(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []

    async def no_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(anyio, "sleep", no_sleep)
    hint = RateLimitedError("try again in 2s")
    only = model(hint, hint, SearchQuery(query="asthma"))

    assert await ask(only) == "asthma"
    assert waits == [2.0]


@pytest.mark.parametrize("hint", ["try again in 45s", "no hint at all"])
async def test_a_long_or_unknown_wait_reports_capacity(hint: str) -> None:
    only = model(RateLimitedError(hint), RateLimitedError(hint))

    with pytest.raises(AtCapacityError):
        await ask(only)


async def test_other_errors_are_not_disguised_as_capacity() -> None:
    only = model(ValueError("bad"), ValueError("bad"))

    with pytest.raises(ValueError, match="bad"):
        await ask(only)


async def test_at_least_one_model_is_required() -> None:
    with pytest.raises(ValueError, match="at least one"):
        await ask()


class GoogleQuotaError(Exception):
    """Shaped like google.genai.errors.ClientError for a used-up free-tier quota."""

    code = 429


def test_googles_rate_limits_are_recognised() -> None:
    assert is_rate_limit(GoogleQuotaError("quota"))
    assert is_rate_limit(Exception("429 RESOURCE_EXHAUSTED. Please retry in 33.4s."))
    assert not is_rate_limit(Exception("400 INVALID_ARGUMENT"))
    assert retry_after(Exception("RESOURCE_EXHAUSTED. Please retry in 33.4s.")) == 33.4


async def test_a_backup_model_answers_when_every_groq_model_is_busy() -> None:
    groq_a = model(RateLimitedError("busy"), RateLimitedError("busy"))
    groq_b = model(RateLimitedError("busy"), RateLimitedError("busy"))
    gemini = model({"query": "migraine"})

    assert await ask(groq_a, groq_b, gemini) == "migraine"


async def test_when_the_backup_is_busy_too_it_reports_capacity() -> None:
    groq = model(RateLimitedError("try again in 50s"), RateLimitedError("try again in 50s"))
    gemini = model(GoogleQuotaError("retry in 40s"), GoogleQuotaError("retry in 40s"))

    with pytest.raises(AtCapacityError):
        await ask(groq, gemini)


def test_without_a_google_key_there_is_no_backup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ragmed_llm.gemini.ProviderKeys", lambda: SimpleNamespace(google_api_key=None)
    )

    assert gemini_models(["gemini-3.5-flash-lite"]) == []


def test_with_a_google_key_each_name_becomes_a_fast_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ragmed_llm.gemini.ProviderKeys",
        lambda: SimpleNamespace(google_api_key=SecretStr("test-only")),
    )

    models = gemini_models(["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"])

    assert [m.model for m in models] == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    assert all(m.max_retries == 0 and m.thinking_level == "minimal" for m in models)
