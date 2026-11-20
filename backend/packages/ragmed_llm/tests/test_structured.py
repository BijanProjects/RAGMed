from typing import Any, cast

import anyio
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from ragmed_llm import AtCapacityError, ask_structured, retry_after

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
