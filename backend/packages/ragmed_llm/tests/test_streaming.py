from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel, GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk

from ragmed_llm import AtCapacityError, Headed, stream_text
from ragmed_llm.streaming import chunk_text

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class RateLimitedError(Exception):
    status_code = 429


class Failing(GenericFakeChatModel):
    """Raises `error` after streaming `before` (possibly nothing)."""

    error: Any = None
    before: str = ""

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        from langchain_core.outputs import ChatGenerationChunk  # noqa: PLC0415

        if self.before:
            yield ChatGenerationChunk(message=AIMessageChunk(content=self.before))
        raise self.error


def fake(text: str) -> BaseChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


async def collect(*models: BaseChatModel) -> list[str]:
    return [piece async for piece in stream_text(list(models), "system", "user")]


async def test_text_arrives_in_pieces() -> None:
    pieces = await collect(fake("Call your local emergency number now."))

    assert len(pieces) > 1
    assert "".join(pieces) == "Call your local emergency number now."


async def test_a_model_that_fails_before_any_text_hands_over() -> None:
    busy = Failing(messages=iter([]), error=RateLimitedError("429"))

    assert "".join(await collect(busy, fake("From the backup."))) == "From the backup."


async def test_every_model_busy_is_at_capacity() -> None:
    busy = Failing(messages=iter([]), error=RateLimitedError("429"))

    with pytest.raises(AtCapacityError):
        await collect(busy)


async def test_a_failure_part_way_is_raised_not_hidden() -> None:
    broken = Failing(messages=iter([]), error=RuntimeError("cut"), before="Half a ")

    with pytest.raises(RuntimeError, match="cut"):
        await collect(broken, fake("never used"))


def test_gemini_parts_are_joined_and_reasoning_is_not_text() -> None:
    chunk = AIMessageChunk(content=[{"type": "text", "text": "Hi "}, {"type": "thinking"}, "there"])

    assert chunk_text(chunk) == "Hi there"


def test_a_header_line_is_split_from_the_streamed_body() -> None:
    seen: list[str] = []
    headed = Headed(seen.append)
    for piece in ["RES", "UME\nGood, I'm ", "glad it passed."]:
        headed.feed(piece)
    headed.close()

    assert headed.header == "RESUME"
    assert headed.body == "Good, I'm glad it passed."
    assert "".join(seen) == headed.body


def test_a_reply_without_a_header_is_all_body() -> None:
    seen: list[str] = []
    headed = Headed(seen.append)
    headed.feed("x" * 250)
    headed.close()

    assert headed.header == ""
    assert headed.body == "x" * 250
