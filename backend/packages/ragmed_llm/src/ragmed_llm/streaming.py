"""Stream plain text from a chain of models, so the person sees a reply as it is written.

Groq does not stream replies that must match a JSON schema (they arrive in one piece), so
anything streamed is asked for as plain text. A model that fails before its first token hands
over to the next one, as in `ask_structured`; one that fails part-way raises, because the person
has already seen some of its text.
"""

from collections.abc import AsyncIterator, Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessageChunk, HumanMessage, SystemMessage

from ragmed_llm.structured import AtCapacityError, is_rate_limit


def chunk_text(chunk: BaseMessageChunk) -> str:
    """The visible text of a chunk (Gemini sends a list of parts; reasoning is not text)."""
    content = chunk.content
    if isinstance(content, str):
        return content
    return "".join(
        part if isinstance(part, str) else str(part.get("text", ""))
        for part in content
        if isinstance(part, str) or part.get("type") == "text"
    )


async def stream_text(
    models: Sequence[BaseChatModel], system: str, user: str
) -> AsyncIterator[str]:
    """Pieces of the first model's reply as they arrive."""
    if not models:
        raise ValueError("at least one model is required")
    messages = [SystemMessage(system), HumanMessage(user)]
    failure: Exception | None = None
    for model in models:
        started = False
        try:
            async for chunk in model.astream(messages):
                text = chunk_text(chunk)
                if text:
                    started = True
                    yield text
        except Exception as error:
            if started:
                raise
            failure = error
            continue
        if started:
            return
    if failure is not None and is_rate_limit(failure):
        raise AtCapacityError(str(failure)) from failure
    if failure is not None:
        raise failure
    raise RuntimeError("no model returned any text")


class Headed:
    """Splits a streamed reply whose first line is a header ("RESUME", "ADD: ...") from its body.

    Feed it the pieces; the body goes to `write` as it arrives. A reply with no newline in the
    first `HEADER_LIMIT` characters is treated as having no header.
    """

    HEADER_LIMIT = 200

    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write
        self._start = ""
        self.header: str | None = None
        self.body = ""

    def feed(self, text: str) -> None:
        if self.header is None:
            self._start += text
            if "\n" in self._start:
                self.header, rest = self._start.split("\n", 1)
                self.header = self.header.strip()
                self._emit(rest.lstrip("\n"))
            elif len(self._start) > self.HEADER_LIMIT:
                self.header = ""
                self._emit(self._start)
            return
        self._emit(text)

    def close(self) -> None:
        if self.header is None:  # never saw a newline: a header alone, or a short reply
            self.header, self.body = self._start.strip(), ""

    def _emit(self, text: str) -> None:
        if text:
            self.body += text
            self._write(text)
