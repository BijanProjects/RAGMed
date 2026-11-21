"""Ask a chain of models for a typed answer, moving on when one is rate limited.

Groq's free tier allows 8,000 tokens a minute per model, so each role lists several models,
each with its own budget, followed by Gemini models as a backup with separate limits. A
rate-limited or unparseable reply moves straight to the next model.
If every model is limited but the budget refills within `MAX_RATE_LIMIT_WAIT` seconds, the call
waits once; otherwise it raises `AtCapacityError`, which agents turn into a polite message.
"""

import re
from collections.abc import Sequence

import anyio
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

MAX_RATE_LIMIT_WAIT = 10.0
TOO_MANY_REQUESTS = 429
_TRY_AGAIN = re.compile(r"(?:try again|retry) in (\d+(?:\.\d+)?)s", re.IGNORECASE)


class AtCapacityError(RuntimeError):
    """Every model's free-tier budget is used up for now."""


def is_rate_limit(error: BaseException) -> bool:
    """Groq sets `status_code`; Google sets `code` or says RESOURCE_EXHAUSTED."""
    if TOO_MANY_REQUESTS in (getattr(error, "status_code", None), getattr(error, "code", None)):
        return True
    return "RESOURCE_EXHAUSTED" in str(error)


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


async def ask_structured[T: BaseModel](
    models: Sequence[BaseChatModel], schema: type[T], system: str, user: str
) -> T:
    """The first model's answer as `schema`, falling back down the list on failure."""
    if not models:
        raise ValueError("at least one model is required")
    # json_schema mode: Groq's gpt-oss-20b rejects the default function-calling mode.
    first, *rest = [m.with_structured_output(schema, method="json_schema") for m in models]
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
