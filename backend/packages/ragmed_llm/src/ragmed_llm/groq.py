"""Groq chat models, configured the way RAGMed's agents use them."""

from collections.abc import Sequence

from langchain_groq import ChatGroq

from ragmed_core.settings import ProviderKeys


def groq_models(names: Sequence[str], reasoning_effort: str) -> list[ChatGroq]:
    """One model per name, in fallback order. Needs GROQ_API_KEY.

    `max_retries=0` makes a rate-limited model fail at once so the next one takes over.
    Only the gpt-oss models accept a reasoning effort.
    """
    key = ProviderKeys().groq_api_key
    if key is None:
        raise RuntimeError("GROQ_API_KEY is not set (put it in the repository's .env)")
    return [
        ChatGroq(
            model_name=name,
            groq_api_key=key,
            temperature=0,
            max_retries=0,
            request_timeout=60,
            reasoning_effort=reasoning_effort if "gpt-oss" in name else None,
        )
        for name in names
    ]
