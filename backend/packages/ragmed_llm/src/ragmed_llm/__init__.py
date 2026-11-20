"""Typed language-model calls shared by the agents: fallbacks and rate-limit handling."""

from ragmed_llm.structured import AtCapacityError, ask_structured, is_rate_limit, retry_after

__all__ = ["AtCapacityError", "ask_structured", "is_rate_limit", "retry_after"]
