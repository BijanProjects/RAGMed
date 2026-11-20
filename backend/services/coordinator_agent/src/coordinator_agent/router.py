"""Decide what the person wants, and restate a follow-up as a self-contained question."""

from collections.abc import Sequence
from typing import Literal, Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from ragmed_llm import ask_structured

Intent = Literal["symptoms", "medical_question", "book_appointment", "greeting", "out_of_scope"]

ROUTE = """You route messages for RAGMed Clinic's intake assistant. Decide what the person wants:

- symptoms: they describe a health problem, symptom or concern of their own or of someone they
  care for (for example "I've had a cough for a week", "my son has a rash"), or need a
  prescription renewal or a note.
- medical_question: a general health question, not about their own situation (for example
  "What causes asthma?").
- book_appointment: they want to book a visit but have not said what for.
- greeting: hello, thanks, or asking what you can do.
- out_of_scope: anything else (weather, coding, general knowledge).

For medical_question, also write `question`: their question rewritten to stand on its own using
the conversation so far. Otherwise leave `question` empty."""


class Route(BaseModel):
    intent: Intent
    question: str = Field(description="Self-contained question; empty unless medical_question.")


class Router(Protocol):
    async def route(self, message: str, history: str) -> Route: ...


class LlmRouter:
    def __init__(self, models: Sequence[BaseChatModel]) -> None:
        self._models = list(models)

    async def route(self, message: str, history: str) -> Route:
        user = f"Conversation so far:\n{history or '(none)'}\n\nNew message:\n{message}"
        return await ask_structured(self._models, Route, ROUTE, user)
