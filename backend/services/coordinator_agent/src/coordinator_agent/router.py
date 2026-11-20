"""Decide what the person wants, and restate a follow-up as a self-contained question."""

from collections.abc import Sequence
from typing import Literal, Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from ragmed_llm import ask_structured

Intent = Literal["medical_question", "book_appointment", "greeting", "out_of_scope"]

ROUTE = """You route messages for a medical information assistant. Decide what the person wants:

- medical_question: symptoms, conditions, treatments, tests, prevention, or any health concern.
- book_appointment: they want to book, change or cancel a visit with a clinician.
- greeting: hello, thanks, or asking what you can do.
- out_of_scope: anything else (weather, coding, general knowledge).

For a medical question, also write `question`: the person's question rewritten to stand on its
own, using the conversation so far (for example "what about children?" after a question about
asthma becomes "How is asthma treated in children?"). Keep their symptoms and details."""


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
