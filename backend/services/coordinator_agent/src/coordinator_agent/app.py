"""Build the Care Coordinator from settings and serve it over A2A."""

import dataclasses
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field, SecretStr
from pydantic_settings import SettingsConfigDict
from starlette.types import ASGIApp

from coordinator_agent.executor import CoordinatorExecutor
from coordinator_agent.graph import Diagnosis, build_graph
from coordinator_agent.intake import Interviewer, LlmInterviewer
from coordinator_agent.limits import Limits
from coordinator_agent.mailer import ConsoleMailer, Mailer, SmtpMailer
from coordinator_agent.router import LlmRouter, Router
from coordinator_agent.scheduling import McpScheduling, Scheduling
from ragmed_a2a.client import RemoteAgent
from ragmed_a2a.server import CardSpec, Skill, agent_card, create_app
from ragmed_core.ratelimit import RateLimiter
from ragmed_core.settings import CoreSettings, Environment


class CoordinatorSettings(CoreSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGMED_", env_file=("../.env", ".env"), extra="ignore", frozen=True
    )

    service_name: str = "coordinator-agent"
    host: str = "127.0.0.1"
    port: int = 8000
    public_url: str = "http://127.0.0.1:8000"
    diagnosis_url: str = "http://127.0.0.1:8001"
    diagnosis_token: SecretStr | None = None
    scheduling_url: str = "http://127.0.0.1:9002/mcp"
    scheduling_token: SecretStr | None = None
    small_models: tuple[str, ...] = ("openai/gpt-oss-20b", "qwen/qwen3.8-27b")
    small_reasoning: str = "low"
    # Used only when every Groq model is busy; needs GOOGLE_API_KEY (see ragmed_llm.gemini).
    backup_models: tuple[str, ...] = ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite")
    # Shared rate-limit counters; without it, counters live in this process's memory.
    valkey_url: str | None = None
    # Saved conversations; without it, they live in memory and are lost on restart.
    database_url: SecretStr | None = None
    # Sends booking verification codes, e.g. Gmail: smtp.gmail.com, 587, the address and an
    # app password. Without a password, local runs log the code instead of emailing it.
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: SecretStr | None = None
    mail_from: str | None = None  # defaults to smtp_username
    # Callers allowed to use this agent (the web app): JSON object of name to token.
    service_tokens: dict[str, SecretStr] = Field(default_factory=dict)


CARD = CardSpec(
    name="RAGMed Care Coordinator",
    description=(
        "Talks with the patient: screens every message for emergencies, answers health "
        "questions with cited sources through the Diagnosis agent, and suggests next steps. "
        "Educational support, not a diagnosis."
    ),
    url="",
    skills=[
        Skill(
            id="triage_conversation",
            name="Triage conversation",
            description="Chat about symptoms and health questions; replies are text plus a JSON "
            "data part with kind, urgency, sources and the suggested next step.",
            tags=["medical", "triage", "chat"],
            examples=["I have had a headache for three days", "What causes asthma?"],
        )
    ],
)


def _models(settings: CoordinatorSettings) -> list[BaseChatModel]:  # pragma: no cover - needs a key
    """The Groq models in fallback order, then the Gemini backup."""
    from ragmed_llm.gemini import gemini_models  # noqa: PLC0415
    from ragmed_llm.groq import groq_models  # noqa: PLC0415

    return [
        *groq_models(settings.small_models, settings.small_reasoning),
        *gemini_models(settings.backup_models),
    ]


def _llm_router(settings: CoordinatorSettings) -> Router:  # pragma: no cover - needs a key
    return LlmRouter(_models(settings))


def _llm_interviewer(settings: CoordinatorSettings) -> Interviewer:  # pragma: no cover
    return LlmInterviewer(_models(settings))


def make_mailer(settings: CoordinatorSettings) -> Mailer:
    if settings.smtp_username and settings.smtp_password:
        return SmtpMailer(
            host=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password.get_secret_value(),
            sender=settings.mail_from or settings.smtp_username,
        )
    if settings.environment is Environment.PRODUCTION:
        raise RuntimeError(
            "RAGMED_SMTP_USERNAME and RAGMED_SMTP_PASSWORD must be set in production"
        )
    return ConsoleMailer()


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value else None


def create_coordinator_app(
    settings: CoordinatorSettings,
    *,
    router: Router | None = None,
    interviewer: Interviewer | None = None,
    diagnosis: Diagnosis | None = None,
    scheduling: Scheduling | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    limiter: RateLimiter | None = None,
    limits: Limits | None = None,
    mailer: Mailer | None = None,
) -> ASGIApp:
    tokens = {name: token.get_secret_value() for name, token in settings.service_tokens.items()}
    if not tokens and settings.environment is Environment.PRODUCTION:
        raise RuntimeError("RAGMED_SERVICE_TOKENS must be set in production")
    graph = build_graph(
        router or _llm_router(settings),
        interviewer or _llm_interviewer(settings),
        diagnosis or RemoteAgent(settings.diagnosis_url, _secret(settings.diagnosis_token)),
        scheduling or McpScheduling(settings.scheduling_url, _secret(settings.scheduling_token)),
        checkpointer=checkpointer or InMemorySaver(),
        limiter=limiter,
        limits=limits,
        mailer=mailer or make_mailer(settings),
    )
    spec = dataclasses.replace(CARD, url=settings.public_url, requires_auth=bool(tokens))
    executor = CoordinatorExecutor(graph, limiter=limiter, limits=limits)
    return create_app(agent_card(spec), executor, tokens=tokens or None)
