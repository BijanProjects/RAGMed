"""Build the Care Coordinator from settings and serve it over A2A."""

import dataclasses

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field, SecretStr
from pydantic_settings import SettingsConfigDict
from starlette.types import ASGIApp

from coordinator_agent.executor import CoordinatorExecutor
from coordinator_agent.graph import Diagnosis, build_graph
from coordinator_agent.router import LlmRouter, Router
from ragmed_a2a.client import RemoteAgent
from ragmed_a2a.server import CardSpec, Skill, agent_card, create_app
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
    small_models: tuple[str, ...] = ("openai/gpt-oss-20b", "qwen/qwen3.8-27b")
    small_reasoning: str = "low"
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


def _llm_router(settings: CoordinatorSettings) -> Router:  # pragma: no cover - needs a key
    from ragmed_llm.groq import groq_models  # noqa: PLC0415

    return LlmRouter(groq_models(settings.small_models, settings.small_reasoning))


def create_coordinator_app(
    settings: CoordinatorSettings,
    *,
    router: Router | None = None,
    diagnosis: Diagnosis | None = None,
) -> ASGIApp:
    tokens = {name: token.get_secret_value() for name, token in settings.service_tokens.items()}
    if not tokens and settings.environment is Environment.PRODUCTION:
        raise RuntimeError("RAGMED_SERVICE_TOKENS must be set in production")
    diagnosis_token = (
        settings.diagnosis_token.get_secret_value() if settings.diagnosis_token else None
    )
    graph = build_graph(
        router or _llm_router(settings),
        diagnosis or RemoteAgent(settings.diagnosis_url, diagnosis_token),
        # In memory for now: conversations are lost on restart. Postgres comes with deployment.
        checkpointer=InMemorySaver(),
    )
    spec = dataclasses.replace(CARD, url=settings.public_url, requires_auth=bool(tokens))
    return create_app(agent_card(spec), CoordinatorExecutor(graph), tokens=tokens or None)
