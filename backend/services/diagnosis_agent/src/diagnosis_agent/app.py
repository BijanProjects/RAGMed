"""Build the Diagnosis agent from settings and serve it over A2A."""

import dataclasses

from pydantic import Field, SecretStr
from pydantic_settings import SettingsConfigDict
from starlette.types import ASGIApp

from diagnosis_agent.executor import DiagnosisExecutor
from diagnosis_agent.graph import build_graph
from diagnosis_agent.knowledge import Knowledge, McpKnowledge
from diagnosis_agent.reasoner import LlmReasoner, Reasoner
from ragmed_a2a.server import CardSpec, Skill, agent_card, create_app
from ragmed_core.settings import CoreSettings, Environment


class DiagnosisSettings(CoreSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGMED_", env_file=("../.env", ".env"), extra="ignore", frozen=True
    )

    service_name: str = "diagnosis-agent"
    host: str = "127.0.0.1"
    port: int = 8001
    public_url: str = "http://127.0.0.1:8001"
    # Groq free tier: 8,000 tokens a minute per model, so each role falls back to a model with
    # its own budget. Reasoning effort: "medium" made the small model's quick steps take ~19 s
    # instead of ~0.7 s, while the answer needs "medium" to read the question carefully.
    large_models: tuple[str, ...] = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")
    small_models: tuple[str, ...] = ("openai/gpt-oss-20b", "qwen/qwen3.8-27b")
    small_reasoning: str = "low"
    large_reasoning: str = "medium"
    mcp_url: str = "http://127.0.0.1:9001/mcp"
    mcp_token: SecretStr | None = None
    # Callers allowed to use this agent: JSON object of name to token.
    service_tokens: dict[str, SecretStr] = Field(default_factory=dict)


CARD = CardSpec(
    name="RAGMed Diagnosis Agent",
    description=(
        "Answers a medical question from a curated knowledge base (MedQuAD), with citations, "
        "possible conditions, an urgency level and whether to see a clinician. Not a diagnosis."
    ),
    url="",
    skills=[
        Skill(
            id="medical_triage_answer",
            name="Medical triage answer",
            description="A cited answer plus structured triage (urgency, recommend_visit, "
            "clinician type and visit length) as a JSON data part.",
            tags=["medical", "rag", "triage"],
            examples=["What are the symptoms of asthma?", "I have had a rash and fever"],
        )
    ],
)


def _groq_reasoner(settings: DiagnosisSettings) -> Reasoner:  # pragma: no cover - needs a key
    from ragmed_llm.groq import groq_models  # noqa: PLC0415

    return LlmReasoner(
        small=groq_models(settings.small_models, settings.small_reasoning),
        large=groq_models(settings.large_models, settings.large_reasoning),
    )


def create_diagnosis_app(
    settings: DiagnosisSettings,
    *,
    knowledge: Knowledge | None = None,
    reasoner: Reasoner | None = None,
) -> ASGIApp:
    tokens = {name: token.get_secret_value() for name, token in settings.service_tokens.items()}
    if not tokens and settings.environment is Environment.PRODUCTION:
        raise RuntimeError("RAGMED_SERVICE_TOKENS must be set in production")
    mcp_token = settings.mcp_token.get_secret_value() if settings.mcp_token else None
    graph = build_graph(
        knowledge or McpKnowledge(settings.mcp_url, mcp_token),
        reasoner or _groq_reasoner(settings),
    )
    spec = dataclasses.replace(CARD, url=settings.public_url, requires_auth=bool(tokens))
    return create_app(agent_card(spec), DiagnosisExecutor(graph), tokens=tokens or None)
