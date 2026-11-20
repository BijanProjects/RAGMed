import httpx
import pytest

from diagnosis_agent.app import DiagnosisSettings, create_diagnosis_app
from diagnosis_agent.executor import AT_CAPACITY
from diagnosis_agent.testing import PASSAGES, FakeKnowledge, FakeReasoner
from ragmed_a2a.client import Progress, RemoteAgent, Reply
from ragmed_core.settings import Environment
from ragmed_llm import AtCapacityError

pytestmark = pytest.mark.anyio
BASE = "http://diagnosis.test"
TOKEN = "c" * 40


def settings(**overrides: object) -> DiagnosisSettings:
    return DiagnosisSettings(public_url=BASE, **overrides)  # type: ignore[arg-type]


async def ask(
    text: str, reasoner: FakeReasoner, knowledge: FakeKnowledge
) -> list[Progress | Reply]:
    app = create_diagnosis_app(settings(), knowledge=knowledge, reasoner=reasoner)
    agent = RemoteAgent(BASE, transport=httpx.ASGITransport(app=app))
    try:
        return [event async for event in agent.stream(text)]
    finally:
        await agent.close()


async def test_answers_over_a2a_with_progress_and_structured_data() -> None:
    events = await ask(
        "Why do I wheeze?", FakeReasoner(answerable=[True]), FakeKnowledge([PASSAGES])
    )

    progress = [e.text for e in events if isinstance(e, Progress)]
    reply = events[-1]
    assert "Searching the medical knowledge base" in progress
    assert isinstance(reply, Reply)
    assert reply.state == "completed"
    assert reply.text == "Inhalers help [2]."
    (data,) = reply.data
    assert data["urgency"] == "routine"
    assert data["citations"][0]["answer_id"] == "a2"
    assert data["sufficient_evidence"] is True


async def test_an_empty_question_is_rejected() -> None:
    events = await ask("   ", FakeReasoner(answerable=[]), FakeKnowledge([PASSAGES]))

    reply = events[-1]
    assert isinstance(reply, Reply)
    assert reply.state == "rejected"


async def test_a_failure_inside_the_graph_fails_the_task_politely() -> None:
    class Broken(FakeReasoner):
        async def search_query(self, question: str, previous: str | None) -> str:
            raise RuntimeError("model is down")

    events = await ask("Why?", Broken(answerable=[]), FakeKnowledge([PASSAGES]))

    reply = events[-1]
    assert isinstance(reply, Reply)
    assert reply.state == "failed"
    assert "try again" in reply.text
    assert "model is down" not in reply.text  # internal errors are not shown to users


def test_production_requires_service_tokens() -> None:
    with pytest.raises(RuntimeError, match="RAGMED_SERVICE_TOKENS"):
        create_diagnosis_app(
            settings(environment=Environment.PRODUCTION),
            knowledge=FakeKnowledge([PASSAGES]),
            reasoner=FakeReasoner(answerable=[]),
        )


async def test_tokens_protect_the_agent_and_appear_on_its_card() -> None:
    app = create_diagnosis_app(
        settings(service_tokens={"coordinator": TOKEN}),
        knowledge=FakeKnowledge([PASSAGES]),
        reasoner=FakeReasoner(answerable=[]),
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as http:
        card = (await http.get("/.well-known/agent-card.json")).json()
        refused = await http.post("/a2a", json={})

    assert card["skills"][0]["id"] == "medical_triage_answer"
    assert "bearer" in card["securitySchemes"]
    assert refused.status_code == 401


async def test_at_capacity_gets_a_clear_message() -> None:
    class Busy(FakeReasoner):
        async def search_query(self, question: str, previous: str | None) -> str:
            raise AtCapacityError("429")

    events = await ask("Why?", Busy(answerable=[]), FakeKnowledge([PASSAGES]))

    reply = events[-1]
    assert isinstance(reply, Reply)
    assert reply.state == "failed"
    assert reply.text == AT_CAPACITY
