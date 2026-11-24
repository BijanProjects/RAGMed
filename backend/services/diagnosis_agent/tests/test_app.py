from typing import Any

import httpx
import pytest

from diagnosis_agent.app import DiagnosisSettings, create_diagnosis_app
from diagnosis_agent.executor import AT_CAPACITY
from diagnosis_agent.schema import SearchQuery
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
    assert data["citations"][0]["excerpt"] == "Asthma is treated with inhalers."
    assert data["citations"][0]["url"] == "https://medlineplus.gov/asthma.html"
    assert data["sufficient_evidence"] is True


async def consult(
    request: dict[str, Any], reasoner: FakeReasoner, knowledge: FakeKnowledge
) -> list[Progress | Reply]:
    app = create_diagnosis_app(settings(), knowledge=knowledge, reasoner=reasoner)
    agent = RemoteAgent(BASE, transport=httpx.ASGITransport(app=app))
    try:
        return [e async for e in agent.stream("Intake consult", data={"consult": request})]
    finally:
        await agent.close()


CASE = {"case": "- Main problem: wheezing at night", "complaint": "wheezing at night"}


async def test_a_consult_returns_grounded_leads_and_names_each_agent() -> None:
    reasoner, knowledge = FakeReasoner(answerable=[], condition="asthma"), FakeKnowledge([PASSAGES])

    events = await consult({**CASE, "latest": "wheezing at night"}, reasoner, knowledge)

    reply = events[-1]
    assert isinstance(reply, Reply)
    assert reply.state == "completed"
    (data,) = reply.data
    assert [lead["condition"] for lead in data["leads"]] == ["asthma"]  # the uncited guess is gone
    assert [s["answer_id"] for s in data["leads"][0]["sources"]] == ["a1", "a2"]
    assert data["ask_next"] == ["whether exercise brings it on"]
    assert data["warning_signs"] == ["lips turning blue"]
    assert data["passages_read"] == 2
    assert reasoner.calls == ["rewrite", "consult"]
    assert knowledge.queries == ["asthma", "query 1"]  # the named condition is searched too
    steps = [(e.agent, e.text) for e in events if isinstance(e, Progress)]
    assert ("Knowledge base", "Searched for “query 1”: 2 passages") in steps
    assert ("Diagnosis agent", "Weighed the evidence: considering asthma") in steps


async def test_a_later_consult_searches_by_its_leads_without_rewriting() -> None:
    reasoner, knowledge = FakeReasoner(answerable=[]), FakeKnowledge([PASSAGES])

    await consult(
        {**CASE, "latest": "worse with exercise", "leads": ["asthma"]}, reasoner, knowledge
    )

    assert reasoner.calls == ["consult"]
    assert knowledge.queries == ["asthma", "wheezing at night. worse with exercise"]


async def test_a_consult_with_nothing_found_says_so() -> None:
    reasoner, knowledge = FakeReasoner(answerable=[]), FakeKnowledge([[]])

    events = await consult(CASE, reasoner, knowledge)

    reply = events[-1]
    assert isinstance(reply, Reply)
    (data,) = reply.data
    assert data["leads"] == []
    assert data["enough"] is False
    assert "consult" not in reasoner.calls  # no passages: no model call
    assert any(isinstance(e, Progress) and "No close match" in e.text for e in events)


async def test_a_malformed_consult_is_rejected() -> None:
    events = await consult({"case": "x"}, FakeReasoner(answerable=[]), FakeKnowledge([PASSAGES]))

    reply = events[-1]
    assert isinstance(reply, Reply)
    assert reply.state == "rejected"


async def test_an_empty_question_is_rejected() -> None:
    events = await ask("   ", FakeReasoner(answerable=[]), FakeKnowledge([PASSAGES]))

    reply = events[-1]
    assert isinstance(reply, Reply)
    assert reply.state == "rejected"


async def test_a_failure_inside_the_graph_fails_the_task_politely() -> None:
    class Broken(FakeReasoner):
        async def search_query(self, question: str, previous: str | None) -> SearchQuery:
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

    assert [skill["id"] for skill in card["skills"]] == ["medical_triage_answer", "consult"]
    assert "bearer" in card["securitySchemes"]
    assert refused.status_code == 401


async def test_at_capacity_gets_a_clear_message() -> None:
    class Busy(FakeReasoner):
        async def search_query(self, question: str, previous: str | None) -> SearchQuery:
            raise AtCapacityError("429")

    events = await ask("Why?", Busy(answerable=[]), FakeKnowledge([PASSAGES]))

    reply = events[-1]
    assert isinstance(reply, Reply)
    assert reply.state == "failed"
    assert reply.text == AT_CAPACITY
