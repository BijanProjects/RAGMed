from typing import Any

import httpx
import pytest

from coordinator_agent.app import CoordinatorSettings, create_coordinator_app
from coordinator_agent.emergency import CRISIS_GUIDANCE, EMERGENCY_GUIDANCE
from coordinator_agent.executor import AT_CAPACITY
from coordinator_agent.reply import BOOKING, DISCLAIMER, GREETING, OUT_OF_SCOPE
from coordinator_agent.testing import ANSWER, FakeDiagnosis, FakeRouter
from ragmed_a2a.client import Progress, RemoteAgent, Reply
from ragmed_core.settings import Environment
from ragmed_llm import AtCapacityError

pytestmark = pytest.mark.anyio
BASE = "http://coordinator.test"


class Chat:
    """A conversation with the Coordinator over A2A, in memory."""

    def __init__(self, router: FakeRouter, diagnosis: FakeDiagnosis) -> None:
        app = create_coordinator_app(
            CoordinatorSettings(public_url=BASE), router=router, diagnosis=diagnosis
        )
        self.agent = RemoteAgent(BASE, transport=httpx.ASGITransport(app=app))
        self.context_id: str | None = None

    async def send(self, text: str) -> tuple[list[str], Reply]:
        events = [e async for e in self.agent.stream(text, context_id=self.context_id)]
        reply = events[-1]
        assert isinstance(reply, Reply)
        self.context_id = reply.context_id
        return [e.text for e in events if isinstance(e, Progress)], reply


def data(reply: Reply) -> dict[str, Any]:
    (payload,) = reply.data
    return payload


async def test_a_red_flag_is_escalated_before_any_model_is_called() -> None:
    router, diagnosis = FakeRouter(), FakeDiagnosis()

    _, reply = await Chat(router, diagnosis).send("I have crushing chest pain")

    assert router.seen == []
    assert diagnosis.questions == []
    assert reply.text.startswith(EMERGENCY_GUIDANCE)
    assert data(reply)["kind"] == "emergency"
    assert data(reply)["red_flag"] == "chest_pain"


async def test_self_harm_gets_the_crisis_line() -> None:
    _, reply = await Chat(FakeRouter(), FakeDiagnosis()).send("I want to end my life")

    assert reply.text.startswith(CRISIS_GUIDANCE)


async def test_a_medical_question_is_answered_with_sources_and_next_step() -> None:
    diagnosis = FakeDiagnosis()

    progress, reply = await Chat(FakeRouter(), diagnosis).send("how do you treat asthma")

    assert diagnosis.questions == ["How is asthma treated?"]  # the router's rewrite
    assert "Searching the medical knowledge base" in progress
    assert "Asthma is treated with inhalers [1]." in reply.text
    assert "1. How is asthma treated?" in reply.text
    assert "see a pulmonologist soon" in reply.text
    assert DISCLAIMER in reply.text
    payload = data(reply)
    assert payload["kind"] == "answer"
    assert payload["sources"][0]["answer_id"] == "a1"
    assert payload["recommend_visit"] is True


async def test_an_emergency_judged_by_diagnosis_leads_with_guidance() -> None:
    diagnosis = FakeDiagnosis(result={**ANSWER, "urgency": "emergency"})

    _, reply = await Chat(FakeRouter(), diagnosis).send("my chest feels odd")

    assert reply.text.startswith(EMERGENCY_GUIDANCE)
    assert "Suggested next step" not in reply.text


async def test_without_enough_evidence_the_reply_says_so() -> None:
    weak = {**ANSWER, "sufficient_evidence": False, "citations": [], "specialty": None}

    _, reply = await Chat(FakeRouter(), FakeDiagnosis(result=weak)).send("rare thing?")

    assert data(reply)["kind"] == "no_answer"
    assert "see a clinician soon" in reply.text


async def test_a_diagnosis_failure_passes_its_message_on() -> None:
    diagnosis = FakeDiagnosis(state="failed", text="RAGMed is at capacity right now.")

    _, reply = await Chat(FakeRouter(), diagnosis).send("asthma?")

    assert reply.state == "completed"
    assert reply.text.startswith("RAGMed is at capacity right now.")
    assert data(reply)["kind"] == "no_answer"


@pytest.mark.parametrize(
    ("intent", "text", "kind"),
    [
        ("greeting", GREETING, "greeting"),
        ("book_appointment", BOOKING, "booking"),
        ("out_of_scope", OUT_OF_SCOPE, "out_of_scope"),
    ],
)
async def test_other_intents_get_a_direct_reply(intent: Any, text: str, kind: str) -> None:
    diagnosis = FakeDiagnosis()

    _, reply = await Chat(FakeRouter(intent=intent), diagnosis).send("hello there")

    assert diagnosis.questions == []
    assert reply.text == text
    assert data(reply)["kind"] == kind


async def test_follow_ups_see_the_conversation_so_far() -> None:
    router = FakeRouter()
    chat = Chat(router, FakeDiagnosis())

    await chat.send("how do you treat asthma")
    await chat.send("what about children?")

    first_history, second_history = router.seen[0][1], router.seen[1][1]
    assert first_history == ""
    assert "patient: how do you treat asthma" in second_history
    assert "assistant: Asthma is treated with inhalers" in second_history


async def test_separate_conversations_do_not_share_history() -> None:
    router = FakeRouter()
    await Chat(router, FakeDiagnosis()).send("first person")
    await Chat(router, FakeDiagnosis()).send("second person")

    assert router.seen[1][1] == ""


async def test_at_capacity_gets_a_clear_message() -> None:
    router = FakeRouter(error=AtCapacityError("429"))

    _, reply = await Chat(router, FakeDiagnosis()).send("asthma?")

    assert reply.state == "failed"
    assert reply.text == AT_CAPACITY


async def test_an_unexpected_error_fails_politely() -> None:
    router = FakeRouter(error=RuntimeError("secret internals"))

    _, reply = await Chat(router, FakeDiagnosis()).send("asthma?")

    assert reply.state == "failed"
    assert "secret internals" not in reply.text


@pytest.mark.parametrize("text", ["   ", "x" * 2001])
async def test_empty_or_huge_messages_are_rejected(text: str) -> None:
    _, reply = await Chat(FakeRouter(), FakeDiagnosis()).send(text)

    assert reply.state == "rejected"


def test_production_requires_service_tokens() -> None:
    settings = CoordinatorSettings(environment=Environment.PRODUCTION)

    with pytest.raises(RuntimeError, match="RAGMED_SERVICE_TOKENS"):
        create_coordinator_app(settings, router=FakeRouter(), diagnosis=FakeDiagnosis())
