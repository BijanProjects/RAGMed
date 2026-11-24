from typing import Any

import httpx
import pytest
from a2a.helpers import new_data_part, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskState

from ragmed_a2a.client import AgentEvent, Progress, RemoteAgent, Reply
from ragmed_a2a.server import CardSpec, Skill, agent_card, create_app, start_task
from ragmed_core.tracing import TraceContext, trace_scope

pytestmark = pytest.mark.anyio

TOKEN = "s" * 40
BASE = "http://agent.test"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Echo(AgentExecutor):
    """Reports one progress step, then echoes the input as text and data."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = await start_task(context, event_queue)
        text = context.get_user_input()
        if text == "fail":
            await updater.failed(updater.new_agent_message([new_text_part("it broke")]))
            return
        await updater.start_work()
        step = updater.new_agent_message([new_text_part("Thinking")])
        await updater.update_status(TaskState.TASK_STATE_WORKING, step)
        await updater.add_artifact(
            [new_text_part(f"echo: {text}"), new_data_part({"n": len(text)})]
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def build(tokens: dict[str, str] | None = None) -> httpx.ASGITransport:
    spec = CardSpec(
        "Echo", "Echoes text", BASE, [Skill("echo", "Echo", "Echo it", ("test",))],
        requires_auth=bool(tokens),
    )  # fmt: skip
    return httpx.ASGITransport(app=create_app(agent_card(spec), Echo(), tokens=tokens))


async def collect(agent: RemoteAgent, text: str, **kwargs: Any) -> list[AgentEvent]:
    try:
        return [event async for event in agent.stream(text, **kwargs)]
    finally:
        await agent.close()


async def test_card_advertises_the_endpoint_skills_and_auth() -> None:
    async with httpx.AsyncClient(transport=build({"c": TOKEN}), base_url=BASE) as http:
        card = (await http.get("/.well-known/agent-card.json")).json()

    assert card["supportedInterfaces"][0]["url"] == f"{BASE}/a2a"
    assert card["skills"][0]["id"] == "echo"
    assert card["capabilities"]["streaming"] is True
    assert card["securitySchemes"]["bearer"]["httpAuthSecurityScheme"]["scheme"] == "bearer"


async def test_health_check_is_public() -> None:
    async with httpx.AsyncClient(transport=build({"c": TOKEN}), base_url=BASE) as http:
        assert (await http.get("/healthz")).json() == {"status": "ok"}


@pytest.mark.parametrize("header", [None, "Bearer wrong", f"Basic {TOKEN}", "Bearer "])
async def test_rpc_needs_a_valid_bearer_token(header: str | None) -> None:
    headers = {"Authorization": header} if header else {}
    async with httpx.AsyncClient(transport=build({"c": TOKEN}), base_url=BASE) as http:
        response = await http.post("/a2a", json={}, headers=headers)

    assert response.status_code == 401


def test_short_tokens_are_refused() -> None:
    with pytest.raises(ValueError, match="32"):
        build({"c": "short"})


async def test_stream_yields_progress_then_one_reply() -> None:
    events = await collect(RemoteAgent(BASE, TOKEN, transport=build({"c": TOKEN})), "hello")

    assert Progress("Thinking") in events
    reply = events[-1]
    assert isinstance(reply, Reply)
    assert reply.state == "completed"
    assert reply.text == "echo: hello"
    assert reply.data == [{"n": 5.0}]
    assert reply.context_id
    assert reply.task_id


async def test_a_failed_task_reports_its_message() -> None:
    events = await collect(RemoteAgent(BASE, transport=build()), "fail")

    reply = events[-1]
    assert isinstance(reply, Reply)
    assert reply.state == "failed"
    assert reply.text == "it broke"


async def test_context_id_continues_a_conversation() -> None:
    first = (await collect(RemoteAgent(BASE, transport=build()), "one"))[-1]
    assert isinstance(first, Reply)

    second = (
        await collect(RemoteAgent(BASE, transport=build()), "two", context_id=first.context_id)
    )[-1]

    assert isinstance(second, Reply)
    assert second.context_id == first.context_id


async def test_calls_carry_a_child_of_the_current_trace() -> None:
    seen: list[str] = []
    inner = build()

    class Recording(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("traceparent", ""))
            return await inner.handle_async_request(request)

    trace = TraceContext.new()
    with trace_scope(trace):
        await collect(RemoteAgent(BASE, transport=Recording()), "hi")

    assert seen
    assert all(h.startswith(f"00-{trace.trace_id}-") for h in seen)
    assert all(h != trace.to_traceparent() for h in seen)  # a new span, same trace
