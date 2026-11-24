"""Serve the conversation over A2A: progress as status updates, the reply as text plus data."""

from typing import Any, cast

import structlog
from a2a.helpers import get_data_parts, new_data_part, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from coordinator_agent.limits import Limits, client_ip, message_limit
from ragmed_a2a.server import report_draft, report_progress, start_task
from ragmed_core.clinic import COORDINATOR_AGENT
from ragmed_core.ratelimit import MemoryRateLimiter, RateLimiter
from ragmed_llm import AtCapacityError

log = structlog.get_logger()

MAX_MESSAGE_CHARS = 2000
AT_CAPACITY = (
    "RAGMed is at capacity right now: it runs on free model quotas that refill every minute. "
    "Please try again in a minute. If this is an emergency, call your local emergency number."
)
FAILED = "Something went wrong on our side. Please try again."
ROUTING = "Reading your message"


ACTIONS = {"show_slots", "book", "resend", "confirm"}


def read_action(context: RequestContext) -> dict[str, Any] | None:
    """A button press from the web app, sent as a data part: {"action": {"type": ...}}."""
    if context.message is None:
        return None
    for data in get_data_parts(context.message.parts):
        action = data.get("action") if isinstance(data, dict) else None
        if isinstance(action, dict) and action.get("type") in ACTIONS:
            fields = ("type", "start", "clinician_type", "name", "email", "code")
            return {k: v for k, v in action.items() if k in fields}
    return None


class CoordinatorExecutor(AgentExecutor):
    def __init__(
        self,
        graph: CompiledStateGraph[Any],
        limiter: RateLimiter | None = None,
        limits: Limits | None = None,
    ) -> None:
        self._graph = graph
        self._limiter = limiter or MemoryRateLimiter()
        self._limits = limits or Limits()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = await start_task(context, event_queue)
        message = context.get_user_input().strip()
        action = read_action(context)
        if (not message and action is None) or len(message) > MAX_MESSAGE_CHARS:
            reason = f"Please send a message of 1 to {MAX_MESSAGE_CHARS:,} characters."
            await updater.reject(updater.new_agent_message([new_text_part(reason)]))
            return
        ip = client_ip(context)
        refusal = await message_limit(self._limiter, self._limits, ip, str(context.context_id))
        if refusal:
            await updater.reject(updater.new_agent_message([new_text_part(refusal)]))
            return

        await updater.start_work()
        await report_progress(updater, ROUTING, COORDINATOR_AGENT)
        # One saved conversation per A2A context, so follow-up questions have their history.
        config: RunnableConfig = {"configurable": {"thread_id": str(context.context_id)}}
        reply: dict[str, Any] | None = None
        try:
            turn = {
                "message": message,
                "action": action,
                "session": str(context.context_id),
                "client": ip,
            }
            async for event in self._graph.astream(turn, config, stream_mode=["custom", "updates"]):
                # With several stream modes, each event is a (mode, payload) pair.
                mode, data = cast(tuple[str, dict[str, Any]], event)
                if mode == "custom" and "draft" in data:
                    await report_draft(
                        updater,
                        str(data["draft"]),
                        agent=str(data.get("agent") or COORDINATOR_AGENT),
                        label=str(data.get("label") or ""),
                        reset=bool(data.get("reset")),
                    )
                elif mode == "custom" and "progress" in data:
                    agent = str(data.get("agent") or COORDINATOR_AGENT)
                    await report_progress(updater, data["progress"], agent)
                elif mode == "updates":
                    for values in data.values():
                        if values and "reply" in values:
                            reply = values["reply"]
        except AtCapacityError:
            log.warning("coordinator at capacity")
            await updater.failed(updater.new_agent_message([new_text_part(AT_CAPACITY)]))
            return
        except Exception:
            log.exception("coordinator failed")
            await updater.failed(updater.new_agent_message([new_text_part(FAILED)]))
            return

        if reply is None:  # pragma: no cover - every path sets a reply
            await updater.failed(updater.new_agent_message([new_text_part(FAILED)]))
            return
        await updater.add_artifact(
            [new_text_part(reply["text"]), new_data_part(reply)], name="reply"
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("replies run to completion")
