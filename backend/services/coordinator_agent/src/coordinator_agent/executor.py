"""Serve the conversation over A2A: progress as status updates, the reply as text plus data."""

from typing import Any, cast

import structlog
from a2a.helpers import new_data_part, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskState
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ragmed_a2a.server import start_task
from ragmed_llm import AtCapacityError

log = structlog.get_logger()

MAX_MESSAGE_CHARS = 2000
AT_CAPACITY = (
    "RAGMed is at capacity right now: it runs on free model quotas that refill every minute. "
    "Please try again in a minute. If this is an emergency, call your local emergency number."
)
FAILED = "Something went wrong on our side. Please try again."
ROUTING = "Reading your message"


class CoordinatorExecutor(AgentExecutor):
    def __init__(self, graph: CompiledStateGraph[Any]) -> None:
        self._graph = graph

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = await start_task(context, event_queue)
        message = context.get_user_input().strip()
        if not message or len(message) > MAX_MESSAGE_CHARS:
            reason = f"Please send a message of 1 to {MAX_MESSAGE_CHARS:,} characters."
            await updater.reject(updater.new_agent_message([new_text_part(reason)]))
            return

        await updater.start_work(updater.new_agent_message([new_text_part(ROUTING)]))
        # One saved conversation per A2A context, so follow-up questions have their history.
        config: RunnableConfig = {"configurable": {"thread_id": str(context.context_id)}}
        reply: dict[str, Any] | None = None
        try:
            async for event in self._graph.astream(
                {"message": message}, config, stream_mode=["custom", "updates"]
            ):
                # With several stream modes, each event is a (mode, payload) pair.
                mode, data = cast(tuple[str, dict[str, Any]], event)
                if mode == "custom" and "progress" in data:
                    step = updater.new_agent_message([new_text_part(data["progress"])])
                    await updater.update_status(TaskState.TASK_STATE_WORKING, step)
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
