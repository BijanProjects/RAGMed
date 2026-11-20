"""Serve the graph over A2A: progress as status updates, the answer as text plus data."""

from typing import Any

import structlog
from a2a.helpers import new_data_part, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskState
from langgraph.graph.state import CompiledStateGraph

from diagnosis_agent.reasoner import AtCapacityError
from diagnosis_agent.schema import DiagnosisResult
from ragmed_a2a.server import start_task

log = structlog.get_logger()

# What the person sees while each step runs.
PROGRESS = {
    "rewrite": "Understanding the question",
    "retrieve": "Searching the medical knowledge base",
    "generate": "Reading the sources and writing the answer",
    "check": "Checking the answer against its sources",
}
MAX_QUESTION_CHARS = 4000
AT_CAPACITY = (
    "RAGMed is at capacity right now: it runs on free model quotas that refill every minute. "
    "Please try again in a minute."
)


class DiagnosisExecutor(AgentExecutor):
    def __init__(self, graph: CompiledStateGraph[Any]) -> None:
        self._graph = graph

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = await start_task(context, event_queue)
        question = context.get_user_input().strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            reason = "Send a question of 1 to 4,000 characters."
            await updater.reject(updater.new_agent_message([new_text_part(reason)]))
            return

        await updater.start_work()
        result: DiagnosisResult | None = None
        try:
            async for update in self._graph.astream({"question": question}, stream_mode="updates"):
                for node, values in update.items():
                    if node in PROGRESS:
                        message = updater.new_agent_message([new_text_part(PROGRESS[node])])
                        await updater.update_status(TaskState.TASK_STATE_WORKING, message)
                    if values and "result" in values:
                        result = values["result"]
        except AtCapacityError:
            log.warning("diagnosis at capacity")
            await updater.failed(updater.new_agent_message([new_text_part(AT_CAPACITY)]))
            return
        except Exception:
            log.exception("diagnosis failed")
            failure = "The diagnosis service had a problem. Please try again."
            await updater.failed(updater.new_agent_message([new_text_part(failure)]))
            return

        if result is None:  # pragma: no cover - the graph always ends with a result
            await updater.failed(updater.new_agent_message([new_text_part("No answer produced")]))
            return
        await updater.add_artifact(
            [new_text_part(result.answer), new_data_part(result.model_dump())], name="diagnosis"
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("diagnosis tasks run to completion")
