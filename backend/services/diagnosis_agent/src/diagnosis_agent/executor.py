"""Serve the graphs over A2A: progress as status updates, the answer as text plus data.

Two skills share the endpoint. A plain text message gets a full cited answer (graph.py). A
message with the data part {"consult": {...}} gets a quick working picture of an intake that is
still under way (consult.py).
"""

from collections.abc import Callable
from typing import Any

import structlog
from a2a.helpers import get_data_parts, new_data_part, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ValidationError

from diagnosis_agent.consult import ConsultRequest
from ragmed_a2a.server import report_progress, start_task
from ragmed_core.clinic import DIAGNOSIS_AGENT, KNOWLEDGE_BASE
from ragmed_llm import AtCapacityError

log = structlog.get_logger()

# What the person sees while each step of a full answer runs, and who is acting.
PROGRESS = {
    "rewrite": ("Understanding the question", DIAGNOSIS_AGENT),
    "retrieve": ("Searching the medical knowledge base", KNOWLEDGE_BASE),
    "generate": ("Reading the sources and writing the answer", DIAGNOSIS_AGENT),
    "check": ("Checking the answer against its sources", DIAGNOSIS_AGENT),
}
MAX_QUESTION_CHARS = 4000
AT_CAPACITY = (
    "RAGMed is at capacity right now: it runs on free model quotas that refill every minute. "
    "Please try again in a minute."
)
FAILED = "The diagnosis service had a problem. Please try again."

Step = tuple[str, str] | None  # (text, agent), or None to stay quiet


def answer_step(node: str, _values: dict[str, Any]) -> Step:
    return PROGRESS.get(node)


def consult_step(node: str, values: dict[str, Any]) -> Step:
    """Consult progress says what was found, so the care team's work is visible."""
    if node == "search":
        found = len(values.get("passages", []))
        return f"Searched for “{values.get('query', '')}”: {found} passages", KNOWLEDGE_BASE
    if node == "weigh" and "result" in values:
        leads = [lead.condition for lead in values["result"].leads]
        text = (
            "Weighed the evidence: considering " + ", ".join(leads)
            if leads
            else "No close match in the knowledge base yet"
        )
        return text, DIAGNOSIS_AGENT
    return None


def consult_request(context: RequestContext) -> ConsultRequest | None:
    if context.message is None:
        return None
    for data in get_data_parts(context.message.parts):
        if isinstance(data, dict) and isinstance(data.get("consult"), dict):
            return ConsultRequest.model_validate(data["consult"])
    return None


class DiagnosisExecutor(AgentExecutor):
    def __init__(
        self, graph: CompiledStateGraph[Any], consult: CompiledStateGraph[Any] | None = None
    ) -> None:
        self._graph = graph
        self._consult = consult

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = await start_task(context, event_queue)
        try:
            request = consult_request(context)
        except ValidationError:
            await _reject(updater, "Send a consult with a case and a complaint.")
            return
        if request is not None and self._consult is not None:
            await updater.start_work()
            result = await _run(updater, self._consult, {"request": request}, consult_step)
            if result is not None:
                await _finish(updater, "Working picture of the intake so far", result, "consult")
            return

        question = context.get_user_input().strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            await _reject(updater, "Send a question of 1 to 4,000 characters.")
            return
        await updater.start_work()
        result = await _run(updater, self._graph, {"question": question}, answer_step)
        if result is not None:
            await _finish(updater, str(getattr(result, "answer", "")), result, "diagnosis")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("diagnosis tasks run to completion")


async def _run(
    updater: TaskUpdater,
    graph: CompiledStateGraph[Any],
    state: dict[str, Any],
    step: Callable[[str, dict[str, Any]], Step],
) -> BaseModel | None:
    """Run a graph, reporting each step; on failure, fail the task and return None."""
    result: BaseModel | None = None
    try:
        async for update in graph.astream(state, stream_mode="updates"):
            for node, values in update.items():
                said = step(node, values or {})
                if said:
                    await report_progress(updater, *said)
                if values and "result" in values:
                    result = values["result"]
    except AtCapacityError:
        log.warning("diagnosis at capacity")
        await updater.failed(updater.new_agent_message([new_text_part(AT_CAPACITY)]))
        return None
    except Exception:
        log.exception("diagnosis failed")
        await updater.failed(updater.new_agent_message([new_text_part(FAILED)]))
        return None
    if result is None:  # pragma: no cover - the graphs always end with a result
        await updater.failed(updater.new_agent_message([new_text_part("No answer produced")]))
    return result


async def _finish(updater: TaskUpdater, text: str, result: BaseModel, name: str) -> None:
    await updater.add_artifact([new_text_part(text), new_data_part(result.model_dump())], name=name)
    await updater.complete()


async def _reject(updater: TaskUpdater, reason: str) -> None:
    await updater.reject(updater.new_agent_message([new_text_part(reason)]))
