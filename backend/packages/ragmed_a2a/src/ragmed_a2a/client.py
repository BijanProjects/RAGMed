"""Call another agent over A2A and read its reply as simple events."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_data_parts, get_text_parts, new_text_message
from a2a.types import Role, SendMessageRequest, StreamResponse, TaskState

from ragmed_core.tracing import current_trace

_FINAL_STATES = {
    TaskState.TASK_STATE_COMPLETED: "completed",
    TaskState.TASK_STATE_FAILED: "failed",
    TaskState.TASK_STATE_CANCELED: "canceled",
    TaskState.TASK_STATE_REJECTED: "rejected",
    TaskState.TASK_STATE_INPUT_REQUIRED: "input_required",
    TaskState.TASK_STATE_AUTH_REQUIRED: "auth_required",
}


@dataclass(frozen=True, slots=True)
class Progress:
    """A status update while the other agent works, e.g. "Searching the knowledge base"."""

    text: str


@dataclass(frozen=True, slots=True)
class Reply:
    """The other agent's result: its text, any structured data, and how the task ended."""

    state: str
    text: str = ""
    data: list[dict[str, Any]] = field(default_factory=list)
    context_id: str = ""
    task_id: str = ""


AgentEvent = Progress | Reply


async def _add_traceparent(request: httpx.Request) -> None:
    trace = current_trace()
    if trace is not None:
        request.headers["traceparent"] = trace.child().to_traceparent()


class RemoteAgent:
    """Another agent, found through its Agent Card at `url`."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        *,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._url = url
        self._http = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            transport=transport,
            event_hooks={"request": [_add_traceparent]},
        )

    async def stream(
        self, text: str, *, context_id: str | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Send `text`; yield progress updates, then exactly one `Reply`."""
        client = await create_client(self._url, ClientConfig(httpx_client=self._http))
        message = new_text_message(text, context_id=context_id, role=Role.ROLE_USER)
        reply_text: list[str] = []
        reply_data: list[dict[str, Any]] = []
        context, task_id, state = context_id or "", "", "failed"
        async for event in client.send_message(SendMessageRequest(message=message)):
            update = _read(event)
            context = update.get("context_id") or context
            task_id = update.get("task_id") or task_id
            if "progress" in update:
                yield Progress(update["progress"])
            reply_text.extend(update.get("text", []))
            reply_data.extend(update.get("data", []))
            state = update.get("state", state)
        yield Reply(state, "\n".join(reply_text), reply_data, context, task_id)

    async def close(self) -> None:
        await self._http.aclose()


def _read(event: StreamResponse) -> dict[str, Any]:
    """Pull the parts we care about out of one A2A stream event."""
    kind = event.WhichOneof("payload")
    if kind == "task":
        return {"context_id": event.task.context_id, "task_id": event.task.id}
    if kind == "artifact_update":
        parts = event.artifact_update.artifact.parts
        return {"text": get_text_parts(parts), "data": get_data_parts(parts)}
    if kind == "status_update":
        status = event.status_update.status
        update: dict[str, Any] = {
            "context_id": event.status_update.context_id,
            "task_id": event.status_update.task_id,
        }
        message_text = " ".join(get_text_parts(status.message.parts)).strip()
        if status.state in _FINAL_STATES:
            update["state"] = _FINAL_STATES[status.state]
            if message_text:
                update["text"] = [message_text]
        elif message_text:
            update["progress"] = message_text
        return update
    if kind == "message":
        return {"text": get_text_parts(event.message.parts), "state": "completed"}
    return {}
