"""Talks to the Scheduling MCP server: open slots, and booking a visit with its report."""

from typing import Any, Protocol

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError


class SchedulingError(RuntimeError):
    """The scheduling service refused; the message is safe to show the patient."""


class Scheduling(Protocol):
    async def find_slots(self, clinician_type: str, visit_minutes: int) -> list[dict[str, Any]]: ...

    async def book(
        self,
        *,
        clinician_type: str,
        visit_minutes: int,
        start: str,
        session_id: str,
        urgency: str,
        report: dict[str, Any],
    ) -> dict[str, Any]: ...


class McpScheduling:
    def __init__(self, url: str, token: str | None = None, timeout: float = 30.0) -> None:
        self._url = url
        self._token = token
        self._timeout = timeout

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        transport = StreamableHttpTransport(self._url, auth=self._token)
        try:
            async with Client(transport, timeout=self._timeout) as client:
                result = await client.call_tool(tool, arguments)
        except ToolError as error:
            raise SchedulingError(str(error)) from error
        content = result.structured_content or {}
        return content.get("result", content)

    async def find_slots(self, clinician_type: str, visit_minutes: int) -> list[dict[str, Any]]:
        slots: list[dict[str, Any]] = await self._call(
            "find_slots", {"clinician_type": clinician_type, "visit_minutes": visit_minutes}
        )
        return slots

    async def book(
        self,
        *,
        clinician_type: str,
        visit_minutes: int,
        start: str,
        session_id: str,
        urgency: str,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        booked: dict[str, Any] = await self._call(
            "book_visit",
            {
                "clinician_type": clinician_type,
                "visit_minutes": visit_minutes,
                "start": start,
                "session_id": session_id,
                "urgency": urgency,
                "report": report,
            },
        )
        return booked
