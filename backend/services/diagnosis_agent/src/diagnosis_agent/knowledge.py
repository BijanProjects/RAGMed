"""Talks to the Medical Knowledge MCP server: search, and the versioned answer prompt."""

from typing import Any, Protocol

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from diagnosis_agent.schema import Passage


class Knowledge(Protocol):
    async def search(self, query: str, top_k: int) -> list[Passage]: ...

    async def answer_instructions(self, question: str, passages: str) -> str: ...


class McpKnowledge:
    """Calls the MCP server over HTTP, authenticating with this agent's service token."""

    def __init__(self, url: str, token: str | None = None, timeout: float = 15.0) -> None:
        self._url = url
        self._token = token
        self._timeout = timeout

    def _client(self) -> Client[StreamableHttpTransport]:
        transport = StreamableHttpTransport(self._url, auth=self._token)
        return Client(transport, timeout=self._timeout)

    async def search(self, query: str, top_k: int) -> list[Passage]:
        async with self._client() as client:
            result = await client.call_tool(
                "search_medical_knowledge", {"query": query, "top_k": top_k}
            )
        hits: list[dict[str, Any]] = (result.structured_content or {}).get("result", [])
        return [Passage.model_validate(hit) for hit in hits]

    async def answer_instructions(self, question: str, passages: str) -> str:
        async with self._client() as client:
            prompt = await client.get_prompt(
                "grounded_answer", {"question": question, "passages": passages}
            )
        content = prompt.messages[0].content
        return str(getattr(content, "text", ""))
