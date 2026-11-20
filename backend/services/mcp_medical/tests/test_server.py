import json
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport
from fastmcp.exceptions import ToolError
from mcp.types import TextContent, TextResourceContents

from mcp_medical.knowledge import KnowledgeBase
from mcp_medical.server import DISCLAIMER, MAX_TOP_K, DatasetCard, build_server

pytestmark = pytest.mark.anyio

McpClient = Client[FastMCPTransport]
TOOLS = ["get_passage", "list_topics", "search_medical_knowledge"]


@pytest.fixture
def client(knowledge: KnowledgeBase, card: DatasetCard) -> McpClient:
    """Talks real MCP to the server, in memory: no network needed."""
    return Client(build_server(knowledge, card))


async def call(client: McpClient, tool: str, arguments: dict[str, Any]) -> Any:
    async with client:
        result = await client.call_tool(tool, arguments)
    assert result.structured_content is not None
    return result.structured_content


async def search(client: McpClient, **arguments: Any) -> list[dict[str, Any]]:
    content = await call(client, "search_medical_knowledge", arguments)
    hits: list[dict[str, Any]] = content["result"]
    return hits


async def read_text(client: McpClient, uri: str) -> str:
    async with client:
        (content,) = await client.read_resource(uri)
    assert isinstance(content, TextResourceContents)
    return content.text


async def test_exposes_exactly_three_read_only_tools(client: McpClient) -> None:
    async with client:
        tools = await client.list_tools()

    assert sorted(t.name for t in tools) == TOOLS
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)


async def test_search_returns_the_best_answer_first(client: McpClient) -> None:
    hits = await search(client, query="diabetes insulin treatment", top_k=2)

    assert hits[0]["answer_id"] == "d1"
    assert hits[0]["source"] == "MedQuAD"
    assert 0 < hits[0]["score"] <= 1
    assert len(hits) == 2


async def test_search_can_filter_by_question_type(client: McpClient) -> None:
    hits = await search(client, query="asthma", qtype="treatment")

    assert {h["qtype"] for h in hits} == {"treatment"}
    assert hits[0]["answer_id"] == "a2"


async def test_search_returns_one_passage_per_answer(knowledge: KnowledgeBase) -> None:
    hits = await knowledge.search("asthma wheezing inhalers", top_k=10)

    assert len({h.answer_id for h in hits}) == len(hits)


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "", "top_k": 5},
        {"query": "asthma", "top_k": 0},
        {"query": "asthma", "top_k": MAX_TOP_K + 1},
        {"query": "x" * 2001},
    ],
)
async def test_search_rejects_out_of_range_arguments(
    client: McpClient, arguments: dict[str, Any]
) -> None:
    with pytest.raises(ToolError):
        await search(client, **arguments)


async def test_search_rejects_an_unknown_question_type(client: McpClient) -> None:
    with pytest.raises(ToolError, match="list_topics"):
        await search(client, query="x", qtype="astrology")


async def test_get_passage_returns_the_full_answer(client: McpClient) -> None:
    answer = await call(client, "get_passage", {"answer_id": "f1"})

    assert answer == {
        "answer_id": "f1",
        "question": "What causes influenza?",
        "qtype": "causes",
        "text": "Influenza is caused by a virus.",
        "source": "MedQuAD",
    }


async def test_get_passage_with_an_unknown_id_fails(client: McpClient) -> None:
    with pytest.raises(ToolError, match="no answer"):
        await call(client, "get_passage", {"answer_id": "nope"})


async def test_list_topics_counts_answers(client: McpClient) -> None:
    topics = await call(client, "list_topics", {})

    assert topics == {"treatment": 2, "symptoms": 1, "causes": 1}


async def test_resources_give_the_dataset_card_and_disclaimer(client: McpClient) -> None:
    card = json.loads(await read_text(client, "ragmed://dataset/card"))

    assert card["licence"] == "CC BY 4.0"
    assert await read_text(client, "ragmed://disclaimer") == DISCLAIMER


async def test_grounded_answer_prompt_carries_the_rules(client: McpClient) -> None:
    async with client:
        prompt = await client.get_prompt(
            "grounded_answer", {"question": "What is asthma?", "passages": "[1] Asthma is..."}
        )

    content = prompt.messages[0].content
    assert isinstance(content, TextContent)
    assert "ONLY the numbered passages" in content.text
    assert "[1] Asthma is..." in content.text
    assert "What is asthma?" in content.text
    assert DISCLAIMER in content.text
    assert "medication dose" in content.text
