"""The MCP server: a small, fixed set of read-only tools over the knowledge base.

An agent can do exactly what is listed here and nothing else: search, read one answer, and list
the question types. It cannot write, delete or reach the database directly.
"""

import json
from dataclasses import asdict, dataclass
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import TokenVerifier
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_medical.knowledge import SOURCE, KnowledgeBase

MAX_TOP_K = 10
MAX_QUERY_CHARS = 2000
PROMPT_VERSION = "2"

DISCLAIMER = (
    "RAGMed is an educational project, not a medical device. This information is not a "
    "diagnosis and does not replace a clinician. In an emergency, call your local emergency "
    "number."
)

GROUNDED_ANSWER = """You are a careful medical information assistant. Follow these rules:

- Answer only medical questions. For anything else, say you can only help with medical questions.
- Use ONLY the numbered passages below. Do not add facts that are not stated in them.
- Cite every claim with the passage number in square brackets, like [2].
- If the passages do not answer the question, say: "I do not have enough information to answer
  this. Please consult a medical professional."
- Never suggest a medication dose.
- If the question has several parts, answer each part in turn.
- End with this disclaimer: {disclaimer}

Passages:
{passages}

Question:
{question}
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)


@dataclass(frozen=True, slots=True)
class DatasetCard:
    name: str
    licence: str
    citation: str
    url: str
    answers: int
    passages: int
    embedding_model: str
    indexed_on: str


def build_server(
    knowledge: KnowledgeBase, card: DatasetCard, *, auth: TokenVerifier | None = None
) -> FastMCP:
    mcp = FastMCP(
        "ragmed-medical-knowledge",
        instructions=(
            "Search a curated medical knowledge base (MedQuAD, from NIH sources). Cite the "
            "passage ids you use and always include the ragmed://disclaimer text."
        ),
        auth=auth,
    )

    @mcp.tool(annotations=READ_ONLY)
    async def search_medical_knowledge(
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=MAX_QUERY_CHARS,
                description="A self-contained medical question or search query.",
            ),
        ],
        top_k: Annotated[
            int, Field(ge=1, le=MAX_TOP_K, description="How many answers to return.")
        ] = 5,
        qtype: Annotated[
            str | None,
            Field(description="Only return this question type; see list_topics."),
        ] = None,
    ) -> list[dict[str, Any]]:
        """Find the passages that best answer a medical question, best first.

        Each result has an id to cite, the question it answers, its type, the passage text and
        a similarity score between 0 and 1.
        """
        if qtype is not None and qtype not in knowledge.topics:
            raise ToolError(f"unknown qtype {qtype!r}; see list_topics")
        return [asdict(p) for p in await knowledge.search(query, top_k, qtype)]

    @mcp.tool(annotations=READ_ONLY)
    def get_passage(
        answer_id: Annotated[str, Field(description="The answer_id from a search result.")],
    ) -> dict[str, str]:
        """Return one full answer, with its question and type."""
        answer = knowledge.answer(answer_id)
        if answer is None:
            raise ToolError(f"no answer with id {answer_id!r}")
        return {
            "answer_id": answer.id,
            "question": answer.question,
            "qtype": answer.qtype,
            "text": answer.answer,
            "source": SOURCE,
        }

    @mcp.tool(annotations=READ_ONLY)
    def list_topics() -> dict[str, int]:
        """The question types in the knowledge base and how many answers each has."""
        return knowledge.topics

    @mcp.resource("ragmed://dataset/card", mime_type="application/json")
    def dataset_card() -> str:
        """Where the knowledge comes from, its licence and when it was indexed."""
        return json.dumps(asdict(card), indent=2)

    @mcp.resource("ragmed://disclaimer")
    def disclaimer() -> str:
        """Text to append to every answer."""
        return DISCLAIMER

    @mcp.prompt(name="grounded_answer", meta={"version": PROMPT_VERSION})
    def grounded_answer(question: str, passages: str) -> str:
        """Answer a question using only the given numbered passages, with citations."""
        return GROUNDED_ANSWER.format(disclaimer=DISCLAIMER, passages=passages, question=question)

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "answers": knowledge.size})

    return mcp
