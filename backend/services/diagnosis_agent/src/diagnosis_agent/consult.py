"""The `consult` skill: a quick, grounded working picture while the intake is still under way.

    search → weigh → END

The Coordinator asks for one on each intake turn and uses it to choose its next question, so it
stays cheap: one small-model call on shortened passages (plus a query rewrite on the first turn,
before there are leads to search for by name). Every lead must cite a passage it came from.
"""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from diagnosis_agent.graph import cite, label, merge_passages
from diagnosis_agent.knowledge import Knowledge
from diagnosis_agent.reasoner import Reasoner
from diagnosis_agent.schema import (
    ConsultDraft,
    ConsultLead,
    ConsultResult,
    Passage,
)

TOP_K = 4
LEAD_K = 1
MAX_LEADS = 3
MAX_ITEMS = 3
PASSAGE_CHARS = 600
MAX_QUERY_CHARS = 400


class ConsultRequest(BaseModel):
    """What the Coordinator sends as the A2A data part {"consult": {...}}."""

    case: str = Field(max_length=4000, description="The intake so far, one fact per line.")
    complaint: str = Field(max_length=2000, description="The patient's own first words.")
    latest: str = Field(default="", max_length=2000, description="Their latest answer.")
    leads: list[str] = Field(default_factory=list, description="Leads from the last consult.")


class State(TypedDict, total=False):
    request: ConsultRequest
    query: str
    passages: list[Passage]
    result: ConsultResult


def shorten(passages: list[Passage]) -> str:
    """Numbered passages, each cut to `PASSAGE_CHARS`: enough to weigh, cheap enough per turn."""
    return "\n\n".join(
        f"[{n}] {label(p)}\n{p.text[:PASSAGE_CHARS]}" for n, p in enumerate(passages, start=1)
    )


def ground(draft: ConsultDraft, passages: list[Passage], query: str) -> ConsultResult:
    """Keep only leads that cite a real passage, and short lists."""
    leads: list[ConsultLead] = []
    for lead in draft.leads:
        numbers = sorted({n for n in lead.citations if 1 <= n <= len(passages)})
        if not numbers or not lead.condition.strip():
            continue  # a lead the knowledge base does not describe is a guess: drop it
        sources = [cite(n, passages[n - 1]) for n in numbers]
        leads.append(ConsultLead(condition=lead.condition.strip(), fits=lead.fits, sources=sources))
    return ConsultResult(
        leads=leads[:MAX_LEADS],
        ask_next=[a for a in draft.ask_next if a.strip()][:MAX_ITEMS],
        warning_signs=[w for w in draft.warning_signs if w.strip()][:MAX_ITEMS],
        urgency=draft.urgency,
        enough=draft.enough and bool(leads),
        search_query=query,
        passages_read=len(passages),
    )


def nothing_found(query: str) -> ConsultResult:
    return ConsultResult(
        leads=[],
        ask_next=[],
        warning_signs=[],
        urgency="routine",
        enough=False,
        search_query=query,
        passages_read=0,
    )


def build_consult_graph(knowledge: Knowledge, reasoner: Reasoner) -> CompiledStateGraph[Any]:
    async def search(state: State) -> State:
        request = state["request"]
        leads = [lead.strip() for lead in request.leads if lead.strip()][:MAX_LEADS]
        if leads:
            # Search by the leads' names (the knowledge base files answers under condition
            # names) and by what the patient has said, so a new symptom can bring a new lead.
            query = f"{request.complaint}. {request.latest}".strip(" .")[:MAX_QUERY_CHARS]
        else:
            rewritten = await reasoner.search_query(request.case, None)
            query = rewritten.query
            leads = [rewritten.condition.strip()] if rewritten.condition.strip() else []
        named: list[Passage] = []
        for lead in leads:
            named = merge_passages(named, await knowledge.search(lead, LEAD_K))
        passages = merge_passages(named, await knowledge.search(query, TOP_K))
        return {"query": query, "passages": passages}

    async def weigh(state: State) -> State:
        if not state["passages"]:
            return {"result": nothing_found(state["query"])}
        draft = await reasoner.consult(state["request"].case, shorten(state["passages"]))
        return {"result": ground(draft, state["passages"], state["query"])}

    graph = StateGraph(State)
    graph.add_node("search", search)
    graph.add_node("weigh", weigh)
    graph.add_edge(START, "search")
    graph.add_edge("search", "weigh")
    graph.add_edge("weigh", END)
    return graph.compile()
