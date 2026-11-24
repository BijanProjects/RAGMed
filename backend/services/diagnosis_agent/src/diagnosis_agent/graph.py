"""Corrective RAG: search, answer only if the evidence is enough, otherwise search again once.

    rewrite → retrieve → generate ─ answerable ───────────▶ check → END
       ▲                     └ not answerable, first try ─┐
       └──────────────────────────────────────────────────┘
                             └ not answerable, second try ▶ insufficient → END

The answer step also judges the evidence (`answerable`), to stay within Groq's free tier.
"""

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from diagnosis_agent.knowledge import Knowledge
from diagnosis_agent.reasoner import Reasoner
from diagnosis_agent.schema import (
    Citation,
    DiagnosisAnswer,
    DiagnosisResult,
    Passage,
    Urgency,
)
from ragmed_core.clinic import GENERAL_PHYSICIAN

TOP_K = 5
CONDITION_K = 2
MAX_PASSAGES = 6
MAX_SEARCHES = 2
NO_EVIDENCE_REASON = "No matching sources were found, so a full assessment is needed."
INSUFFICIENT = (
    "I do not have enough information in my medical sources to answer this. Please consult "
    "a medical professional, who can look at your situation directly."
)
# A number followed by a dose unit, e.g. "500 mg", "2.5ml", "10 units". Sentences with one are
# removed whatever the model says: RAGMed never suggests a medication dose.
_DOSE = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:mg|mcg|µg|g|ml|mL|IU|units?)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
MIN_LABEL = 20  # questions shorter than this rarely say what they are about


class State(TypedDict, total=False):
    question: str
    query: str
    condition: str
    searches: int
    passages: list[Passage]
    draft: DiagnosisAnswer | None
    result: DiagnosisResult


def merge_passages(first: list[Passage], then: list[Passage]) -> list[Passage]:
    """Both lists in order, each answer once, at most `MAX_PASSAGES`."""
    seen: set[str] = set()
    merged: list[Passage] = []
    for passage in [*first, *then]:
        if passage.answer_id not in seen:
            seen.add(passage.answer_id)
            merged.append(passage)
    return merged[:MAX_PASSAGES]


def number_passages(passages: list[Passage], only: set[int] | None = None) -> str:
    """Passages as "[n] (type) question\ntext"; `only` keeps just those numbers."""
    return "\n\n".join(
        f"[{n}] ({p.qtype}) {p.question}\n{p.text}"
        for n, p in enumerate(passages, start=1)
        if only is None or n in only
    )


_URGENCY_ORDER: list[Urgency] = ["self_care", "routine", "urgent", "emergency"]
EXCERPT_CHARS = 1500


def cite(number: int, passage: Passage) -> Citation:
    """A source as readers see it: its title and the text that was read."""
    return Citation(
        number=number,
        answer_id=passage.answer_id,
        question=label(passage),
        excerpt=passage.text[:EXCERPT_CHARS],
    )


def label(passage: Passage) -> str:
    """How to name a source. Some MedQuAD questions lack their subject ("Who is at risk for?"),
    so those show the passage's opening words instead."""
    question = passage.question.strip()
    if re.search(r"\b(for|of|about|is \(are\))\s*\?$", question) or len(question) < MIN_LABEL:
        opening = " ".join(passage.text.split()[:12])
        return f"{question} {opening}…"
    return question


def remove_doses(text: str) -> str:
    return " ".join(s for s in _SENTENCE.split(text) if not _DOSE.search(s)).strip()


def _normal(text: str) -> str:
    return " ".join(text.casefold().split()).strip(" .")


def remove_sentences(text: str, unsupported: list[str]) -> str:
    """Drop the sentences a grounding check flagged, matching loosely on case and spacing."""
    flagged = [_normal(u) for u in unsupported if _normal(u)]
    kept = [
        s
        for s in _SENTENCE.split(text)
        if not any(f in _normal(s) or _normal(s) in f for f in flagged if _normal(s))
    ]
    return " ".join(kept).strip()


def insufficient_result(
    query: str, urgency: Urgency = "routine", draft: DiagnosisAnswer | None = None
) -> DiagnosisResult:
    """No usable answer: always advise a clinician, and never lower an urgency already found."""
    floor = max(urgency, "routine", key=_URGENCY_ORDER.index)
    return DiagnosisResult(
        answer=INSUFFICIENT,
        citations=[],
        possible_conditions=[],
        urgency=floor,
        recommend_visit=True,
        # Without evidence, a general physician sees them first, with time to assess.
        clinician_type=draft.clinician_type if draft else GENERAL_PHYSICIAN,
        visit_minutes=draft.visit_minutes if draft else 30,
        visit_reason=draft.visit_reason if draft else NO_EVIDENCE_REASON,
        sufficient_evidence=False,
        search_query=query,
    )


def finalise(draft: DiagnosisAnswer, passages: list[Passage], query: str) -> DiagnosisResult:
    """Apply the rules that must hold whatever the model returned."""
    answer = remove_doses(draft.answer)
    if not answer:
        return insufficient_result(query, draft.urgency, draft)
    numbers = sorted({n for n in draft.citations if 1 <= n <= len(passages)})
    return DiagnosisResult(
        answer=answer,
        citations=[cite(n, passages[n - 1]) for n in numbers],
        possible_conditions=draft.possible_conditions,
        urgency=draft.urgency,
        recommend_visit=draft.recommend_visit or draft.urgency in ("urgent", "emergency"),
        clinician_type=draft.clinician_type,
        visit_minutes=draft.visit_minutes,
        visit_reason=draft.visit_reason,
        sufficient_evidence=True,
        search_query=query,
    )


def build_graph(knowledge: Knowledge, reasoner: Reasoner) -> CompiledStateGraph[Any]:
    async def rewrite(state: State) -> State:
        previous = state.get("query") if state.get("searches") else None
        search = await reasoner.search_query(state["question"], previous)
        return {
            "query": search.query,
            "condition": search.condition.strip(),
            "searches": state.get("searches", 0) + 1,
        }

    async def retrieve(state: State) -> State:
        passages = await knowledge.search(state["query"], TOP_K)
        if state.get("condition"):
            # Detailed queries can rank rare variants above the common condition's own page
            # (e.g. "migraine with aura" finds hemiplegic migraine first), so its plain name is
            # searched too, and those passages lead.
            named = await knowledge.search(state["condition"], CONDITION_K)
            passages = merge_passages(named, passages)
        return {"passages": passages}

    async def generate(state: State) -> State:
        if not state["passages"]:
            return {"draft": None}
        instructions = await knowledge.answer_instructions(
            state["question"], number_passages(state["passages"])
        )
        return {"draft": await reasoner.answer(instructions)}

    def after_generate(state: State) -> str:
        draft = state["draft"]
        if draft is not None and draft.answerable:
            return "check"
        return "rewrite" if state["searches"] < MAX_SEARCHES else "insufficient"

    async def check(state: State) -> State:
        # Only the answer text is checked and trimmed, against just the passages it cites;
        # triage comes from the draft, so a check can never lower the urgency.
        draft = state["draft"]
        assert draft is not None  # noqa: S101 - after_generate only routes here with a draft
        cited = number_passages(state["passages"], only=set(draft.citations))
        verdict = await reasoner.check(draft.answer, cited or number_passages(state["passages"]))
        trimmed = draft.model_copy(
            update={"answer": remove_sentences(draft.answer, verdict.unsupported)}
        )
        return {"result": finalise(trimmed, state["passages"], state["query"])}

    async def insufficient(state: State) -> State:
        draft = state.get("draft")
        urgency = draft.urgency if draft else "routine"
        return {"result": insufficient_result(state["query"], urgency, draft)}

    graph = StateGraph(State)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("check", check)
    graph.add_node("insufficient", insufficient)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", after_generate, ["check", "rewrite", "insufficient"])
    graph.add_edge("check", END)
    graph.add_edge("insufficient", END)
    return graph.compile()
