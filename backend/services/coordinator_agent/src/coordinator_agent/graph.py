"""The conversation: screen for emergencies, route, ask the Diagnosis agent, reply.

    screen ─ red flag ──────────────────────────────▶ emergency ─┐
       └ none ▶ route ─ medical ▶ diagnose ──────────────────────┤
                   ├ greeting / booking / off-topic ▶ canned ────┤
                                                                 ▼
                                                             remember → END

Conversation state is saved per thread (the A2A context id), so follow-up questions work.
"""

import operator
from collections.abc import AsyncIterator
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from coordinator_agent.emergency import EMERGENCY_GUIDANCE, screen
from coordinator_agent.reply import (
    BOOKING,
    DISCLAIMER,
    GREETING,
    OUT_OF_SCOPE,
    Reply,
    from_diagnosis,
)
from coordinator_agent.router import Router
from ragmed_a2a.client import AgentEvent, Progress
from ragmed_a2a.client import Reply as AgentReply

HISTORY_TURNS = 6
HISTORY_CHARS = 600
UNAVAILABLE = "I could not get an answer just now. Please try again in a moment."


class Diagnosis(Protocol):
    def stream(self, text: str) -> AsyncIterator[AgentEvent]: ...


class Turn(TypedDict):
    role: str
    text: str


class State(TypedDict, total=False):
    message: str
    history: Annotated[list[Turn], operator.add]
    intent: str
    question: str
    reply: dict[str, Any]


def history_text(history: list[Turn]) -> str:
    recent = history[-HISTORY_TURNS:]
    return "\n".join(f"{t['role']}: {t['text'][:HISTORY_CHARS]}" for t in recent)


def build_graph(
    router: Router, diagnosis: Diagnosis, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> CompiledStateGraph[Any]:
    async def screen_node(state: State) -> State:
        flag = screen(state["message"])
        if flag is None:
            return {"intent": ""}
        reply = Reply(
            kind="emergency",
            text=f"{flag.guidance}\n\n_{DISCLAIMER}_",
            urgency="emergency",
            recommend_visit=True,
            red_flag=flag.category,
        )
        return {"intent": "emergency", "reply": reply.model_dump()}

    def after_screen(state: State) -> str:
        return "remember" if state["intent"] == "emergency" else "route"

    async def route_node(state: State) -> State:
        route = await router.route(state["message"], history_text(state.get("history", [])))
        return {"intent": route.intent, "question": route.question or state["message"]}

    def after_route(state: State) -> str:
        return "diagnose" if state["intent"] == "medical_question" else "canned"

    async def diagnose(state: State) -> State:
        write = get_stream_writer()
        final: AgentReply | None = None
        async for event in diagnosis.stream(state["question"]):
            if isinstance(event, Progress):
                write({"progress": event.text})
            else:
                final = event
        if final and final.state == "completed" and final.data:
            # If the Diagnosis agent judges it an emergency, the reply leads with that guidance.
            return {"reply": from_diagnosis(final.data[0], EMERGENCY_GUIDANCE).model_dump()}
        text = (final.text if final and final.text else UNAVAILABLE) + f"\n\n_{DISCLAIMER}_"
        return {"reply": Reply(kind="no_answer", text=text).model_dump()}

    async def canned(state: State) -> State:
        texts = {"greeting": GREETING, "book_appointment": BOOKING}
        kinds = {"greeting": "greeting", "book_appointment": "booking"}
        intent = state["intent"]
        reply = Reply(
            kind=kinds.get(intent, "out_of_scope"),  # type: ignore[arg-type]
            text=texts.get(intent, OUT_OF_SCOPE),
        )
        return {"reply": reply.model_dump()}

    async def remember(state: State) -> State:
        turns: list[Turn] = [
            {"role": "patient", "text": state["message"]},
            {"role": "assistant", "text": state["reply"]["text"]},
        ]
        return {"history": turns}

    graph = StateGraph(State)
    graph.add_node("screen", screen_node)
    graph.add_node("route", route_node)
    graph.add_node("diagnose", diagnose)
    graph.add_node("canned", canned)
    graph.add_node("remember", remember)
    graph.add_edge(START, "screen")
    graph.add_conditional_edges("screen", after_screen, ["remember", "route"])
    graph.add_conditional_edges("route", after_route, ["diagnose", "canned"])
    graph.add_edge("diagnose", "remember")
    graph.add_edge("canned", "remember")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer)
