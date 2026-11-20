"""The clinic conversation: screen, interview, assess, report, and book.

    screen ─ red flag ─────────────────────────────────────────────▶ emergency
       └ dispatch ─ a button (show times, book) ─────────────────────▶ act
                  ├ intake under way ─▶ interview ─ more to ask ─────▶ (question)
                  │                          └ enough ─▶ assess ─────▶ (draft report + offer)
                  └ route ─ symptoms or booking ─▶ interview
                          ├ general question ─▶ answer (Diagnosis agent, cited)
                          └ greeting / off-topic ─▶ canned
    every path ends in remember → END

Emergency screening runs on every message, including mid-interview. Conversation state is saved
per thread (the A2A context id).
"""

import operator
from collections.abc import AsyncIterator
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from coordinator_agent.emergency import EMERGENCY_GUIDANCE, screen
from coordinator_agent.intake import (
    FINAL_QUESTION,
    MAX_QUESTIONS,
    Interviewer,
    case_summary,
    is_repeat,
    merge,
    quick_replies,
)
from coordinator_agent.reply import (
    BOOKING_START,
    DISCLAIMER,
    GREETING,
    OUT_OF_SCOPE,
    Offer,
    Reply,
    build_report,
    from_diagnosis,
    report_reply,
)
from coordinator_agent.router import Router
from coordinator_agent.scheduling import Scheduling, SchedulingError
from ragmed_a2a.client import AgentEvent, Progress
from ragmed_a2a.client import Reply as AgentReply
from ragmed_core.clinic import CLINICIAN_TYPES, GENERAL_PHYSICIAN

HISTORY_TURNS = 6
HISTORY_CHARS = 600
UNAVAILABLE = "I could not get an answer just now. Please try again in a moment."
NO_REPORT = "There is no report to book a visit for yet. Tell me what's bothering you first."


class Diagnosis(Protocol):
    def stream(self, text: str) -> AsyncIterator[AgentEvent]: ...


class Turn(TypedDict):
    role: str
    text: str


class State(TypedDict, total=False):
    # Input for this turn
    message: str
    action: dict[str, Any] | None
    session: str
    # Saved across turns
    history: Annotated[list[Turn], operator.add]
    phase: str  # "chat" | "intake" | "report" | "booked"
    case: dict[str, str]
    interview: list[Turn]
    questions: int
    asked_final: bool
    report: dict[str, Any] | None
    offer: dict[str, Any] | None
    # This turn's outcome
    intent: str
    question: str
    reply: dict[str, Any]


def history_text(turns: list[Turn], limit: int = HISTORY_TURNS) -> str:
    return "\n".join(f"{t['role']}: {t['text'][:HISTORY_CHARS]}" for t in turns[-limit:])


async def _ask_diagnosis(diagnosis: Diagnosis, text: str) -> AgentReply | None:
    write = get_stream_writer()
    final: AgentReply | None = None
    async for event in diagnosis.stream(text):
        if isinstance(event, Progress):
            write({"progress": event.text})
        else:
            final = event
    return final


def build_graph(  # noqa: PLR0915 - one closure per step, next to the services it uses
    router: Router,
    interviewer: Interviewer,
    diagnosis: Diagnosis,
    scheduling: Scheduling,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any]:
    async def screen_node(state: State) -> State:
        flag = screen(state.get("message", ""))
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

    def dispatch(state: State) -> str:
        if state["intent"] == "emergency":
            return "remember"
        if state.get("action"):
            return "act"
        return "interview" if state.get("phase") == "intake" else "route"

    async def route_node(state: State) -> State:
        route = await router.route(state["message"], history_text(state.get("history", [])))
        update: State = {"intent": route.intent, "question": route.question or state["message"]}
        if route.intent == "symptoms":
            update |= _new_intake()
        elif route.intent == "book_appointment":
            update |= _new_intake()
            update["reply"] = Reply(kind="question", text=BOOKING_START).model_dump()
            update["interview"] = [{"role": "assistant", "text": BOOKING_START}]
        return update

    def after_route(state: State) -> str:
        if state["intent"] == "symptoms":
            return "interview"
        if state["intent"] == "book_appointment":
            return "remember"
        return "answer" if state["intent"] == "medical_question" else "canned"

    async def interview(state: State) -> State:
        said: Turn = {"role": "patient", "text": state["message"]}
        transcript: list[Turn] = [*state.get("interview", []), said]
        questions = state.get("questions", 0)
        known = dict(state.get("case", {}))
        if not known.get("main_problem"):  # the patient's own first words, never our question
            known["main_problem"] = next(t["text"] for t in transcript if t["role"] == "patient")
        turn = await interviewer.next_turn(known, history_text(transcript, 40))
        case = merge(known, turn.updates)
        if state.get("asked_final"):
            if not _nothing_more(state["message"]):
                case["other_notes"] = state["message"]
            return {"case": case, "interview": transcript, "phase": "assessing"}

        asked = [t["text"] for t in transcript if t["role"] == "assistant"]
        done = (
            turn.complete
            or not turn.next_question.strip()
            or questions >= MAX_QUESTIONS
            or is_repeat(turn.next_question, asked)  # a third try at the same question
        )
        if done:  # always end with the open question
            question, replies, asked_final = FINAL_QUESTION, ["No, that's everything"], True
        else:
            question, replies, asked_final = turn.next_question.strip(), turn.quick_replies, False
        reply = Reply(kind="question", text=question, quick_replies=quick_replies(replies))
        return {
            "case": case,
            "interview": [*transcript, {"role": "assistant", "text": question}],
            "questions": questions + 1,
            "asked_final": asked_final,
            "reply": reply.model_dump(),
        }

    def after_interview(state: State) -> str:
        return "assess" if state.get("phase") == "assessing" else "remember"

    async def assess(state: State) -> State:
        get_stream_writer()({"progress": "Preparing your draft report"})
        final = await _ask_diagnosis(diagnosis, case_summary(state["case"]))
        if not (final and final.state == "completed" and final.data):
            text = (final.text if final and final.text else UNAVAILABLE) + f"\n\n_{DISCLAIMER}_"
            return {"phase": "intake", "reply": Reply(kind="no_answer", text=text).model_dump()}
        report = build_report(state["case"], final.data[0])
        reply = report_reply(report, EMERGENCY_GUIDANCE)
        offer = reply.offer.model_dump() if reply.offer else None
        return {"phase": "report", "report": report, "offer": offer, "reply": reply.model_dump()}

    async def act(state: State) -> State:
        action = state.get("action") or {}
        offer = state.get("offer")
        if not offer or not state.get("report"):
            return {"reply": Reply(kind="no_answer", text=NO_REPORT).model_dump()}
        kind = str(action.get("clinician_type") or offer["clinician_type"])
        if kind not in CLINICIAN_TYPES or kind not in (offer["clinician_type"], GENERAL_PHYSICIAN):
            kind = offer["clinician_type"]  # only the recommended clinician or a GP
        try:
            if action.get("type") == "book":
                return await _book(state, offer, kind, str(action.get("start", "")))
            slots = await scheduling.find_slots(kind, int(offer["visit_minutes"]))
        except SchedulingError as error:
            return {"reply": Reply(kind="no_answer", text=str(error)).model_dump()}
        text = (
            f"Here are open times for a {offer['visit_minutes']}-minute visit with {kind}. "
            "Pick one to book it."
            if slots
            else f"There are no open times with {kind} in the next week. Please call the clinic."
        )
        chosen = Offer(**{**offer, "clinician_type": kind})
        return {"reply": Reply(kind="slots", text=text, slots=slots, offer=chosen).model_dump()}

    async def _book(state: State, offer: dict[str, Any], kind: str, start: str) -> State:
        report = {**(state.get("report") or {}), "clinician_type": kind}
        booked = await scheduling.book(
            clinician_type=kind,
            visit_minutes=int(offer["visit_minutes"]),
            start=start,
            session_id=state["session"],
            urgency=offer["urgency"] if offer["urgency"] != "emergency" else "urgent",
            report=report,
        )
        text = (
            f"**Booked.** Your {booked['minutes']}-minute visit with {booked['physician']} "
            f"({booked['clinician_type']}) is confirmed. Your reference is "
            f"**{booked['reference']}**. Your draft report has been sent to the physician."
        )
        reply = Reply(kind="booked", text=text, appointment=booked)
        return {"phase": "booked", "reply": reply.model_dump()}

    async def answer(state: State) -> State:
        final = await _ask_diagnosis(diagnosis, state["question"])
        if final and final.state == "completed" and final.data:
            return {"reply": from_diagnosis(final.data[0], EMERGENCY_GUIDANCE).model_dump()}
        text = (final.text if final and final.text else UNAVAILABLE) + f"\n\n_{DISCLAIMER}_"
        return {"reply": Reply(kind="no_answer", text=text).model_dump()}

    async def canned(state: State) -> State:
        greeting = state["intent"] == "greeting"
        reply = Reply(
            kind="greeting" if greeting else "out_of_scope",
            text=GREETING if greeting else OUT_OF_SCOPE,
        )
        return {"reply": reply.model_dump()}

    async def remember(state: State) -> State:
        said = state.get("message") or f"[{(state.get('action') or {}).get('type', 'action')}]"
        turns: list[Turn] = [
            {"role": "patient", "text": said},
            {"role": "assistant", "text": state["reply"]["text"]},
        ]
        return {"history": turns, "action": None}

    graph = StateGraph(State)
    for name, node in [
        ("screen", screen_node),
        ("route", route_node),
        ("interview", interview),
        ("assess", assess),
        ("act", act),
        ("answer", answer),
        ("canned", canned),
        ("remember", remember),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "screen")
    graph.add_conditional_edges("screen", dispatch, ["remember", "act", "interview", "route"])
    graph.add_conditional_edges("route", after_route, ["interview", "remember", "answer", "canned"])
    graph.add_conditional_edges("interview", after_interview, ["assess", "remember"])
    for name in ("assess", "act", "answer", "canned"):
        graph.add_edge(name, "remember")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer)


def _nothing_more(text: str) -> bool:
    """True for answers like "No, that's everything" to the final question."""
    words = " ".join(text.casefold().replace("\u2019", "'").split()).strip(" .!")
    return words in {"no", "nope", "nothing", "none", "no thanks", "that's all", "that's it"} or (
        words.startswith("no") and ("everything" in words or "that's all" in words)
    )


def _new_intake() -> State:
    return {
        "phase": "intake",
        "case": {},
        "interview": [],
        "questions": 0,
        "asked_final": False,
        "report": None,
        "offer": None,
    }
