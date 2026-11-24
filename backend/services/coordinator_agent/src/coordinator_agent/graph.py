"""The clinic conversation: screen, interview, assess, report, and book.

    screen ─ red flag ─────────────────────────────────────────────▶ emergency
       └ dispatch ─ a button (show times, book) ─────────────────────▶ act
                  ├ after an emergency ─▶ support ─ still happening ─▶ (stay with them)
                  │                              └ it has passed ────▶ (resume)
                  ├ intake under way ─▶ interview ─ more to ask ─────▶ (question)
                  │                          └ enough ─▶ assess ─────▶ (report + handoff + offer)
                  └ route ─ symptoms or booking ─▶ interview
                          ├ general question ─▶ answer (Diagnosis agent, cited)
                          └ greeting / off-topic ─▶ canned
    every path ends in remember → END

Booking takes two steps: choosing a time emails a one-time code to the patient, and the visit is
booked only once that code is entered (see verification.py).

Each interview turn, the Coordinator first asks the Diagnosis agent for a consult (its working
picture from the knowledge base), then decides the next question from it. At the end it writes a
handoff for the clinician with an urgency score. Every step is reported with the agent doing it,
so the web app can show the care team at work.

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
    WARNING_CHECK,
    Interviewer,
    briefing,
    case_lines,
    case_summary,
    checklist_question,
    classify_checklist,
    is_repeat,
    lead_names,
    merge,
    natural,
    quick_replies,
)
from coordinator_agent.limits import Limits, booking_limit
from coordinator_agent.mailer import ConsoleMailer, Mailer, MailError
from coordinator_agent.reply import (
    BOOKING_START,
    DISCLAIMER,
    GREETING,
    OUT_OF_SCOPE,
    Offer,
    Reply,
    at_least,
    build_report,
    from_diagnosis,
    report_reply,
)
from coordinator_agent.router import Route, Router, is_clinic_request
from coordinator_agent.scheduling import Scheduling, SchedulingError
from coordinator_agent.verification import (
    CODE_MINUTES,
    EMAIL,
    MAX_ATTEMPTS,
    Pending,
    code_matches,
    is_expired,
    new_code,
    pending_booking,
    public_view,
    send_limit,
)
from ragmed_a2a.client import AgentEvent, Draft, Progress
from ragmed_a2a.client import Reply as AgentReply
from ragmed_core.clinic import (
    CLINICIAN_TYPES,
    COORDINATOR_AGENT,
    DIAGNOSIS_AGENT,
    GENERAL_PHYSICIAN,
    SCHEDULING_AGENT,
)
from ragmed_core.ratelimit import MemoryRateLimiter, RateLimiter

HISTORY_TURNS = 6
MAX_CONSULTS = 6  # knowledge-base consults per intake; each costs one small-model call
HISTORY_CHARS = 600
UNAVAILABLE = "I could not get an answer just now. Please try again in a moment."
ADDED = "Added after the report"
AFTER_ALARM = ["I've called for help", "It's not happening right now"]
NO_REPORT = "There is no report to book a visit for yet. Tell me what's bothering you first."


class Diagnosis(Protocol):
    def stream(
        self, text: str, *, data: dict[str, Any] | None = None
    ) -> AsyncIterator[AgentEvent]: ...


class Turn(TypedDict):
    role: str
    text: str


class State(TypedDict, total=False):
    # Input for this turn
    message: str
    action: dict[str, Any] | None
    session: str
    client: str
    # Saved across turns
    history: Annotated[list[Turn], operator.add]
    phase: str  # "chat" | "intake" | "emergency" | "report" | "booked"
    alert: dict[str, str] | None  # the red flag, and the phase to go back to once it passes
    signs_offered: list[str]  # the warning signs on the checklist
    signs_reasked: bool  # the checklist was asked again after an unclear answer
    signs_present: str  # the warning signs the patient reported
    case: dict[str, str]
    interview: list[Turn]
    questions: int
    asked: list[str]  # the questions alone, without acknowledgements
    asked_final: bool
    signs_checked: bool  # the warning-sign checklist has been asked
    consult: dict[str, Any] | None  # the Diagnosis agent's latest working picture
    consults: int
    report: dict[str, Any] | None
    offer: dict[str, Any] | None
    pending: Pending | None  # a booking waiting for its email code
    # This turn's outcome
    intent: str
    question: str
    reply: dict[str, Any]


def history_text(turns: list[Turn], limit: int = HISTORY_TURNS) -> str:
    return "\n".join(f"{t['role']}: {t['text'][:HISTORY_CHARS]}" for t in turns[-limit:])


def draft(text: str) -> None:
    """Stream a piece of the Coordinator's own reply to the web app as it is written."""
    get_stream_writer()({"draft": text})


def step(text: str, agent: str = COORDINATOR_AGENT) -> None:
    """Report what one member of the care team is doing (streamed to the web app)."""
    get_stream_writer()({"progress": text, "agent": agent})


async def _ask_diagnosis(
    diagnosis: Diagnosis, text: str, data: dict[str, Any] | None = None
) -> AgentReply | None:
    final: AgentReply | None = None
    write = get_stream_writer()
    async for event in diagnosis.stream(text, data=data):
        if isinstance(event, Progress):
            step(event.text, event.agent or DIAGNOSIS_AGENT)
        elif isinstance(event, Draft):
            write({"draft": event.text, "reset": event.reset, "label": event.label})
        else:
            final = event
    return final


def build_graph(  # noqa: PLR0915 - one closure per step, next to the services it uses
    router: Router,
    interviewer: Interviewer,
    diagnosis: Diagnosis,
    scheduling: Scheduling,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    limiter: RateLimiter | None = None,
    limits: Limits | None = None,
    mailer: Mailer | None = None,
) -> CompiledStateGraph[Any]:
    limiter = limiter or MemoryRateLimiter()
    limits = limits or Limits()
    mailer = mailer or ConsoleMailer()

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
            quick_replies=AFTER_ALARM,
        )
        phase = state.get("phase") or "chat"
        before = (state.get("alert") or {}).get("before", phase) if phase == "emergency" else phase
        alert = {"category": flag.category, "guidance": flag.guidance, "before": before}
        return {"intent": "emergency", "phase": "emergency", "alert": alert, **said(reply)}

    def dispatch(state: State) -> str:
        if state["intent"] == "emergency":
            return "remember"
        if state.get("action"):
            return "act"
        if state.get("phase") == "emergency":
            return "support"
        return "interview" if state.get("phase") == "intake" else "route"

    async def support(state: State) -> State:
        """After an emergency warning: stay with the patient and answer what they ask."""
        alert = state.get("alert") or {"category": "emergency", "guidance": "", "before": "chat"}
        turns: list[Turn] = [
            *state.get("history", []),
            {"role": "patient", "text": state["message"]},
        ]
        answer = await interviewer.support(alert["category"], history_text(turns, 8), draft)
        if answer.resume:
            phase = alert["before"] if alert["before"] in ("intake", "chat", "report") else "chat"
            asked = state.get("asked", [])
            again = f"\n\nLet's carry on. {asked[-1]}" if phase == "intake" and asked else ""
            reply = Reply(kind="question", text=f"{natural(answer.reply)}{again}")
            return {"phase": phase, "alert": None, **said(reply)}
        step("Staying with you: the emergency comes first")
        guidance = alert.get("guidance") or EMERGENCY_GUIDANCE
        reply = Reply(
            kind="emergency",
            text=f"{natural(answer.reply)}\n\n{guidance}",
            urgency="emergency",
            recommend_visit=True,
            red_flag=alert["category"],
            quick_replies=AFTER_ALARM,
        )
        return said(reply)

    async def route_node(state: State) -> State:
        route = await router.route(state["message"], history_text(state.get("history", [])))
        if route.intent == "out_of_scope" and is_clinic_request(state["message"]):
            route = Route(intent="symptoms", question="")
        if route.intent == "follow_up" and not state.get("report"):
            route = Route(intent="symptoms", question="")  # nothing to follow up on yet
        update: State = {"intent": route.intent, "question": route.question or state["message"]}
        if route.intent == "symptoms":
            update |= _new_intake()
        elif route.intent == "book_appointment":
            update |= _new_intake()
            update["reply"] = Reply(kind="question", text=BOOKING_START).model_dump()
            update["interview"] = [{"role": "assistant", "text": BOOKING_START}]
        return update

    def said(reply: Reply) -> State:
        return {"reply": reply.model_dump()}

    async def follow_up(state: State) -> State:
        """A message about the report just made: answer it there, and add any new detail."""
        report = dict(state.get("report") or {})
        turns: list[Turn] = [
            *state.get("history", []),
            {"role": "patient", "text": state["message"]},
        ]
        booked = state.get("phase") == "booked"
        summary = (
            f"{report.get('summary', '')}\nUrgency: {report.get('urgency', '')}; visit: "
            f"{report.get('visit_minutes', '')} minutes with {report.get('clinician_type', '')}\n"
            + (
                "The visit is booked."
                if booked
                else "Not booked yet: the patient can book it with the buttons under your reply."
            )
        )
        answer = await interviewer.follow_up(summary, history_text(turns, 6), draft)
        update: State = {}
        if answer.adds.strip():
            step("Added to your report for the clinician")
            case = {**report.get("case", {})}
            added = case.get(ADDED)
            case[ADDED] = f"{added}; {answer.adds.strip()}" if added else answer.adds.strip()
            report = {**report, "case": case}
            update["report"] = report
        offer = state.get("offer") if state.get("phase") == "report" else None
        reply = Reply(
            kind="answer",
            text=natural(answer.reply) or "Noted.",
            offer=Offer(**offer) if offer else None,
        )
        return {**update, **said(reply)}

    def after_route(state: State) -> str:
        if state["intent"] == "follow_up":
            return "follow_up"
        if state["intent"] == "symptoms":
            return "interview"
        if state["intent"] == "book_appointment":
            return "remember"
        return "answer" if state["intent"] == "medical_question" else "canned"

    async def consult(known: dict[str, str], latest: str, previous: Any) -> dict[str, Any] | None:
        """The Diagnosis agent's working picture of the case so far, or None if unavailable."""
        step("Asking the Diagnosis agent what the knowledge base suggests")
        request = {
            "case": case_lines(known)[:4000],
            "complaint": known["main_problem"][:2000],
            "latest": latest[:2000],
            "leads": lead_names(previous),
        }
        final = await _ask_diagnosis(diagnosis, "Intake consult", {"consult": request})
        if final and final.state == "completed" and final.data:
            return final.data[0]
        step("The knowledge base is unavailable; continuing with a careful general intake")
        return None

    async def interview(state: State) -> State:
        said: Turn = {"role": "patient", "text": state["message"]}
        transcript: list[Turn] = [*state.get("interview", []), said]
        questions = state.get("questions", 0)
        known = dict(state.get("case", {}))
        if not known.get("main_problem"):  # the patient's own first words, never our question
            known["main_problem"] = next(t["text"] for t in transcript if t["role"] == "patient")
        picture, consults = state.get("consult"), state.get("consults", 0)
        asked = state.get("asked", [])
        if not state.get("asked_final") and consults < MAX_CONSULTS:
            # The answer alone ("No, it builds up slowly") means little without its question.
            latest = f"{asked[-1]} {state['message']}" if asked else state["message"]
            picture, consults = await consult(known, latest, picture) or picture, consults + 1
        turn = await interviewer.next_turn(known, history_text(transcript, 40), briefing(picture))
        checklist = bool(asked) and asked[-1] == WARNING_CHECK
        offered = state.get("signs_offered", [])
        reading = classify_checklist(state["message"], offered)[0] if checklist else ""
        if reading == "unclear" and not state.get("signs_reasked"):
            # Not an answer to the checklist: record it as usual and ask the checklist once more.
            step("Your answer did not say which warning signs apply; asking once more")
            case = merge(known, turn.updates)
            text = f"{natural(turn.acknowledgement)} {checklist_question(case, again=True)}".strip()
            again = Reply(
                kind="question",
                text=text,
                quick_replies=offered,
                multi_select=True,
            )
            return {
                "case": case,
                "consult": picture,
                "consults": consults,
                "interview": [*transcript, {"role": "assistant", "text": text}],
                "asked": [*asked, WARNING_CHECK],
                "signs_reasked": True,
                "reply": again.model_dump(),
            }
        # The checklist answer is recorded by rule, not by the model, which can misread a "Yes".
        case = (
            record_signs(known, state["message"], offered)
            if checklist
            else merge(known, turn.updates)
        )
        progress: State = {"case": case, "consult": picture, "consults": consults}
        if state.get("asked_final"):
            if not _nothing_more(state["message"]):
                case["other_notes"] = state["message"]
            return {**progress, "interview": transcript, "phase": "assessing"}
        if checklist and "warning_signs" in case:
            # A warning sign: no more questions, the clinician needs this today.
            step("You have a warning sign: preparing an urgent report now")
            return {
                **progress,
                "interview": transcript,
                "signs_present": case["warning_signs"],
                "phase": "assessing",
            }

        done = (
            turn.complete
            or not turn.next_question.strip()
            or questions >= MAX_QUESTIONS
            or is_repeat(turn.next_question, asked)  # a third try at the same question
        )
        signs = [str(s) for s in (picture or {}).get("warning_signs", []) if str(s).strip()]
        pressing = (picture or {}).get("urgency") in ("urgent", "emergency")
        shown_signs = quick_replies(signs, True)  # a checklist is never asked without options
        check_signs = (
            bool(shown_signs)
            and not state.get("signs_checked")
            and questions < MAX_QUESTIONS
            and (questions >= 1 or done or pressing)  # after the first answer, never skipped
        )
        if check_signs:
            step("Checking the warning signs the Diagnosis agent flagged")
            question, replies, asked_final, multi = WARNING_CHECK, shown_signs, False, True
        elif done:  # always end with the open question
            step("I have what the clinician needs; one last open question")
            question, replies, asked_final = FINAL_QUESTION, ["No, that's everything"], True
            multi = False
        else:
            question, replies, asked_final = turn.next_question.strip(), turn.quick_replies, False
            multi = turn.multi_select and len(replies) > 1
            step(f"Asking next: {turn.reason.strip() or 'to fill in the picture'}")
        acknowledgement = natural(turn.acknowledgement)
        # The checklist is worded for whoever it is about ("has your child had any of these?").
        shown = checklist_question(case) if question == WARNING_CHECK else question
        text = f"{acknowledgement} {shown}" if acknowledgement else shown
        reply = Reply(
            kind="question",
            text=text,
            quick_replies=quick_replies(replies, multi),
            multi_select=multi,
        )
        return {
            **progress,
            "interview": [*transcript, {"role": "assistant", "text": text}],
            "asked": [*asked, question],
            "questions": questions + 1,
            "asked_final": asked_final,
            "signs_checked": state.get("signs_checked", False) or check_signs,
            # Only what the patient saw: options too long to show were not offered.
            "signs_offered": shown_signs if check_signs else offered,
            "reply": reply.model_dump(),
        }

    def after_interview(state: State) -> str:
        return "assess" if state.get("phase") == "assessing" else "remember"

    async def assess(state: State) -> State:
        step("Preparing your draft report: asking the Diagnosis agent for a full assessment")
        picture = state.get("consult")
        final = await _ask_diagnosis(diagnosis, case_summary(state["case"], lead_names(picture)))
        if not (final and final.state == "completed" and final.data):
            text = (final.text if final and final.text else UNAVAILABLE) + f"\n\n_{DISCLAIMER}_"
            return {"phase": "intake", "reply": Reply(kind="no_answer", text=text).model_dump()}
        data = final.data[0]
        signs = state.get("signs_present", "")
        if signs:  # never less than urgent once the patient has a warning sign
            data = {**data, "urgency": at_least(str(data.get("urgency", "routine")), "urgent")}
        step("Writing the handoff note and urgency score for your clinician")
        assessment = (
            f"{data.get('answer', '')}\nPossible conditions: "
            f"{', '.join(data.get('possible_conditions', [])) or 'none named'}\n"
            f"Urgency from the evidence: {data.get('urgency', 'routine')}"
        )
        handoff = await interviewer.handoff(state["case"], briefing(picture), assessment)
        report = build_report(state["case"], data, {**handoff.model_dump(), "warning_signs": signs})
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
            match action.get("type"):
                case "book":
                    return await _send_code(state, action, kind)
                case "resend":
                    return await _resend_code(state)
                case "confirm":
                    return await _confirm(state, offer, str(action.get("code", "")).strip())
                case _:
                    step(f"Looking for open times with {kind}", SCHEDULING_AGENT)
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

    def _refuse(text: str) -> State:
        return {"reply": Reply(kind="no_answer", text=text).model_dump()}

    async def _email_code(booking: Pending, code: str) -> State:
        try:
            await mailer.send_code(booking["email"], code, CODE_MINUTES)
        except MailError as error:
            return _refuse(str(error))
        text = (
            f"We sent a 6-digit code to **{booking['email']}**. Enter it here to confirm your "
            f"visit. It expires in {CODE_MINUTES} minutes."
        )
        reply = Reply(kind="verify_email", text=text, verification=public_view(booking))
        return {"pending": booking, "reply": reply.model_dump()}

    async def _send_code(state: State, action: dict[str, Any], kind: str) -> State:
        name = str(action.get("name", "")).strip()[:80]
        email = str(action.get("email", "")).strip()
        start = str(action.get("start", "")).strip()
        if not name or not EMAIL.match(email) or len(email) > 254:  # noqa: PLR2004
            return _refuse("Please enter your name and a valid email so we can send your code.")
        refusal = await send_limit(limiter, state.get("client", "unknown"), email)
        if refusal:
            return _refuse(refusal)
        code = new_code()
        booking = pending_booking(code, start=start, clinician_type=kind, name=name, email=email)
        return await _email_code(booking, code)

    async def _resend_code(state: State) -> State:
        pending = state.get("pending")
        if not pending:
            return _refuse("Please choose a time first.")
        refusal = await send_limit(limiter, state.get("client", "unknown"), pending["email"])
        if refusal:
            return _refuse(refusal)
        code = new_code()
        booking = pending_booking(
            code,
            start=pending["start"],
            clinician_type=pending["clinician_type"],
            name=pending["name"],
            email=pending["email"],
        )
        return await _email_code(booking, code)

    async def _confirm(state: State, offer: dict[str, Any], code: str) -> State:
        pending = state.get("pending")
        if not pending:
            return _refuse("Please choose a time first.")
        if is_expired(pending) or pending["attempts"] >= MAX_ATTEMPTS:
            text = "That code has expired. Press “Send a new code” to get another one."
            return _verify_again(pending, text)
        if not code_matches(pending, code):
            attempts = pending["attempts"] + 1
            left = MAX_ATTEMPTS - attempts
            text = (
                f"That code is not right. You have {left} {'try' if left == 1 else 'tries'} left."
                if left
                else "That code is not right, and no tries are left. Send a new code."
            )
            return _verify_again({**pending, "attempts": attempts}, text)
        refusal = await booking_limit(limiter, limits, state.get("client", "unknown"))
        if refusal:
            return {"pending": None, **_refuse(refusal)}
        booked = await _book(
            state,
            offer,
            kind=pending["clinician_type"],
            start=pending["start"],
            name=pending["name"],
            email=pending["email"],
        )
        return {"pending": None, **booked}

    def _verify_again(pending: Pending, text: str) -> State:
        reply = Reply(kind="verify_email", text=text, verification=public_view(pending))
        return {"pending": pending, "reply": reply.model_dump()}

    async def _book(
        state: State, offer: dict[str, Any], *, kind: str, start: str, name: str, email: str
    ) -> State:
        report = {**(state.get("report") or {}), "clinician_type": kind}
        step("Booking your visit and sending your report to the clinician", SCHEDULING_AGENT)
        booked = await scheduling.book(
            clinician_type=kind,
            visit_minutes=int(offer["visit_minutes"]),
            start=start,
            session_id=state["session"],
            urgency=offer["urgency"] if offer["urgency"] != "emergency" else "urgent",
            report=report,
            patient_name=name,
            patient_email=email.strip(),
        )
        text = (
            f"**Booked.** Your {int(booked['minutes'])}-minute visit with {booked['physician']} "
            f"({booked['clinician_type']}) is confirmed. Your reference is "
            f"**{booked['reference']}**. An invitation has been sent to {email.strip()}, and your "
            "draft report has been sent to the physician."
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
        ("support", support),
        ("follow_up", follow_up),
        ("answer", answer),
        ("canned", canned),
        ("remember", remember),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "screen")
    graph.add_conditional_edges(
        "screen", dispatch, ["remember", "act", "support", "interview", "route"]
    )
    graph.add_conditional_edges(
        "route", after_route, ["follow_up", "interview", "remember", "answer", "canned"]
    )
    graph.add_conditional_edges("interview", after_interview, ["assess", "remember"])
    for name in ("assess", "act", "support", "follow_up", "answer", "canned"):
        graph.add_edge(name, "remember")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer)


def record_signs(case: dict[str, str], answer: str, offered: list[str]) -> dict[str, str]:
    """The checklist answer, in the case: the signs reported, or that none of them were. An
    unclear answer is recorded as possibly any of them, to be safe."""
    listed = ", ".join(offered) or "the warning signs asked about"
    reading, named = classify_checklist(answer, offered)
    if reading == "none":
        previous = case.get("negatives")
        none = f"none of: {listed}"
        return {**case, "negatives": f"{previous}; {none}" if previous else none}
    reported = ", ".join(named) if named else f"answered “{answer.strip()[:200]}” to: {listed}"
    return {**case, "warning_signs": reported}


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
        "asked": [],
        "asked_final": False,
        "signs_checked": False,
        "signs_present": "",
        "signs_reasked": False,
        "alert": None,
        "consult": None,
        "consults": 0,
        "report": None,
        "offer": None,
        "pending": None,
    }
