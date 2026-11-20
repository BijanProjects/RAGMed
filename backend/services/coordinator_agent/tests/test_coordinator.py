from typing import Any

import httpx
import pytest

from coordinator_agent.app import CoordinatorSettings, create_coordinator_app
from coordinator_agent.emergency import CRISIS_GUIDANCE, EMERGENCY_GUIDANCE
from coordinator_agent.executor import AT_CAPACITY
from coordinator_agent.intake import FINAL_QUESTION, MAX_QUESTIONS
from coordinator_agent.reply import BOOKING_START, DISCLAIMER, GREETING, OUT_OF_SCOPE
from coordinator_agent.router import is_clinic_request
from coordinator_agent.testing import (
    ANSWER,
    FakeDiagnosis,
    FakeInterviewer,
    FakeRouter,
    FakeScheduling,
)
from ragmed_a2a.client import Progress, RemoteAgent, Reply
from ragmed_core.clinic import DRAFT_NOTICE
from ragmed_core.settings import Environment
from ragmed_llm import AtCapacityError

pytestmark = pytest.mark.anyio
BASE = "http://coordinator.test"
PATIENT = {"name": "Alex Sample", "email": "alex@example.org"}


class Chat:
    """One patient's conversation with the Coordinator over A2A, in memory."""

    def __init__(
        self,
        router: FakeRouter | None = None,
        interviewer: FakeInterviewer | None = None,
        diagnosis: FakeDiagnosis | None = None,
        scheduling: FakeScheduling | None = None,
    ) -> None:
        self.router = router or FakeRouter()
        self.interviewer = interviewer or FakeInterviewer()
        self.diagnosis = diagnosis or FakeDiagnosis()
        self.scheduling = scheduling or FakeScheduling()
        app = create_coordinator_app(
            CoordinatorSettings(public_url=BASE),
            router=self.router,
            interviewer=self.interviewer,
            diagnosis=self.diagnosis,
            scheduling=self.scheduling,
        )
        self.agent = RemoteAgent(BASE, transport=httpx.ASGITransport(app=app))
        self.context_id: str | None = None
        self.progress: list[str] = []

    async def send(self, text: str = "", action: dict[str, Any] | None = None) -> Reply:
        data = {"action": action} if action else None
        events = [e async for e in self.agent.stream(text, context_id=self.context_id, data=data)]
        reply = events[-1]
        assert isinstance(reply, Reply)
        self.context_id = reply.context_id
        self.progress = [e.text for e in events if isinstance(e, Progress)]
        return reply

    async def interview_to_report(self) -> Reply:
        """Complaint, two answers, then 'nothing more': the default fake interview."""
        await self.send("I keep getting headaches on one side")
        await self.send("About a month")
        final = await self.send("7 out of 10")
        assert final.text == FINAL_QUESTION
        return await self.send("No, that's everything")


def data(reply: Reply) -> dict[str, Any]:
    (payload,) = reply.data
    return payload


# Safety -----------------------------------------------------------------------------------


async def test_a_red_flag_is_escalated_before_any_model_is_called() -> None:
    chat = Chat()

    reply = await chat.send("I have crushing chest pain")

    assert chat.router.seen == []
    assert chat.interviewer.calls == []
    assert reply.text.startswith(EMERGENCY_GUIDANCE)
    assert data(reply)["kind"] == "emergency"
    assert data(reply)["red_flag"] == "chest_pain"


async def test_self_harm_gets_the_crisis_line() -> None:
    reply = await Chat().send("I want to end my life")

    assert reply.text.startswith(CRISIS_GUIDANCE)


async def test_a_red_flag_mid_interview_is_escalated_and_the_interview_resumes() -> None:
    chat = Chat()
    await chat.send("I keep getting headaches")

    alarm = await chat.send("now my face is drooping")
    resumed = await chat.send("About a month")

    assert data(alarm)["kind"] == "emergency"
    assert data(resumed)["kind"] == "question"


# The intake interview --------------------------------------------------------------------


async def test_symptoms_start_an_interview_with_one_question_at_a_time() -> None:
    chat = Chat()

    first = await chat.send("I keep getting headaches on one side")
    second = await chat.send("About a month")

    assert first.text == "How long have you had it?"
    assert data(first)["kind"] == "question"
    assert data(first)["quick_replies"] == ["A day", "A week"]  # over-long suggestions dropped
    assert second.text == "How bad is it, from 0 to 10?"
    assert len(chat.router.seen) == 1  # during the interview, answers are not re-routed
    case, transcript = chat.interviewer.calls[1]
    assert chat.interviewer.calls[0][0] == {"main_problem": "I keep getting headaches on one side"}
    assert case["main_problem"] == "I keep getting headaches on one side"
    assert "patient: About a month" in transcript


async def test_the_interview_always_ends_with_the_open_question() -> None:
    chat = Chat()
    await chat.send("I keep getting headaches on one side")
    await chat.send("About a month")

    final = await chat.send("7 out of 10")

    assert final.text == FINAL_QUESTION
    assert data(final)["quick_replies"] == ["No, that's everything"]


async def test_a_question_asked_twice_is_not_asked_a_third_time() -> None:
    same = "How long does each headache last?"
    chat = Chat(interviewer=FakeInterviewer(questions=[same, same, same, "Never reached?"]))

    replies = [await chat.send("headaches")] + [await chat.send("not sure") for _ in range(2)]

    assert [r.text for r in replies] == [same, same, FINAL_QUESTION]


async def test_the_interview_stops_after_the_question_limit() -> None:
    chat = Chat(interviewer=FakeInterviewer(questions=[f"Q{n}?" for n in range(30)]))
    reply = await chat.send("headaches")
    for _ in range(MAX_QUESTIONS):
        reply = await chat.send("an answer")

    assert reply.text == FINAL_QUESTION


async def test_a_complete_interview_produces_a_draft_report_and_an_offer() -> None:
    chat = Chat()

    report = await chat.interview_to_report()

    payload = data(report)
    assert payload["kind"] == "report"
    assert DRAFT_NOTICE in report.text
    assert "Main problem: I keep getting headaches on one side" in report.text
    assert "migraine" in report.text
    assert "30 minutes with Neurology" in report.text
    assert payload["offer"] == {
        "clinician_type": "Neurology",
        "visit_minutes": 30,
        "urgency": "routine",
        "alternative": "General physician",
    }
    assert "Other notes" not in payload["report"]["case"]  # "no, that's everything" is not a note
    assert "Main problem: I keep getting headaches on one side" in chat.diagnosis.questions[0]
    assert "Preparing your draft report" in chat.progress


async def test_a_real_answer_to_the_open_question_goes_into_the_report() -> None:
    chat = Chat()
    await chat.send("I keep getting headaches on one side")
    await chat.send("About a month")
    await chat.send("7 out of 10")

    report = await chat.send("I'm worried because my mother had a stroke")

    assert data(report)["report"]["case"]["Other notes"] == (
        "I'm worried because my mother had a stroke"
    )


async def test_an_emergency_assessment_offers_no_booking() -> None:
    chat = Chat(diagnosis=FakeDiagnosis(result={**ANSWER, "urgency": "emergency"}))

    report = await chat.interview_to_report()

    assert report.text.startswith(EMERGENCY_GUIDANCE)
    assert data(report)["offer"] is None


async def test_a_failed_assessment_can_be_retried() -> None:
    diagnosis = FakeDiagnosis(state="failed", text="RAGMed is at capacity right now.")
    chat = Chat(diagnosis=diagnosis)

    failed = await chat.interview_to_report()
    diagnosis.state = "completed"
    retried = await chat.send("please try again")

    assert data(failed)["kind"] == "no_answer"
    assert failed.text.startswith("RAGMed is at capacity right now.")
    assert data(retried)["kind"] == "report"


async def test_a_booking_request_asks_what_the_visit_is_for_first() -> None:
    chat = Chat(router=FakeRouter(intent="book_appointment"))

    first = await chat.send("I need an appointment")
    await chat.send("A rash on my arm")
    await chat.send("Since Monday")

    assert first.text == BOOKING_START
    case, transcript = chat.interviewer.calls[0]
    assert case == {"main_problem": "A rash on my arm"}  # the patient's words, not our question
    assert transcript.startswith(f"assistant: {BOOKING_START}")


# Booking ---------------------------------------------------------------------------------


async def test_show_times_then_book_sends_the_report_to_the_physician() -> None:
    chat = Chat()
    await chat.interview_to_report()

    slots = await chat.send(action={"type": "show_slots"})
    start = data(slots)["slots"][1]["start"]
    booked = await chat.send(action={"type": "book", "start": start, **PATIENT})

    assert data(slots)["kind"] == "slots"
    assert chat.scheduling.searches == [("Neurology", 30)]
    assert data(booked)["kind"] == "booked"
    assert "RC-TEST42" in booked.text
    assert "invitation has been sent to alex@example.org" in booked.text
    request = chat.scheduling.bookings[0]
    assert (request["patient_name"], request["patient_email"]) == (
        "Alex Sample",
        "alex@example.org",
    )
    assert request["start"] == start
    assert request["session_id"] == chat.context_id
    assert request["urgency"] == "routine"
    assert request["report"]["possible_conditions"] == ["migraine"]


async def test_the_patient_can_choose_a_general_physician_instead() -> None:
    chat = Chat()
    await chat.interview_to_report()

    slots = await chat.send(action={"type": "show_slots", "clinician_type": "General physician"})
    await chat.send(
        action={
            "type": "book",
            "start": SLOT_START,
            "clinician_type": "General physician",
            **PATIENT,
        }
    )

    assert chat.scheduling.searches == [("General physician", 30)]
    assert data(slots)["offer"]["clinician_type"] == "General physician"
    assert chat.scheduling.bookings[0]["clinician_type"] == "General physician"


SLOT_START = "2026-11-23T09:00:00-03:30"


async def test_only_the_recommended_clinician_or_a_gp_can_be_chosen() -> None:
    chat = Chat()
    await chat.interview_to_report()

    await chat.send(action={"type": "show_slots", "clinician_type": "Cardiology"})

    assert chat.scheduling.searches == [("Neurology", 30)]


async def test_scheduling_problems_are_shown_plainly() -> None:
    chat = Chat(scheduling=FakeScheduling(error="That time is no longer available."))
    await chat.interview_to_report()

    reply = await chat.send(action={"type": "book", "start": SLOT_START, **PATIENT})

    assert reply.text == "That time is no longer available."


async def test_booking_needs_the_patients_name_and_email() -> None:
    chat = Chat()
    await chat.interview_to_report()

    reply = await chat.send(action={"type": "book", "start": SLOT_START, "name": "Alex"})

    assert chat.scheduling.bookings == []
    assert "name and email" in reply.text


async def test_booking_without_a_report_is_refused() -> None:
    reply = await Chat().send(action={"type": "show_slots"})

    assert data(reply)["kind"] == "no_answer"
    assert "no report" in reply.text


# Other conversations ---------------------------------------------------------------------


async def test_a_general_question_is_answered_with_sources() -> None:
    chat = Chat(router=FakeRouter(intent="medical_question"))

    reply = await chat.send("what causes migraines")

    assert chat.diagnosis.questions == ["What causes migraines?"]
    assert data(reply)["kind"] == "answer"
    assert "1. What are migraines?" in reply.text
    assert "see Neurology soon" in reply.text
    assert DISCLAIMER in reply.text


@pytest.mark.parametrize(
    ("intent", "text", "kind"),
    [("greeting", GREETING, "greeting"), ("out_of_scope", OUT_OF_SCOPE, "out_of_scope")],
)
async def test_greetings_and_off_topic_get_a_direct_reply(
    intent: Any, text: str, kind: str
) -> None:
    chat = Chat(router=FakeRouter(intent=intent))

    reply = await chat.send("hello there")

    assert chat.interviewer.calls == []
    assert reply.text == text
    assert data(reply)["kind"] == kind


@pytest.mark.parametrize(
    "message",
    ["I want to renew my blood pressure prescription", "Can I get a sick note for work?"],
)
async def test_renewals_and_notes_are_never_turned_away(message: str) -> None:
    chat = Chat(router=FakeRouter(intent="out_of_scope"))

    reply = await chat.send(message)

    assert data(reply)["kind"] != "out_of_scope"
    assert chat.interviewer.calls


def test_ordinary_words_are_not_mistaken_for_a_clinic_request() -> None:
    assert not is_clinic_request("what's the weather forecast? fill in this information")


async def test_separate_conversations_do_not_share_anything() -> None:
    first = Chat()
    await first.send("I keep getting headaches")
    second = Chat(router=first.router, interviewer=FakeInterviewer())

    await second.send("my knee hurts")

    assert second.interviewer.calls[0][0] == {"main_problem": "my knee hurts"}
    assert first.router.seen[1][1] == ""


async def test_at_capacity_gets_a_clear_message() -> None:
    reply = await Chat(router=FakeRouter(error=AtCapacityError("429"))).send("headache")

    assert reply.state == "failed"
    assert reply.text == AT_CAPACITY


async def test_an_unexpected_error_fails_politely() -> None:
    reply = await Chat(router=FakeRouter(error=RuntimeError("secret internals"))).send("x")

    assert reply.state == "failed"
    assert "secret internals" not in reply.text


@pytest.mark.parametrize("text", ["   ", "x" * 2001])
async def test_empty_or_huge_messages_are_rejected(text: str) -> None:
    assert (await Chat().send(text)).state == "rejected"


async def test_unknown_actions_are_ignored() -> None:
    reply = await Chat().send("", action={"type": "delete_everything"})

    assert reply.state == "rejected"


def test_production_requires_service_tokens() -> None:
    settings = CoordinatorSettings(environment=Environment.PRODUCTION)

    with pytest.raises(RuntimeError, match="RAGMED_SERVICE_TOKENS"):
        create_coordinator_app(
            settings,
            router=FakeRouter(),
            interviewer=FakeInterviewer(),
            diagnosis=FakeDiagnosis(),
            scheduling=FakeScheduling(),
        )
