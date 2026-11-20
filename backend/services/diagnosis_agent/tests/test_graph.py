import pytest

from diagnosis_agent.graph import (
    INSUFFICIENT,
    MAX_SEARCHES,
    build_graph,
    finalise,
    insufficient_result,
    number_passages,
    remove_doses,
    remove_sentences,
)
from diagnosis_agent.schema import DiagnosisAnswer, DiagnosisResult
from diagnosis_agent.testing import PASSAGES, FakeKnowledge, FakeReasoner

pytestmark = pytest.mark.anyio


async def run(knowledge: FakeKnowledge, reasoner: FakeReasoner) -> DiagnosisResult:
    state = await build_graph(knowledge, reasoner).ainvoke({"question": "Why do I wheeze?"})
    result = state["result"]
    assert isinstance(result, DiagnosisResult)
    return result


async def test_good_evidence_on_the_first_search_is_answered() -> None:
    reasoner = FakeReasoner(answerable=[True])

    result = await run(FakeKnowledge([PASSAGES]), reasoner)

    assert reasoner.calls == ["rewrite", "answer", "check"]
    assert result.sufficient_evidence
    assert result.answer == "Inhalers help [2]."
    assert [(c.number, c.answer_id) for c in result.citations] == [(2, "a2")]
    assert result.search_query == "query 1"


async def test_weak_evidence_triggers_one_new_search() -> None:
    knowledge = FakeKnowledge([PASSAGES, PASSAGES])
    reasoner = FakeReasoner(answerable=[False, True])

    result = await run(knowledge, reasoner)

    assert knowledge.queries == ["query 1", "query 2"]
    assert reasoner.previous_queries == [None, "query 1"]  # the retry knows what failed
    assert result.sufficient_evidence


async def test_still_weak_after_the_retry_says_so_instead_of_guessing() -> None:
    reasoner = FakeReasoner(answerable=[False] * MAX_SEARCHES)

    result = await run(FakeKnowledge([PASSAGES]), reasoner)

    assert reasoner.calls.count("answer") == MAX_SEARCHES
    assert "check" not in reasoner.calls
    assert result.answer == INSUFFICIENT
    assert not result.sufficient_evidence
    assert result.recommend_visit
    assert result.citations == []


async def test_no_results_skip_the_model() -> None:
    reasoner = FakeReasoner(answerable=[])

    result = await run(FakeKnowledge([[]]), reasoner)

    assert "answer" not in reasoner.calls
    assert result.answer == INSUFFICIENT


def draft(**changes: object) -> DiagnosisAnswer:
    base = DiagnosisAnswer(
        answerable=True,
        answer="Inhalers help [2].",
        citations=[2],
        possible_conditions=[],
        urgency="routine",
        recommend_visit=False,
        clinician_type="General physician",
        visit_minutes=15,
        visit_reason="r",
    )
    return base.model_copy(update=changes)


def test_citations_outside_the_passages_are_dropped() -> None:
    result = finalise(draft(citations=[0, 2, 2, 9]), PASSAGES, "q")

    assert [c.number for c in result.citations] == [2]


def test_urgent_answers_always_recommend_a_visit() -> None:
    assert finalise(draft(urgency="urgent"), PASSAGES, "q").recommend_visit
    assert finalise(draft(urgency="emergency"), PASSAGES, "q").recommend_visit


@pytest.mark.parametrize(
    "dose", ["Take 500 mg twice a day.", "Use 2.5ml at night.", "Inject 10 units daily."]
)
def test_sentences_with_doses_are_removed(dose: str) -> None:
    text = f"Asthma is common [1]. {dose} See a doctor."

    assert remove_doses(text) == "Asthma is common [1]. See a doctor."


def test_an_answer_that_was_only_a_dose_becomes_the_insufficient_message() -> None:
    assert finalise(draft(answer="Take 400 mg."), PASSAGES, "q").answer == INSUFFICIENT


def test_passages_are_numbered_from_one() -> None:
    text = number_passages(PASSAGES)

    assert text.startswith("[1] (causes) What causes asthma?")
    assert "\n\n[2] (treatment)" in text


async def test_the_check_removes_only_flagged_sentences_and_keeps_triage() -> None:
    answer = DiagnosisAnswer(
        answerable=True,
        answer="Inhalers help [2]. Asthma is caused by aliens. Please see a doctor.",
        citations=[2],
        possible_conditions=["asthma"],
        urgency="urgent",
        recommend_visit=True,
        clinician_type="Pulmonology",
        visit_minutes=30,
        visit_reason="r",
    )
    reasoner = FakeReasoner(
        answerable=[True], answer_given=answer, unsupported=["asthma is caused by ALIENS"]
    )

    result = await run(FakeKnowledge([PASSAGES]), reasoner)

    assert result.answer == "Inhalers help [2]. Please see a doctor."
    assert result.urgency == "urgent"
    assert result.recommend_visit


async def test_a_check_that_removes_everything_still_advises_a_clinician() -> None:
    answer = DiagnosisAnswer(
        answerable=True,
        answer="Something unsupported.",
        citations=[1],
        possible_conditions=["x"],
        urgency="self_care",
        recommend_visit=False,
        clinician_type="General physician",
        visit_minutes=15,
        visit_reason="r",
    )
    reasoner = FakeReasoner(
        answerable=[True], answer_given=answer, unsupported=["Something unsupported."]
    )

    result = await run(FakeKnowledge([PASSAGES]), reasoner)

    assert result.answer == INSUFFICIENT
    assert result.recommend_visit
    assert result.urgency == "routine"  # raised from self_care: no answer means see someone
    assert not result.sufficient_evidence


def test_insufficient_never_lowers_an_urgency_already_found() -> None:
    assert insufficient_result("q", "emergency").urgency == "emergency"
    assert insufficient_result("q", "self_care").urgency == "routine"


def test_remove_sentences_ignores_empty_flags() -> None:
    assert remove_sentences("One. Two.", ["", "   "]) == "One. Two."


async def test_unanswerable_keeps_an_urgency_the_model_already_saw() -> None:
    urgent = DiagnosisAnswer(
        answerable=False,
        answer="",
        citations=[],
        possible_conditions=[],
        urgency="emergency",
        recommend_visit=True,
        clinician_type="General physician",
        visit_minutes=15,
        visit_reason="r",
    )
    reasoner = FakeReasoner(answerable=[False] * MAX_SEARCHES, answer_given=urgent)

    result = await run(FakeKnowledge([PASSAGES]), reasoner)

    assert result.answer == INSUFFICIENT
    assert result.urgency == "emergency"


def test_only_cited_passages_can_be_selected() -> None:
    text = number_passages(PASSAGES, only={2})

    assert text.startswith("[2] (treatment)")
    assert "[1]" not in text


def test_without_evidence_a_general_physician_sees_them_for_longer() -> None:
    result = insufficient_result("q")

    assert result.clinician_type == "General physician"
    assert result.visit_minutes == 30


def test_triage_fields_pass_through() -> None:
    result = finalise(draft(clinician_type="Neurology", visit_minutes=30), PASSAGES, "q")

    assert (result.clinician_type, result.visit_minutes) == ("Neurology", 30)
