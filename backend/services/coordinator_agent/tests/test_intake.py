from coordinator_agent.intake import (
    CaseUpdate,
    briefing,
    case_lines,
    is_repeat,
    is_straightforward_request,
    merge,
    natural,
    quick_replies,
)
from coordinator_agent.reply import build_report, triage
from coordinator_agent.testing import ANSWER, CONSULT


def test_merge_appends_new_facts_and_skips_non_answers() -> None:
    case = merge({"onset": "a month ago"}, [
        CaseUpdate(field="onset", value="after a cold"),
        CaseUpdate(field="onset", value="a month ago"),
        CaseUpdate(field="duration", value="unknown"),
        CaseUpdate(field="allergies", value="  "),
        CaseUpdate(field="pattern", value="?"),
        CaseUpdate(field="age", value="34"),
    ])  # fmt: skip

    assert case == {"onset": "a month ago; after a cold", "age": "34"}


def test_case_lines_use_readable_labels() -> None:
    assert case_lines({"severity": "7", "age": "34"}) == "- Severity (0-10): 7\n- Age: 34"


def test_similar_questions_count_as_repeats() -> None:
    asked = [
        "How long does each headache episode usually last?",
        "How long does each headache episode last, from start to finish?",
    ]

    assert is_repeat("How long does each headache episode last?", asked)
    assert not is_repeat("How long does each headache episode last?", asked[:1])
    assert not is_repeat("Do you take any medications?", asked)
    # Rewordings seen live count too.
    assert is_repeat("How long does each individual headache episode last, in hours?", asked)
    assert not is_repeat("When did the headaches first start?", asked)


def test_quick_replies_are_short_and_few() -> None:
    assert quick_replies(["Yes", "", "x" * 41, "No", "A", "B", "C"]) == ["Yes", "No", "A", "B"]


def test_the_first_main_problem_stands_and_unasked_answers_are_dropped() -> None:
    case = merge({"main_problem": "headaches"}, [
        CaseUpdate(field="main_problem", value="head pain"),
        CaseUpdate(field="allergies", value="none mentioned"),
    ])  # fmt: skip

    assert case == {"main_problem": "headaches"}


def test_renewals_and_notes_are_straightforward_requests() -> None:
    assert is_straightforward_request({"main_problem": "I need a renewal of my inhaler"})
    assert is_straightforward_request({"main_problem": "a sick note for work"})
    assert not is_straightforward_request({"main_problem": "headaches for a month"})


def test_a_renewal_is_always_a_short_gp_visit_unless_urgent() -> None:
    case = {"main_problem": "renew my blood pressure prescription"}

    routine = build_report(case, {**ANSWER, "clinician_type": "Cardiology", "visit_minutes": 30})
    urgent = build_report(case, {**ANSWER, "urgency": "urgent", "visit_minutes": 30})

    assert (routine["clinician_type"], routine["visit_minutes"]) == ("General physician", 15)
    assert urgent["visit_minutes"] == 30


def test_a_disclaimer_copied_into_the_assessment_is_removed() -> None:
    answer = (
        "Measles fits [1]. RAGMed is an educational project, not a medical device. "
        "In an emergency, call your local emergency number."
    )

    report = build_report({"main_problem": "rash"}, {**ANSWER, "answer": answer})

    assert report["assessment"] == "Measles fits [1]."


def test_the_coordinator_may_raise_the_urgency_but_never_lower_it() -> None:
    assert triage("routine", None) == (4, "routine")
    assert triage("routine", 3) == (3, "routine")
    assert triage("routine", 2) == (2, "urgent")
    assert triage("urgent", 5) == (2, "urgent")  # the evidence says today: it stays today
    assert triage("emergency", 4) == (1, "emergency")
    assert triage("self_care", 3) == (3, "routine")
    assert triage("routine", 0) == (4, "routine")  # a malformed score is ignored


def test_the_briefing_reads_the_working_picture() -> None:
    text = briefing(CONSULT)

    assert "Considering: migraine (one-sided, recurring)" in text
    assert "Working urgency: routine" in text
    assert "change little: no" in text
    assert "none yet" in briefing(None)
    assert "nothing specific yet" in briefing({**CONSULT, "leads": []})


def test_stock_phrases_are_dropped_from_acknowledgements() -> None:
    assert natural("Sounds tough, thank you for sharing.") == "Sounds tough."
    assert natural("Thanks for sharing that!") == ""
    assert natural("  A month of that sounds tiring. ") == "A month of that sounds tiring."
