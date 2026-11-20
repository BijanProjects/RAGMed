from coordinator_agent.intake import (
    CaseUpdate,
    case_lines,
    is_repeat,
    is_straightforward_request,
    merge,
    quick_replies,
)
from coordinator_agent.reply import build_report
from coordinator_agent.testing import ANSWER


def test_merge_appends_new_facts_and_skips_non_answers() -> None:
    case = merge({"onset": "a month ago"}, [
        CaseUpdate(field="onset", value="after a cold"),
        CaseUpdate(field="onset", value="a month ago"),
        CaseUpdate(field="duration", value="unknown"),
        CaseUpdate(field="allergies", value="  "),
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
