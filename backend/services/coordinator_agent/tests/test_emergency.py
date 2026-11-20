import pytest

from coordinator_agent.emergency import CRISIS_GUIDANCE, EMERGENCY_GUIDANCE, screen

# Curated cases: every one must be escalated (the plan requires 100% recall on these).
EMERGENCIES = [
    ("I have crushing pain in my chest and my left arm hurts", "chest_pain"),
    ("chest tightness for 20 minutes and sweating", "chest_pain"),
    ("I think I'm having a heart attack", "chest_pain"),
    ("my dad's face is drooping and his speech is slurred", "stroke"),
    ("sudden numbness on one side of my body", "stroke"),
    ("I can't breathe properly", "breathing"),
    ("my baby has blue lips", "breathing"),
    ("my throat is swelling after eating peanuts", "anaphylaxis"),
    ("I think I'm having a severe allergic reaction", "anaphylaxis"),
    ("the bleeding won't stop after I cut my hand", "bleeding"),
    ("I'm coughing up blood", "bleeding"),
    ("the bleeding does not stop", "bleeding"),
    ("my mum is unconscious", "consciousness"),
    ("my son had a seizure", "consciousness"),
    ("my toddler swallowed bleach", "poisoning"),
    ("I took too many pills", "poisoning"),
    ("worst headache of my life, came on suddenly", "severe_pain"),
    ("I have a fever and a stiff neck", "severe_pain"),
    ("I want to kill myself", "self_harm"),
    ("I don\u2019t want to live anymore", "self_harm"),  # curly apostrophe, as phones type it
    ("I've been thinking about suicide", "self_harm"),
    ("I keep self-harming", "self_harm"),
]

ROUTINE = [
    "What are the symptoms of asthma?",
    "How is type 2 diabetes treated?",
    "I have had a runny nose for two days",
    "Can I book an appointment for next week?",
    "My knee hurts when I run",
    "What causes migraines?",
    "hello",
]


@pytest.mark.parametrize(("text", "category"), EMERGENCIES)
def test_every_curated_emergency_is_escalated(text: str, category: str) -> None:
    flag = screen(text)

    assert flag is not None, text
    assert flag.category == category


@pytest.mark.parametrize("text", ROUTINE)
def test_routine_questions_pass_through(text: str) -> None:
    assert screen(text) is None


def test_self_harm_gets_crisis_guidance_even_with_other_flags() -> None:
    flag = screen("I want to kill myself, I took too many pills")

    assert flag is not None
    assert flag.guidance == CRISIS_GUIDANCE
    assert "988" in flag.guidance


def test_physical_emergencies_point_to_emergency_services() -> None:
    flag = screen("chest pain")

    assert flag is not None
    assert flag.guidance == EMERGENCY_GUIDANCE
    assert "911" in flag.guidance
