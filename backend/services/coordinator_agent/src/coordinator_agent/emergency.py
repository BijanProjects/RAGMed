"""Red-flag screening with plain rules, before any model is called.

Rules, not a model, so that an outage, a rate limit or a model mistake can never hide an
emergency. They err on the side of caution: a false alarm costs the person one extra message,
a missed emergency could cost much more.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RedFlag:
    category: str
    guidance: str


EMERGENCY_GUIDANCE = (
    "**This may be a medical emergency. Call your local emergency number now** (911 in the US "
    "and Canada, 112 in Europe, 999 in the UK), or go to the nearest emergency department. "
    "Do not wait for an online answer."
)
CRISIS_GUIDANCE = (
    "**You do not have to go through this alone.** If you are thinking about ending your life "
    "or harming yourself, please call or text **988** (Suicide and Crisis Lifeline, US and "
    "Canada) or your local emergency number now. If you are in immediate danger, call "
    "emergency services."
)

_RULES: list[tuple[str, str, list[str]]] = [
    ("self_harm", CRISIS_GUIDANCE, [
        r"\b(kill|hurt|harm)(ing)? (my ?self|myself)\b", r"\bsuicid", r"\bend (my|it) (life|all)\b",
        r"\b(don'?t|do not) want to (live|be alive)\b", r"\bself[- ]harm",
        r"\bbetter off dead\b", r"\bwant to die\b",
    ]),
    ("chest_pain", EMERGENCY_GUIDANCE, [
        r"\bchest (pain|pressure|tightness|hurts?)\b", r"\bpain in (my|the) chest\b",
        r"\bcrushing\b.*\bchest\b", r"\bheart attack\b",
    ]),
    ("stroke", EMERGENCY_GUIDANCE, [
        r"\b(face|mouth) (is )?(drooping|droops?)\b", r"\bslurr(ed|ing) (speech|words)\b",
        r"\b(can'?t|cannot|unable to) (speak|talk|move (my|one) (arm|leg|side))\b",
        r"\b(numb|weak)(ness)? (on|in) (one|the (left|right)) side\b",
        # Someone having a stroke now; "my mother had a stroke" is family history, not a flag.
        r"\b(having|has|is having|'s having) a stroke\b", r"\bstroke symptoms\b",
    ]),
    ("breathing", EMERGENCY_GUIDANCE, [
        r"\b(can'?t|cannot|unable to|struggling to|hard to) breathe?\b", r"\bnot breathing\b",
        r"\b(choking|turning blue|blue lips)\b", r"\bgasping for (air|breath)\b",
        r"\bsevere(ly)? short(ness)? of breath\b",
    ]),
    ("anaphylaxis", EMERGENCY_GUIDANCE, [
        r"\b(throat|tongue|lips?) (is |are )?(swelling|swollen|closing)\b", r"\banaphyla",
        r"\bsevere allergic reaction\b", r"\b(can'?t|cannot) swallow\b",
    ]),
    ("bleeding", EMERGENCY_GUIDANCE, [
        r"\b(heavy|severe|uncontrolled|won'?t stop) bleeding\b", r"\bbleeding (heavily|a lot)\b",
        r"\bbleeding (won'?t|will not|doesn'?t|does not|can'?t|cannot) (stop|be stopped)\b",
        r"\b(coughing|vomiting) (up )?blood\b",
    ]),
    ("consciousness", EMERGENCY_GUIDANCE, [
        r"\b(unconscious|unresponsive|passed out|fainted and)\b", r"\bwon'?t wake up\b",
        r"\b(seizure|convulsion|fitting)\b",
    ]),
    ("poisoning", EMERGENCY_GUIDANCE, [
        r"\boverdos", r"\b(swallowed|drank|ate) (bleach|poison|pills|detergent|battery)\b",
        r"\btook too many (pills|tablets)\b",
    ]),
    ("severe_pain", EMERGENCY_GUIDANCE, [
        r"\bworst headache (of|in) my life\b", r"\bsudden severe headache\b",
        r"\bworst[- ]ever headache\b", r"\bthunderclap\b",
        r"\bstiff neck\b.*\bfever\b|\bfever\b.*\bstiff neck\b",
    ]),
]  # fmt: skip

_COMPILED = [
    (category, guidance, [re.compile(p, re.IGNORECASE) for p in patterns])
    for category, guidance, patterns in _RULES
]


def screen(text: str) -> RedFlag | None:
    """The first red flag found in `text`, or None. Self-harm is checked first."""
    # Phones often type a curly apostrophe (U+2019): "don\u2019t" must match "don't".
    normalised = " ".join(text.replace("\u2019", "'").split())
    for category, guidance, patterns in _COMPILED:
        if any(p.search(normalised) for p in patterns):
            return RedFlag(category, guidance)
    return None
