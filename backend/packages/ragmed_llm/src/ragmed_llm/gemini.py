"""Gemini chat models: the backup when every Groq model is busy.

Groq's free tier allows a few thousand tokens a minute per model, so a burst of visitors can use
it up. Google AI Studio's free tier has separate limits, so listing Gemini models after the Groq
ones lets the conversation carry on instead of asking the patient to wait.

Optional: without GOOGLE_API_KEY there is simply no backup. Note that on the free tier Google may
use requests to improve its products, which is one reason the demo asks visitors not to enter
real personal health information.
"""

from collections.abc import Sequence

from langchain_google_genai import ChatGoogleGenerativeAI

from ragmed_core.settings import ProviderKeys


def gemini_models(
    names: Sequence[str], thinking_level: str = "minimal"
) -> list[ChatGoogleGenerativeAI]:
    """One model per name, in fallback order; empty when GOOGLE_API_KEY is not set.

    Measured in November 2026: gemini-3.1-flash-lite answers in about a second and
    gemini-3.5-flash-lite in about 7; gemini-3.5-flash took 12-42 seconds, too slow for a patient
    waiting in a chat. List the fastest first.
    `max_retries=0` hands over at once to the next model when one is busy.
    """
    key = ProviderKeys().google_api_key
    if key is None:
        return []
    return [
        ChatGoogleGenerativeAI(
            model=name,
            api_key=key,
            max_retries=0,  # Gemini 3 models use fixed sampling, so no temperature
            timeout=60,
            thinking_level=thinking_level,
        )
        for name in names
    ]
