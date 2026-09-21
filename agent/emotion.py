"""Emotion analysis agent node for Mindly.

Estimates high-level conversational emotional states (non-clinical).
Categories: calm, happy, sad, anxious, stressed, angry, overwhelmed, confused, neutral.
"""

from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage


EMOTION_CATEGORIES = [
    "calm",
    "happy",
    "sad",
    "anxious",
    "stressed",
    "angry",
    "overwhelmed",
    "confused",
    "neutral",
]


def classify_emotion_heuristic(user_text: str) -> str:
    """Heuristic rule-based emotion estimation as a reliable baseline or fallback."""
    lower = user_text.lower().strip()

    if any(k in lower for k in ["overwhelm", "drowning", "too much", "can't keep up", "falling apart", "pile on"]):
        return "overwhelmed"

    if any(k in lower for k in ["panic", "anxious", "anxiety", "scared", "terrified", "nervous", "racing", "worry", "worried"]):
        return "anxious"

    if any(k in lower for k in ["stress", "stressed", "pressure", "workload", "deadline", "burnout", "exhausted"]):
        return "stressed"

    if any(k in lower for k in ["sad", "depressed", "lonely", "crying", "heartbroken", "down", "gloomy", "hopeless"]):
        return "sad"

    if any(k in lower for k in ["angry", "mad", "furious", "pissed", "annoyed", "frustrated", "irritated"]):
        return "angry"

    if any(k in lower for k in ["confused", "lost", "don't know what to do", "stuck", "uncertain", "doubt"]):
        return "confused"

    if any(k in lower for k in ["happy", "great", "wonderful", "grateful", "excited", "good day", "glad"]):
        return "happy"

    if any(k in lower for k in ["peaceful", "calm", "relaxed", "serene", "fine", "content"]):
        return "calm"

    return "neutral"


EMOTION_PROMPT = """You are an emotion estimation component for Mindly, a mental wellness support assistant.
Analyze the user's message and identify the dominant emotional tone from EXACTLY ONE of these categories:
- calm
- happy
- sad
- anxious
- stressed
- angry
- overwhelmed
- confused
- neutral

Respond with ONLY the lowercase category name. Nothing else.
"""


def classify_emotion(user_text: str, llm=None) -> str:
    """Classify user emotion using LLM if available, falling back to heuristics."""
    if not user_text.strip():
        return "neutral"

    if llm is not None:
        try:
            messages = [
                SystemMessage(content=EMOTION_PROMPT),
                HumanMessage(content=f"User message: \"{user_text}\"\nEmotion:"),
            ]
            response = llm.invoke(messages)
            raw_emotion = str(response.content).strip().lower()
            for emo in EMOTION_CATEGORIES:
                if emo in raw_emotion:
                    return emo
        except Exception:
            pass

    return classify_emotion_heuristic(user_text)
