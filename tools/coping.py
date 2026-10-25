import re
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import settings


COPING_SYSTEM_PROMPT = """You are a compassionate wellness coping specialist for Mindly.
Your goal is to provide practical, grounded coping strategies for users facing academic stress, burnout, time pressure, or emotional fatigue.

Strict Guidelines:
1. Provide 2-3 structured, actionable, gentle micro-steps (e.g., 20-minute focus blocks, boundary setting, progressive muscle relaxation).
2. Validate the user's feelings warmly.
3. Strict boundaries: NEVER diagnose illnesses, NEVER prescribe medications, and NEVER claim to replace licensed therapists.
4. Keep the output concise, structured with bullet points, and comforting.
"""


def generate_heuristic_coping(user_text: str, emotion: str = "stressed") -> str:
    """Dynamic coping fallback synthesized from user context and emotional state."""
    # Strip LLM prompt wrappers if present
    match = re.search(r'Their situation is:\s*["\']([^"\']+)["\']', user_text, re.IGNORECASE)
    raw = match.group(1).strip() if match else user_text.strip()

    STOPWORDS = {
        "what", "this", "that", "with", "have", "from", "they", "will", "would", "about",
        "there", "their", "could", "should", "your", "mindly", "feel", "feeling", "anxious",
        "stressed", "user", "situation", "generate", "practical", "compassionate", "actionable",
        "micro", "steps", "tailored", "directly", "tell", "help", "need", "give", "some",
        "can", "very", "much", "just", "really", "also", "want", "like"
    }
    clean_words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", raw.lower()) if w not in STOPWORDS]
    topic = " ".join(clean_words[:3]) if clean_words else "current challenges"

    return (
        f"I hear how much tension is present around **{topic}**, especially with feeling {emotion}.\n\n"
        f"Here are three grounded steps you can take right now:\n\n"
        f"1. **Somatic Reset**: Inhale slowly for 4 seconds, hold gently for 2, and exhale for 6. Releasing physical tension sends an immediate safety signal to your brain.\n"
        f"2. **Single Micro-Step**: Choose only one small action to focus on for the next 10 minutes. Action dissolves worry far more effectively than overthinking.\n"
        f"3. **Separate What You Control**: Focus your energy on what you can influence in this present hour.\n\n"
        f"Take a slow breath. What would feel most supportive to focus on next?"
    )


def generate_coping_strategies(user_text: str, emotion: str = "stressed", llm=None) -> str:
    """Generate dynamic coping suggestions using LLM, with contextual synthesis fallback."""
    if llm is None:
        try:
            from config.llm_provider import get_llm
            llm = get_llm(model=settings.coping_model)
        except Exception:
            llm = None

    if llm is not None:
        try:
            messages = [
                SystemMessage(content=COPING_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"The user is feeling {emotion}. Their situation is: \"{user_text}\". "
                            f"Generate compassionate, practical, and highly actionable coping micro-steps tailored directly to their situation."
                ),
            ]
            response = llm.invoke(messages)
            content = str(response.content).strip()
            if len(content) > 40:
                return content
        except Exception:
            pass

    return generate_heuristic_coping(user_text=user_text, emotion=emotion)

