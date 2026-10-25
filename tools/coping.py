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
    """Dynamic fallback coping strategies dynamically synthesized from user context and emotional state."""
    clean_words = [
        w for w in re.findall(r"\b[a-zA-Z]{4,}\b", user_text.lower())
        if w not in {"what", "this", "that", "with", "have", "from", "they", "will", "would", "about", "there", "their", "could", "should", "your", "mindly", "feel", "feeling"}
    ]
    topic = " ".join(clean_words[:4]) if clean_words else "what you are experiencing"

    return (
        f"### 🌿 Personalized Coping & Mindful Grounding Plan\n\n"
        f"Navigating challenges around **{topic}** can feel deeply demanding, especially when carrying feelings of {emotion}.\n\n"
        f"1. **Validate Your Experience**: Acknowledge that feeling {emotion} around {topic} is a natural biological reaction to pressure. Giving yourself permission to feel this immediately eases internal resistance.\n"
        f"2. **The 10-Minute Micro-Action**: When {topic} feels overwhelming, avoid looking at the entire mountain. Commit to just ONE single 10-minute focused micro-step.\n"
        f"3. **Somatic Physiological Reset**: Take 3 slow, deep abdominal breaths (inhale for 4 seconds, pause, exhale smoothly for 6 seconds) to downregulate nervous system arousal.\n\n"
        f"🌱 *Take it one step at a time. What is one small, gentle action that would feel most supportive right now?*"
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

