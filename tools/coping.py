import re
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import settings


COPING_SYSTEM_PROMPT = """You are Mindly, an empathetic wellness specialist and supportive counselor.
Provide warm, intelligent, practical, and compassionate guidance tailored directly to the user's inquiry.
- Speak naturally and conversationally like a caring expert counselor.
- Address their exact situation directly with clear, practical steps.
- Validate their feelings without rigid robotic templates or artificial scripts.
- Offer actionable advice, healthy reframing, and supportive next steps.
- Never force an emotional anxiety narrative onto physical symptom questions."""


def generate_heuristic_coping(user_text: str, emotion: str = "stressed") -> str:
    """Dynamic coping fallback synthesized from user context and emotional state."""
    raw = user_text.strip()
    is_academic = any(w in raw.lower() for w in ["viva", "vivas", "exam", "exams", "study", "workload", "college", "school"])

    if is_academic:
        return (
            "Dealing with heavy academic workloads and exam pressure can feel exhausting. Here are some practical steps to help you navigate it:\n\n"
            "1. **Break It Down with the 20-Minute Focus Block**: Don't try to tackle everything at once. Pick one specific concept or sub-topic and set a timer for 20 minutes, followed by a 5-minute break.\n"
            "2. **Identify High-Yield Tasks**: Prioritize the 20% of material that drives 80% of results. Ask yourself: 'What is the one essential topic I need to understand today?'\n"
            "3. **Physical Reset**: When tension builds up in your chest or shoulders, take 3 slow diaphragmatic breaths. Releasing muscular tension resets your nervous system.\n\n"
            "Be kind to yourself as you work through this. What specific topic or task is feeling heaviest right now?"
        )

    return (
        "When things feel overwhelming, taking a structured step back can help bring clarity and relief. Here are some actionable strategies:\n\n"
        "1. **Pause and Breathe**: Inhale slowly for 4 seconds, hold for 2, and exhale for 6. Slowing your breath signals to your brain that you are safe in this moment.\n"
        "2. **Differentiate Control**: Separate what you can influence right now from what is outside your immediate control. Direct your energy only toward the next small action.\n"
        "3. **Focus on the Next Smallest Step**: Overwhelm dissolves when you narrow your attention to one simple action you can complete in the next 15 minutes.\n\n"
        "I'm here to support you. What part of this feels most challenging today?"
    )


def generate_coping_strategies(user_text: str, emotion: str = "stressed", llm=None) -> str:
    """Generate dynamic coping suggestions using LLM, with contextual synthesis fallback."""
    if llm is None:
        try:
            from config.llm_provider import get_llm
            budget = settings.determine_token_budget(user_text, intent="COPING_REQUEST")
            llm = get_llm(model=settings.coping_model, max_tokens=budget)
        except Exception:
            llm = None

    if llm is not None:
        try:
            messages = [
                SystemMessage(content=COPING_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"User inquiry: \"{user_text}\"\n"
                            f"Context: The user may be feeling {emotion}. "
                            f"Provide compassionate, practical, and highly actionable strategies directly tailored to their exact query."
                ),
            ]
            response = llm.invoke(messages)
            content = str(response.content).strip()
            if len(content) > 40:
                return content
        except Exception:
            pass

    return generate_heuristic_coping(user_text=user_text, emotion=emotion)

