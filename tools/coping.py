"""Coping strategies tool for Mindly.

Provides structured, evidence-informed wellness strategies and cognitive reframing.
Supports secondary model configuration (e.g. MedGemma or specialized coping model)
with automatic fallback to the primary model or deterministic templates.
"""

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
    """Deterministic, high-quality coping strategies for offline or fallback execution."""
    lower = user_text.lower()

    if any(k in lower for k in ["exam", "study", "college", "deadline", "assignment", "workload"]):
        return (
            "### 📚 Actionable Strategies for Academic Workload & Exam Stress\n\n"
            "Academic pressure can easily trigger cognitive overload. Here are three grounded steps to regain control:\n\n"
            "1. **The 15-Minute Micro-Start**: Instead of trying to tackle an entire syllabus or paper, commit to working on just ONE specific paragraph or problem for 15 minutes. Starting dissolves resistance.\n"
            "2. **Triage Your Tasks (Brain Dump & Sort)**: Write everything down on physical paper. Separate into:\n"
            "   - *Must do today* (limit to 1 or 2 items)\n"
            "   - *Can wait until tomorrow*\n"
            "   - *Things out of your control*\n"
            "3. **Structured Rest Intervals**: Work in 25-minute bursts followed by a complete 5-minute cognitive break away from all screens.\n\n"
            "💡 *Remember: Your worth as a human being is not measured by academic perfection. Progress over perfection.*"
        )
    elif any(k in lower for k in ["sleep", "tired", "insomnia", "night"]):
        return (
            "### 🌙 Gentle Evening Decompression & Sleep Routine\n\n"
            "When thoughts keep your mind awake, trying to force sleep often creates more tension. Try these bedtime coping habits:\n\n"
            "1. **Worry Download**: Keep a notepad by your bed. Write down all unfinished thoughts with the note: *'I have noted these down; they will wait for tomorrow morning.'*\n"
            "2. **4-7-8 Evening Breathing**: Inhale quietly for 4 seconds, hold for 7 seconds, exhale audibly through your mouth for 8 seconds. Repeat 4 cycles.\n"
            "3. **Screen Demilitarized Zone**: Dim room lighting and switch screens off 30 minutes before sleep to allow melatonin release.\n\n"
            "✨ *Resting quietly in bed is still restoring your body, even if your mind hasn't completely drifted off yet.*"
        )
    else:
        return (
            "### 🛡️ Daily Emotional Regulation & Stress Management Strategies\n\n"
            "When emotional tension builds up, here is a practical framework to steady yourself:\n\n"
            "1. **Name What You Are Carrying**: Put words to the feeling (e.g., *'I am noticing feelings of anxiety about the future'*). Naming emotions engages your prefrontal cortex and reduces amygdala reactivity.\n"
            "2. **Physical Reset**: Step outside for 5 minutes, drink a cold glass of water, or gently stretch your neck and shoulders.\n"
            "3. **Set a Micro-Boundary**: Give yourself permission to say 'not right now' to non-essential commitments for the next 2 hours.\n\n"
            "🌱 *If stress or anxiety feels persistent and begins disrupting your daily functioning, consider speaking with your campus counseling center or a healthcare professional.*"
        )


def generate_coping_strategies(user_text: str, emotion: str = "stressed", llm=None) -> str:
    """Generate coping suggestions using LLM if available, otherwise heuristics."""
    if llm is not None:
        try:
            messages = [
                SystemMessage(content=COPING_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"The user is feeling {emotion}. Their input is: \"{user_text}\". "
                            f"Provide supportive, structured coping strategies."
                ),
            ]
            response = llm.invoke(messages)
            content = str(response.content).strip()
            if len(content) > 50:
                return content
        except Exception:
            pass

    return generate_heuristic_coping(user_text=user_text, emotion=emotion)
