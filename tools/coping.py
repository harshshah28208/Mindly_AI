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
    """Dynamic coping strategies dynamically synthesized from user context and emotional state."""
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
    topic = " ".join(clean_words[:3]) if clean_words else "upcoming demands"

    # Context 1: Academic / Viva / Exam / Presentation pressure
    if any(w in raw.lower() for w in ["viva", "vivas", "exam", "exams", "interview", "presentation", "defense", "oral", "test", "study"]):
        return (
            f"### 🎓 Evidence-Based Viva & Exam Performance Strategies\n\n"
            f"Feeling anxious about your **{topic}** is a natural physiological reaction to evaluation pressure. "
            f"Your nervous system is primed for high alert, but we can direct that energy toward calm, structured clarity.\n\n"
            f"#### 1. 🌬️ The 3-Second Pre-Answer Pause\n"
            f"In an oral viva or exam, when the examiner asks a question, **pause for 3 full seconds** before speaking. "
            f"Take one quiet diaphragmatic inhale. This prevents racing speech, demonstrates thoughtful composure, and gives your prefrontal cortex time to organize key concepts.\n\n"
            f"#### 2. 🧠 Principles Over Rote Memorization\n"
            f"Viva examiners test fundamental comprehension, not robotic reciting. If you don't recall a specific term immediately, state the underlying concept: "
            f"*\"The core principle here is... let me walk through how it operates.\"* Examiners appreciate genuine reasoning.\n\n"
            f"#### 3. ⏱️ 10-Minute Micro-Chunking\n"
            f"Don't look at the entire syllabus as an overwhelming mountain. Pick **one single sub-topic** right now and review only its core diagram or 3 key points for 10 minutes.\n\n"
            f"Take a slow breath right now. Would you like to do a quick 4-4-4-4 breathing reset together, or practice explaining one concept clearly?"
        )

    # Context 2: General emotional stress or overwhelm
    return (
        f"### 🌿 Personalized Coping Strategies & Mindful Grounding Plan\n\n"
        f"Navigating challenges around **{topic}** can feel deeply demanding, especially when carrying feelings of {emotion}.\n\n"
        f"1. **Validate Your Biological State**: Acknowledge that feeling {emotion} when managing {topic} is your nervous system's normal response to load. Giving yourself permission to pause immediately reduces internal tension.\n"
        f"2. **The 10-Minute Micro-Action**: When {topic} feels heavy, commit to only ONE single 10-minute focused task. Once 10 minutes pass, step away and stretch.\n"
        f"3. **Somatic Downregulation (4-7-8 Breath)**: Inhale quietly through your nose for 4 seconds, hold gently for 7 seconds, and exhale completely through your mouth for 8 seconds. This activates the vagus nerve and lowers your heart rate.\n\n"
        f"🌱 *Take it one breath at a time. What is one small, gentle action that would feel most comforting right now?*"
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

