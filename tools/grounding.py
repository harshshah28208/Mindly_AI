"""Interactive grounding tools for Mindly.

Provides structured sensory and somatic grounding exercises:
1. Box Breathing (4-4-4-4 somatic nervous system regulation)
2. 5-4-3-2-1 Sensory Grounding (cognitive redirection for acute anxiety)
3. Guided Reflection (mindful pause for emotional untangling)

Supports dynamic LLM personalization tailored directly to the user's situation.
"""

from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage


def run_box_breathing() -> str:
    """Provide the 4-4-4-4 box breathing technique."""
    return (
        "### 🌬️ Box Breathing Exercise (4-4-4-4)\n\n"
        "Box breathing is a proven technique used to signal safety to your nervous system and slow your heart rate.\n\n"
        "Find a comfortable posture, relax your shoulders, and follow this gentle cycle:\n\n"
        "1. **Inhale slowly through your nose** for **4 seconds** *(feel your chest and belly expand)*\n"
        "2. **Hold your breath gently** for **4 seconds** *(stay relaxed, no strain)*\n"
        "3. **Exhale smoothly through your mouth** for **4 seconds** *(let tension dissolve)*\n"
        "4. **Hold empty and pause** for **4 seconds** *(rest in the quiet space)*\n\n"
        "🔁 *Repeat this rhythm 3 to 4 times. Notice the subtle release of tension in your jaw and shoulders with each exhale.*"
    )


def run_54321_grounding() -> str:
    """Provide the 5-4-3-2-1 sensory grounding technique."""
    return (
        "### 🌿 5-4-3-2-1 Sensory Grounding Exercise\n\n"
        "When racing thoughts pull you into the future or overwhelm you, anchoring into your immediate physical senses brings you back to the present moment:\n\n"
        "• **5 things you can SEE**: Look around your room. Notice subtle details—a patch of sunlight, a texture on your desk, the color of a book.\n"
        "• **4 things you can TOUCH**: Feel the ground beneath your feet, the fabric of your sleeve, the coolness of a surface, or your hands resting on your lap.\n"
        "• **3 things you can HEAR**: Listen closely. Perhaps a distant hum of traffic, the whisper of air conditioning, or the sound of your own quiet breath.\n"
        "• **2 things you can SMELL**: Take a gentle breath. Can you notice coffee, fresh air, paper, or the subtle scent of your clothing?\n"
        "• **1 thing you can TASTE**: Notice any remaining taste of tea, mint, or simply moisten your lips and swallow softly.\n\n"
        "✨ *Take a slow, deep breath. You are right here, safe in this moment.*"
    )


def run_guided_reflection(emotion: str = "stressed") -> str:
    """Provide a short guided reflection tailored to emotional untangling."""
    return (
        "### 🍃 Guided Mindful Reflection\n\n"
        "Take a slow breath and give yourself permission to pause for a quiet minute. "
        "Reflect gently on these three questions at your own pace:\n\n"
        "1. **What is within my control right now?** *(Focusing on the next small step rather than the entire mountain)*\n"
        "2. **What can I temporarily set down?** *(What expectation or worry doesn't need to be solved in this exact hour?)*\n"
        "3. **What does my body or mind need in this moment?** *(A glass of water, a 10-minute walk, or permission to rest?)*\n\n"
        "💭 *Take your time. What feels like the kindest thing you can do for yourself today?*"
    )


def interactive_grounding_tool(
    activity_type: Optional[str] = None,
    emotion: str = "stressed",
    user_text: str = "",
    llm=None,
) -> str:
    """Execute the selected grounding exercise with dynamic LLM personalization.

    Args:
        activity_type: 'box_breathing', '54321_grounding', or 'guided_reflection'.
                       If None, selects intelligently based on emotion.
        emotion: The user's estimated emotional state.
        user_text: User query context for personalization.
        llm: Language model for generating dynamic attuned grounding.
    """
    resolved_type = activity_type
    if not resolved_type:
        if emotion in ["anxious", "overwhelmed"]:
            resolved_type = "box_breathing"
        elif emotion in ["confused", "angry"]:
            resolved_type = "54321_grounding"
        else:
            resolved_type = "guided_reflection"

    # Dynamic LLM Personalization
    if llm is None and user_text:
        try:
            from config.llm_provider import get_llm
            from config.settings import settings
            budget = settings.determine_token_budget(user_text, intent="GROUNDING_REQUEST")
            llm = get_llm(max_tokens=budget)
        except Exception:
            llm = None

    if llm is not None and user_text:
        try:
            prompt_instruction = ""
            if resolved_type == "box_breathing":
                prompt_instruction = (
                    "Guide the user warmly through the 4-4-4-4 Box Breathing exercise. "
                    "Validate their current feeling, explain how the 4-4-4-4 rhythm calms the autonomic nervous system, "
                    "and guide them through: inhale 4s, hold 4s, exhale 4s, hold 4s."
                )
            elif resolved_type == "54321_grounding":
                prompt_instruction = (
                    "Guide the user warmly through the 5-4-3-2-1 Sensory Grounding exercise. "
                    "Include the exact steps: 5 things you can SEE, 4 things you can TOUCH, 3 things you can HEAR, "
                    "2 things you can SMELL, and 1 thing you can TASTE."
                )
            else:
                prompt_instruction = (
                    "Guide the user through a gentle mindful reflection. "
                    "Offer 3 grounding questions to untangle their thoughts and soothe their overwhelm."
                )

            sys_prompt = (
                "You are a warm, attuned somatic wellness counselor for Mindly. "
                "Personalize this exercise directly for the user's specific feelings and situation. "
                "Keep it soothing, practical, and comforting."
            )
            resp = llm.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=f"User feeling: {emotion}. Context: \"{user_text}\". {prompt_instruction}"),
            ])
            content = str(resp.content).strip()
            if len(content) > 50:
                header = (
                    "### 🌬️ Box Breathing Exercise (4-4-4-4)\n\n" if resolved_type == "box_breathing"
                    else "### 🌿 5-4-3-2-1 Sensory Grounding Exercise\n\n" if resolved_type == "54321_grounding"
                    else "### 🍃 Guided Mindful Reflection\n\n"
                )
                if not content.startswith("#"):
                    return f"{header}{content}"
                return content
        except Exception:
            pass

    # High-quality structured fallback
    if resolved_type == "box_breathing":
        return run_box_breathing()
    elif resolved_type == "54321_grounding":
        return run_54321_grounding()
    else:
        return run_guided_reflection(emotion=emotion)
