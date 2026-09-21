"""Interactive grounding tools for Mindly.

Provides structured sensory and somatic grounding exercises:
1. Box Breathing (4-4-4-4 somatic nervous system regulation)
2. 5-4-3-2-1 Sensory Grounding (cognitive redirection for acute anxiety)
3. Guided Reflection (mindful pause for emotional untangling)
"""

from typing import Optional


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


def interactive_grounding_tool(activity_type: Optional[str] = None, emotion: str = "stressed") -> str:
    """Execute the selected grounding exercise.

    Args:
        activity_type: 'box_breathing', '54321_grounding', or 'guided_reflection'.
                       If None, selects intelligently based on emotion.
        emotion: The user's estimated emotional state.
    """
    if activity_type == "box_breathing" or (not activity_type and emotion in ["anxious", "overwhelmed"]):
        return run_box_breathing()
    elif activity_type == "54321_grounding" or (not activity_type and emotion in ["confused", "angry"]):
        return run_54321_grounding()
    else:
        return run_guided_reflection(emotion=emotion)
