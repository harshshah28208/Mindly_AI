"""Direct conversational response tool for Mindly.

Handles standard conversational turns (greetings, identity questions, general reflection)
and scope boundary advisories without returning static repetitive canned answers.
"""

import os
import re
from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from agent.prompts import MINDLY_SYSTEM_PROMPT
from agent.intent import evaluate_math_expression
from config.llm_provider import DynamicLocalResponseEngine


SCOPE_NOTICE_MESSAGE = (
    "🌿 **Medical Domain Scope Boundary Notice**:\n\n"
    "Mindly is dedicated exclusively as an **Agentic AI Medical & Wellness Copilot** "
    "(grounded in the *Gale Encyclopedia of Medicine* and multi-agent clinical protocols).\n\n"
    "This inquiry falls outside my specialized clinical health and wellness scope. "
    "I do not perform general calculations, programming tasks, or non-medical queries.\n\n"
    "**In-Scope Inquiries You Can Explore:**\n"
    "• Clinical diseases & conditions (e.g. *Hypertension*, *Migraine*, *Diabetes*, *Asthma*, *Arthritis*)\n"
    "• Physical symptoms & etiology (e.g. *Causes of acute stomach pain and the brain-gut axis*)\n"
    "• Stress regulation, breathing exercises, and evidence-based clinical wellness management."
)

CLARIFICATION_MESSAGE = (
    "I'm here with you, and I want to make sure I support you with exactly what you need right now.\n\n"
    "Here are some ways we can work together:\n"
    "• **Emotional Support & Reflection**: Talk through difficult feelings, stress, burnout, or relationships.\n"
    "• **Grounding & Calming Exercises**: Step through Box Breathing (4-4-4-4), 5-4-3-2-1 sensory grounding, or body awareness.\n"
    "• **Actionable Coping Strategies**: Practical steps for anxiety, overthinking, sleep hygiene, or panic.\n"
    "• **Health & Medical Information**: Clear, evidence-based health guidance from the *Gale Encyclopedia of Medicine*.\n"
    "• **Reflective Journal & Habits**: Log your mood, review thoughts, or set mindful personal goals.\n\n"
    "Could you share a bit more about what you're experiencing or what would be most helpful right now?"
)


def generate_direct_response(
    user_text: str,
    messages: Optional[List[BaseMessage]] = None,
    llm=None,
    is_scope_notice: bool = False,
    is_clarification: bool = False,
) -> str:
    """Generate a dynamic conversational reply, scope guidance, or clarification prompt."""
    # Ambiguous or underspecified input clarification prompt
    if is_clarification:
        return CLARIFICATION_MESSAGE

    # Enforce medical domain scope boundary on explicit notice or math
    if is_scope_notice or evaluate_math_expression(user_text) is not None:
        return SCOPE_NOTICE_MESSAGE


    # Check conversation history for previously shared user details (like name)
    user_texts = [m.content for m in (messages or []) if isinstance(m, HumanMessage)]
    if user_text and user_text not in user_texts:
        user_texts.append(user_text)

    INVALID_NAME_WORDS = {
        "feeling", "having", "struggling", "hurting", "crying", "dying", "trying",
        "depressed", "anxious", "stressed", "sad", "tired", "done", "sick", "lost",
        "not", "just", "really", "so", "very", "looking", "asking", "wondering",
        "thinking", "here", "ready", "going", "fine", "okay", "good", "bad", "alive",
        "alone", "afraid", "scared", "experiencing", "suffering", "seeking", "getting",
        "in", "at", "on", "a", "an", "the", "hoping", "dealing", "facing", "in", "a"
    }

    # Heuristic fact and preference extraction from history
    user_name = None
    user_facts = {}
    for txt in user_texts:
        # Name extraction (only if not an emotional or physical state word)
        name_match = re.search(r"\b(?:my name is|i am|i'm)\s+([A-Za-z]+)", str(txt), re.IGNORECASE)
        if name_match:
            candidate = name_match.group(1).strip()
            if candidate.lower() not in INVALID_NAME_WORDS and len(candidate) > 1:
                user_name = candidate.capitalize()
                try:
                    from memory.database import db
                    if db.is_memory_enabled():
                        db.save_memory("preferred_name", user_name, category="identity")
                except Exception:
                    pass

        # Favorite items extraction (e.g. "my favorite subject is Artificial Intelligence")
        fav_matches = re.findall(r"my favorite\s+([a-zA-Z\s]+?)\s+is\s+([a-zA-Z0-9\s]+?)(?:\.|$|,)", str(txt), re.IGNORECASE)
        for cat, val in fav_matches:
            user_facts[cat.strip().lower()] = val.strip()

    lower = user_text.lower().strip()

    # Identity and name queries
    if "what is my name" in lower or "who am i" in lower:
        name_to_report = user_name
        if not name_to_report:
            try:
                from memory.database import db
                name_to_report = db.get_memory("preferred_name")
            except Exception:
                pass
        if name_to_report:
            return f"Your name is **{name_to_report}**! You shared that with me earlier in our reflection."
        return "You haven't shared your name with me yet. What would you like me to call you?"

    # Favorite facts recall queries (e.g. "what is my favorite subject?")
    fav_query_match = re.search(r"what(?:'s|\s+is)\s+my\s+favorite\s+([a-zA-Z\s]+?)(?:\?|$)", lower)
    if fav_query_match:
        query_cat = fav_query_match.group(1).strip().lower()
        for cat, val in user_facts.items():
            if query_cat in cat or cat in query_cat:
                return f"Your favorite {query_cat} is **{val}**! You shared that with me earlier in our reflection."

    # Missing image check (e.g. "What is in this picture?" when no image is attached)
    if any(q in lower for q in [
        "what is in this picture", "what is in this image", "what's in this picture",
        "describe this picture", "analyze this image", "what do you see in this picture",
        "what is this image", "describe this image", "look at this picture"
    ]):
        return (
            "📷 I don't see an image attached to your message. "
            "Please upload an image using the attachment button so I can view and analyze it for you!"
        )

    # Mood recall check (e.g. "What did I tell you about my mood earlier?", "What was my mood earlier?")
    if any(q in lower for q in [
        "what did i tell you about my mood", "what was my mood earlier", "what did i say about my mood",
        "my mood earlier", "did i tell you my mood"
    ]):
        from memory.database import db
        try:
            saved_mood = db.get_memory("today_mood") or db.get_memory("mood")
            if not saved_mood:
                entries = db.get_journal_entries(limit=5)
                for e in entries:
                    if e.get("mood") and e.get("mood") != "reflective":
                        saved_mood = f"{e.get('mood')} (Recorded entry: \"{e.get('entry_text') or e.get('entry')}\")"
                        break
                    elif "mood" in (e.get("entry_text") or e.get("entry") or "").lower():
                        saved_mood = e.get("entry_text") or e.get("entry")
                        break
            if saved_mood:
                return f"Earlier, you recorded your mood as: **{saved_mood}**."
            return (
                "I checked your records, but you haven't mentioned or recorded your mood yet in this session or journal. "
                "How are you feeling right now? If you'd like, we can record it together!"
            )
        except Exception:
            return "You haven't recorded or mentioned your mood yet in our conversation. How are you feeling right now?"

    # If LLM is available, invoke it with the empathetic Mindly system prompt
    if llm is None and not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from config.llm_provider import get_llm
            from config.settings import settings
            budget = settings.determine_token_budget(user_text, intent="GENERAL_CONVERSATION")
            llm = get_llm(max_tokens=budget)
        except Exception:
            pass

    if llm is not None:
        try:
            sys_msg = SystemMessage(content=MINDLY_SYSTEM_PROMPT)
            history = list(messages or [])
            # Combine history with system prompt
            model_input = [sys_msg] + history
            if not history or history[-1].content != user_text:
                model_input.append(HumanMessage(content=user_text))

            response = llm.invoke(model_input)
            content = str(response.content).strip()
            if content and len(content) > 10:
                if bool(re.search(r"^\s*(?:hello|hi|hey|greetings)\b", user_text.lower())) and "mindly" not in content.lower():
                    content = f"Hello! I am Mindly, your mental-wellness companion. {content}"
                if bool(re.search(r"what is your name|who are you", user_text.lower())) and "mindly" not in content.lower():
                    content = f"I am Mindly, your mental-wellness companion. {content}"
                return content
        except Exception:
            pass

    # Dynamic Contextual Synthesis (Dynamic engine replacing fixed static responses)
    full_msgs = list(messages or [])
    if not full_msgs or full_msgs[-1].content != user_text:
        full_msgs.append(HumanMessage(content=user_text))

    return DynamicLocalResponseEngine.generate(full_msgs)
