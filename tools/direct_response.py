"""Direct conversational response tool for Mindly.

Handles standard conversational turns (greetings, identity questions, general reflection)
and scope boundary advisories without returning static repetitive canned answers.
"""

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


def generate_direct_response(
    user_text: str,
    messages: Optional[List[BaseMessage]] = None,
    llm=None,
    is_scope_notice: bool = False,
) -> str:
    """Generate a dynamic conversational reply or medical/wellness scope guidance."""
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

        # Favorite items extraction (e.g. "my favorite subject is Artificial Intelligence")
        fav_matches = re.findall(r"my favorite\s+([a-zA-Z\s]+?)\s+is\s+([a-zA-Z0-9\s]+?)(?:\.|$|,)", str(txt), re.IGNORECASE)
        for cat, val in fav_matches:
            user_facts[cat.strip().lower()] = val.strip()

    lower = user_text.lower().strip()

    # Identity and name queries
    if "what is my name" in lower or "who am i" in lower:
        if user_name:
            return f"Your name is **{user_name}**! You shared that with me earlier in our reflection."
        return "You haven't shared your name with me yet. What would you like me to call you?"

    # Favorite facts recall queries (e.g. "what is my favorite subject?")
    fav_query_match = re.search(r"what(?:'s|\s+is)\s+my\s+favorite\s+([a-zA-Z\s]+?)(?:\?|$)", lower)
    if fav_query_match:
        query_cat = fav_query_match.group(1).strip().lower()
        for cat, val in user_facts.items():
            if query_cat in cat or cat in query_cat:
                return f"Your favorite {query_cat} is **{val}**! You shared that with me earlier in our reflection."


    # If LLM is available, invoke it with the empathetic Mindly system prompt
    if llm is None:
        try:
            from config.llm_provider import get_llm
            llm = get_llm()
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
