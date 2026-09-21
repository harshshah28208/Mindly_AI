"""Direct conversational response tool for Mindly.

Handles standard conversational turns (greetings, identity questions, general reflection)
without forcing unnecessary grounding or coping tools.
"""

import re
from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from agent.prompts import MINDLY_SYSTEM_PROMPT


def generate_direct_response(
    user_text: str,
    messages: Optional[List[BaseMessage]] = None,
    llm=None,
) -> str:
    """Generate a direct conversational reply."""
    # Check conversation history for previously shared user details (like name)
    user_texts = [m.content for m in (messages or []) if isinstance(m, HumanMessage)]
    if user_text and user_text not in user_texts:
        user_texts.append(user_text)

    # Heuristic fact and preference extraction from history
    user_name = None
    user_facts = {}
    for txt in user_texts:
        # Name extraction
        name_match = re.search(r"(?:my name is|i am|i'm)\s+([A-Z][a-zA-Z]+)", str(txt), re.IGNORECASE)
        if name_match:
            user_name = name_match.group(1).capitalize()

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
        # Check direct or partial match in user_facts
        for cat, val in user_facts.items():
            if query_cat in cat or cat in query_cat:
                return f"Your favorite {query_cat} is **{val}**! You shared that with me earlier in our reflection."

    if "what is your name" in lower or "who are you" in lower:
        return (
            "I am **Mindly**, your empathetic mental-wellness support companion. "
            "I'm here to provide a quiet, safe, and confidential space for you to reflect, "
            "untangle your thoughts, or simply be heard without judgment."
        )

    # If LLM is available, invoke it with the empathetic Mindly system prompt
    if llm is not None:
        try:
            sys_msg = SystemMessage(content=MINDLY_SYSTEM_PROMPT)
            history = list(messages or [])
            # Combine history with system prompt
            model_input = [sys_msg] + history
            # If current user_text isn't already the last message in history, add it
            if not history or history[-1].content != user_text:
                model_input.append(HumanMessage(content=user_text))

            response = llm.invoke(model_input)
            content = str(response.content).strip()
            if content:
                return content
        except Exception:
            pass

    # Heuristic fallback for offline/simulation mode
    if any(k in lower for k in ["hello", "hi", "hey", "good morning", "good evening"]):
        suffix = f", {user_name}" if user_name else ""
        return (
            f"Hello{suffix}! I am Mindly, here with you. "
            "Whatever is on your heart or mind today, this is your safe space. How are you feeling right now?"
        )

    suffix = f", {user_name}" if user_name else ""
    return (
        f"Thank you for sharing that with me{suffix}. "
        "I am here and listening attentively. Could you share a bit more about what's been on your mind?"
    )
