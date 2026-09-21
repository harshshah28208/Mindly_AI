"""Long-term memory management tool for Mindly Lab 3."""

import re
from memory.database import db


def handle_memory_command(text: str) -> str:
    """Process persistent memory commands with user consent awareness."""
    lower = text.lower().strip()

    # Check consent
    if not db.is_memory_enabled():
        return (
            "💾 *Long-term memory is currently disabled according to your Companion Preferences. "
            "You can enable memory storage anytime in the Memory Vault tab or sidebar.*"
        )

    # 1. Preferred name save (e.g. "Remember that my preferred name is Rahul", "call me Rahul")
    name_match = re.search(r"(?:preferred\s+name\s+is|call\s+me|my\s+name\s+is)\s+([A-Z][a-zA-Z]+)", text, re.IGNORECASE)
    if name_match:
        name_val = name_match.group(1).capitalize()
        db.save_memory("preferred_name", name_val, category="identity")
        return f"💾 I have remembered that your preferred name is **{name_val}**. I will address you warmly by this name!"

    # 2. Preferred name query (e.g. "What name should you call me?", "What should you call me?")
    if any(k in lower for k in ["what name should you call me", "what should you call me", "what is my preferred name"]):
        saved_name = db.get_memory("preferred_name")
        if saved_name:
            return f"You asked me to call you **{saved_name}**! I have that saved safely in my long-term memory."
        return "You haven't told me a preferred name yet. What would you like me to call you?"

    # 3. View memories
    if any(k in lower for k in ["view memory", "show my memory", "what do you remember about me"]):
        memories = db.get_memories()
        if not memories:
            return "💾 You have no long-term memories stored in your local memory vault."
        lines = ["💾 **Stored User Preferences in Memory Vault:**\n"]
        for m in memories:
            lines.append(f"• **{m['key'].title()}**: {m['value']}")
        return "\n".join(lines)

    # 4. Generic "Remember that <key> is <value>"
    match_general = re.search(r"remember\s+(?:that\s+)?my\s+([a-zA-Z\s]+?)\s+is\s+([a-zA-Z0-9\s]+)", text, re.IGNORECASE)
    if match_general:
        k = match_general.group(1).strip()
        v = match_general.group(2).strip()
        db.save_memory(k, v, category="preference")
        return f"💾 I have saved to long-term memory that your {k} is **{v}**."

    return "I am here. If there is a specific preference you would like me to remember long-term, just let me know!"
