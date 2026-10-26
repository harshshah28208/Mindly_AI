"""Long-term memory management tool for Mindly."""

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

    # 1. Preferred name update or save (e.g. "Remember that my preferred name is Rahul", "call me Rahul", "update my preferred name to Alex")
    name_match = re.search(r"(?:preferred\s+name\s+is|call\s+me|my\s+name\s+is|preferred\s+name\s+to|change\s+my\s+preferred\s+name\s+to|update\s+my\s+preferred\s+name\s+to)\s+([A-Z][a-zA-Z]+)", text, re.IGNORECASE)
    if name_match:
        name_val = name_match.group(1).capitalize()
        db.save_memory("preferred_name", name_val, category="identity")
        return f"💾 I have remembered that your preferred name is **{name_val}**. I will address you warmly by this name!"

    # 2. Preferred name query (e.g. "What is my name?", "What name should you call me?", "What should you call me?")
    if any(k in lower for k in [
        "what name should you call me", "what should you call me", "what is my preferred name",
        "what is my name", "what's my name", "who am i", "do you know my name"
    ]):
        saved_name = db.get_memory("preferred_name")
        if saved_name:
            return f"Your name is **{saved_name}**! You shared that with me earlier in our reflection."
        return "You haven't told me your name yet. What would you like me to call you?"

    # 3. Mood recall (e.g. "What did I tell you about my mood earlier?", "What was my mood earlier?")
    if any(k in lower for k in [
        "what did i tell you about my mood", "what was my mood earlier", "what did i say about my mood",
        "my mood earlier", "did i tell you my mood"
    ]):
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

    # 4. Deletion and purge operations
    if any(k in lower for k in ["clear my memories", "clear all memories", "clear all memory", "delete all memories", "clear memories", "reset memory", "delete my memory"]):
        try:
            db.clear_memories()
            return "🗑️ All long-term memories have been successfully cleared from your local database."
        except Exception as exc:
            return f"Error: Unable to clear memories - {str(exc)}"

    if any(k in lower for k in ["forget my preferred name", "delete my preferred name", "forget my name", "remove my preferred name"]):
        try:
            saved = db.get_memory("preferred_name")
            if saved:
                with db._get_connection() as conn:
                    conn.execute("DELETE FROM memories WHERE key = 'preferred_name'")
                    conn.commit()
                return "🗑️ I have removed your preferred name from long-term memory."
            return "I do not currently have a preferred name stored in memory."
        except Exception as exc:
            return f"Error: Unable to delete preferred name - {str(exc)}"

    # 5. View memories
    if any(k in lower for k in ["view memory", "show my memory", "what do you remember about me"]):
        memories = db.get_memories()
        if not memories:
            return "💾 You have no long-term memories stored in your local memory vault."
        lines = ["💾 **Stored User Preferences in Memory Vault:**\n"]
        for m in memories:
            lines.append(f"• **{m['key'].title()}**: {m['value']}")
        return "\n".join(lines)

    # 6. Generic "Remember that <key> is <value>"
    match_general = re.search(r"remember\s+(?:that\s+)?my\s+([a-zA-Z\s]+?)\s+is\s+([a-zA-Z0-9\s]+)", text, re.IGNORECASE)
    if match_general:
        k = match_general.group(1).strip()
        v = match_general.group(2).strip()
        db.save_memory(k, v, category="preference")
        return f"💾 I have saved to long-term memory that your {k} is **{v}**."

    return "I am here. If there is a specific preference you would like me to remember long-term, just let me know!"
