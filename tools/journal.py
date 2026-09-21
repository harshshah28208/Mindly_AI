"""Wellness journaling tool for Mindly Lab 3."""

import re
from memory.database import db


def handle_journal_command(text: str, mood: str = "reflective") -> str:
    """Process explicit user journaling request and persist entry or view entries."""
    clean = text.strip()
    lower = clean.lower()

    # 1. View / show journal entries
    if any(k in lower for k in ["show my journal", "view journal", "list journal", "my journal entries", "show journal"]):
        try:
            entries = db.get_journal_entries(limit=10)
            if not entries:
                return (
                    "📖 **Your Wellness Journal**\n\n"
                    "You don't have any journal entries yet. "
                    "You can record a thought by saying: *'/journal Today I felt proud of finishing my work.'*"
                )
            lines = ["📖 **Your Recent Wellness Journal Reflections:**\n"]
            for e in entries:
                date_str = e.get("created_at") or e.get("timestamp") or ""
                lines.append(f"• *({date_str})* `[{e.get('mood', 'reflective')}]`\n  > {e.get('entry_text') or e.get('entry')}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: Unable to retrieve journal entries - {str(exc)}"

    # 2. Save new entry
    # Strip trigger prefixes like "journal this:", "journal entry:", "save journal:"
    content = re.sub(
        r"^(?:please\s+)?(?:/journal|journal\s+this|save\s+to\s+journal|journal\s+entry|make\s+a\s+journal\s+entry|journal)\s*[:\-]?\s*",
        "",
        clean,
        flags=re.IGNORECASE,
    ).strip()
    if not content:
        content = clean

    # Estimate mood tag if provided
    tags = ""
    lower_content = content.lower()
    if "stress" in lower_content or "exam" in lower_content:
        tags = "academic,stress"
    elif "grateful" in lower_content or "happy" in lower_content:
        tags = "gratitude"
    elif "calm" in lower_content or "peace" in lower_content:
        tags = "peaceful"
    else:
        tags = "reflection"

    try:
        entry_id = db.create_journal_entry(entry=content, mood=mood, tags=tags)
        return (
            f"📖 **Journal Entry Saved**\n\n"
            f"> \"{content}\"\n\n"
            f"*(Recorded securely in your local reflection journal with mood tag: `{mood}`). "
            f"You can review your past entries at any time in the Wellness Journal tab.)*"
        )
    except Exception as exc:
        return f"Error: Unable to save journal entry - {str(exc)}"
