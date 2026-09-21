"""Goal management and planning tool for Mindly Lab 3."""

import re
from typing import Optional
from memory.database import db


def handle_goal_command(text: str) -> str:
    """Process goal creation, viewing, and updates."""
    lower = text.lower().strip()

    # 1. View / list goals
    if any(k in lower for k in ["view goals", "list goals", "my goals", "show my goals", "what are my goals", "show goals"]):
        try:
            goals = db.get_goals(status="active")
            if not goals:
                return (
                    "🎯 **Your Active Goals**\n\n"
                    "You don't have any active goals saved yet. "
                    "You can create one by saying something like: *'Create a goal to study for 2 hours daily.'*"
                )
            lines = ["🎯 **Your Current Wellness & Habit Goals:**\n"]
            for g in goals:
                target = f" *(Target: {g['target_date']})*" if g.get("target_date") else ""
                lines.append(f"• **Goal #{g['id']}**: {g['title']}{target} — `[{g['status'].upper()}]`")
            lines.append("\n*You can mark any goal completed by saying 'Complete goal #ID'.*")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: Unable to retrieve goals - {str(exc)}"

    # 2. Complete goal
    complete_match = re.search(r"(?:complete|finish|mark\s+completed?)\s+(?:goal\s+)?#?(\d+)", lower)
    if complete_match:
        try:
            goal_id = int(complete_match.group(1))
            success = db.update_goal_status(goal_id, "completed")
            if success:
                return f"🎉 **Goal Completed!** Goal #{goal_id} has been marked as completed. Wonderful work on your progress!"
            return f"Could not find active goal #{goal_id} to complete."
        except Exception as exc:
            return f"Error: Unable to complete goal - {str(exc)}"

    # 3. Create goal
    # Strip prefixes like "create goal:", "set a goal to", "i want to"
    title = re.sub(
        r"^(?:please\s+)?(?:/goal|create\s+(?:a\s+)?goal\s+(?:to\s+)?|set\s+(?:a\s+)?goal\s+(?:to\s+)?|add\s+(?:a\s+)?goal\s+(?:to\s+)?|my\s+goal\s+is\s+to\s+)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    if not title:
        title = text.strip()

    try:
        goal_id = db.create_goal(title=title)
        return (
            f"🎯 **Goal Created Successfully** (Goal #{goal_id})\n\n"
            f"• **Title**: {title}\n"
            f"• **Status**: `ACTIVE`\n\n"
            f"*(Stored in your local Goals & Habits tracker. You can view all goals by asking 'Show my goals' or in the Goals tab.)*"
        )
    except Exception as exc:
        return f"Error: Unable to create goal - {str(exc)}"
