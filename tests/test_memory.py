"""Unit tests for Mindly Lab 3 SQLite Persistent Database, Journal, and Goals."""

import unittest
from memory.database import db
from tools.goals import handle_goal_command
from tools.journal import handle_journal_command
from tools.memory import handle_memory_command


class TestMindlyDatabaseAndTools(unittest.TestCase):
    """Test SQLite database operations and dedicated tools."""

    def setUp(self):
        # Ensure memory consent is active
        db.set_memory_enabled(True)

    def test_01_memory_crud_and_consent(self):
        """Verify long-term memory save, retrieve, clear, and consent toggle."""
        db.save_memory("favorite_book", "Atomic Habits", category="books")
        val = db.get_memory("favorite_book")
        self.assertEqual(val, "Atomic Habits")

        # Disable consent
        db.set_memory_enabled(False)
        saved = db.save_memory("disallowed_key", "value")
        self.assertFalse(saved)

        # Re-enable consent
        db.set_memory_enabled(True)
        db.clear_memories()
        self.assertEqual(len(db.get_memories()), 0)

    def test_02_memory_tool_commands(self):
        """Verify memory tool parses user requests."""
        res_save = handle_memory_command("Remember that my preferred name is Rahul.")
        self.assertIn("Rahul", res_save)

        res_query = handle_memory_command("What name should you call me?")
        self.assertIn("Rahul", res_query)

    def test_03_journal_crud(self):
        """Verify journal creation and retrieval."""
        entry_id = db.create_journal_entry("Completed my machine learning lab.", mood="grateful", tags="study")
        self.assertTrue(entry_id > 0)

        entries = db.get_journal_entries(limit=5)
        self.assertTrue(len(entries) > 0)
        first = entries[0]
        self.assertIn("Completed my machine learning lab", first["entry"])
        self.assertEqual(first["mood"], "grateful")

    def test_04_journal_tool_command(self):
        """Verify handle_journal_command parses user journaling inputs."""
        reply = handle_journal_command("Journal this: Had a restful afternoon walk in nature.")
        self.assertIn("Journal Entry Saved", reply)
        self.assertIn("afternoon walk", reply)

    def test_05_goals_crud(self):
        """Verify goal creation, status update, and completion."""
        goal_id = db.create_goal("Study for 2 hours daily", description="Exam preparation")
        self.assertTrue(goal_id > 0)

        # Mark completed
        updated = db.update_goal_status(goal_id, "completed")
        self.assertTrue(updated)

        goals = db.get_goals(status="completed")
        self.assertTrue(any(g["id"] == goal_id for g in goals))

    def test_06_goal_tool_command(self):
        """Verify handle_goal_command parses goal creation and listing."""
        create_res = handle_goal_command("Create a goal to meditate for 10 minutes every morning.")
        self.assertIn("Goal Created Successfully", create_res)
        self.assertIn("meditate", create_res)

        list_res = handle_goal_command("View goals")
        self.assertIn("Your Current Wellness", list_res)


if __name__ == "__main__":
    unittest.main()
