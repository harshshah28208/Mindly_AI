"""Tests for Wellness Journaling CRUD, Persistence, and Failure Handling (Category R)."""

import pytest
from memory.database import db
from tools.journal import handle_journal_command


def test_journal_entry_save_and_retrieval(clean_test_user):
    """Verify journal entry creation, timestamping, and listing for clean user."""
    # 1. Save entry
    entry_id = db.add_journal(
        user_id=clean_test_user,
        entry_text="Today I finished my AI project and felt relieved.",
        mood="peaceful",
    )
    assert entry_id > 0

    # 2. Retrieve entry
    journals = db.get_journals(user_id=clean_test_user, limit=5)
    assert len(journals) >= 1
    first = journals[0]
    assert first["entry_text"] == "Today I finished my AI project and felt relieved."
    assert first["mood"] == "peaceful"
    assert "created_at" in first
    assert len(first["created_at"]) > 5


def test_handle_journal_command_save():
    """Verify natural language journal handler successfully writes entries."""
    res = handle_journal_command("Journal this: Had a restful walk in the park.", mood="relaxed")
    assert "saved to your Wellness Journal" in res.lower() or "journal entry" in res.lower()


def test_handle_journal_command_show():
    """Verify natural language request to list entries returns saved reflections."""
    handle_journal_command("Writing down my morning gratitude.", mood="grateful")
    res = handle_journal_command("Show my journal entries")
    assert "wellness journal" in res.lower()


def test_journal_database_failure_resilience(monkeypatch):
    """Verify application does not crash if database operations encounter an error."""
    def mock_add_journal(*args, **kwargs):
        raise RuntimeError("Disk I/O failure simulated")

    monkeypatch.setattr(db, "create_journal_entry", mock_add_journal)
    monkeypatch.setattr(db, "add_journal", mock_add_journal)
    res = handle_journal_command("Journal this: Attempting to save under failure.")
    assert "error" in res.lower() or "unable to save" in res.lower()
