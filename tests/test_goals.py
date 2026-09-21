"""Tests for Goals and Habits Tracking (Category S)."""

import pytest
from memory.database import db
from tools.goals import handle_goal_command


def test_goal_creation_and_retrieval(clean_test_user):
    """Verify goal creation, defaults, and retrieval for a test user."""
    goal_id = db.add_goal(
        user_id=clean_test_user,
        title="Study discrete mathematics for 2 hours daily",
        description="Exam revision",
        target_date="2026-10-01",
    )
    assert goal_id > 0

    goals = db.get_goals(user_id=clean_test_user, status="active")
    assert len(goals) == 1
    assert goals[0]["title"] == "Study discrete mathematics for 2 hours daily"
    assert goals[0]["progress"] == 0
    assert goals[0]["status"] == "active"


def test_goal_update_and_completion(clean_test_user):
    """Verify updating progress and marking goal completed."""
    goal_id = db.add_goal(
        user_id=clean_test_user,
        title="Practice box breathing every morning",
    )
    
    # Update progress
    success = db.update_goal_progress(goal_id=goal_id, progress=50, status="active")
    assert success is True

    goals = db.get_goals(user_id=clean_test_user)
    assert goals[0]["progress"] == 50

    # Complete goal
    success_comp = db.update_goal_progress(goal_id=goal_id, progress=100, status="completed")
    assert success_comp is True

    completed_goals = db.get_goals(user_id=clean_test_user, status="completed")
    assert len(completed_goals) == 1
    assert completed_goals[0]["status"] == "completed"


def test_handle_goal_command_create_and_list():
    """Verify handle_goal_command parses natural language goal creation and listing."""
    res_create = handle_goal_command("Create a goal to drink 2 liters of water daily.")
    assert "goal created" in res_create.lower() or "added goal" in res_create.lower()

    res_list = handle_goal_command("Show my goals")
    assert "wellness goals" in res_list.lower() or "goals" in res_list.lower()


def test_invalid_goal_id_handling():
    """Verify non-existent goal ID update returns false without raising uncaught exceptions."""
    res = db.update_goal_progress(goal_id=999999, progress=100, status="completed")
    assert res is False


def test_goal_database_failure_resilience(monkeypatch):
    """Verify goal tool catches database exceptions gracefully."""
    def mock_add_goal(*args, **kwargs):
        raise RuntimeError("Database connection lost")

    monkeypatch.setattr(db, "create_goal", mock_add_goal)
    monkeypatch.setattr(db, "add_goal", mock_add_goal)
    res = handle_goal_command("Create a goal to sleep by 11pm")
    assert "error" in res.lower() or "unable" in res.lower()
