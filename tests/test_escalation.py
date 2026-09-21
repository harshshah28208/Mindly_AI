"""Tests for Human Supervisor Escalation in Demo Mode (Category V)."""

import json
from pathlib import Path
import pytest
from config.settings import settings
from tools.escalation import escalation_service, SupervisorEscalationService


def test_v_supervisor_escalation_demo_mode():
    """Category V: Verify escalation in demo mode logs alert without claiming real 911 dispatch."""
    res = escalation_service.escalate(
        risk_level="HIGH",
        input_source="text",
        matched_trigger="I want to hurt myself",
    )
    assert res["status"] == "escalated"
    assert res["mode"] == "demo"
    assert "escalation_id" in res
    assert "demo" in res["display_card"].lower()
    # Must NOT claim a real phone call or 911 dispatch
    assert "dispatched 911" not in res["display_card"].lower()
    assert "calling emergency services" not in res["display_card"].lower()


def test_v_supervisor_escalation_data_minimization():
    """Category V: Verify supervisor alert adheres to data minimization and excludes long private transcripts."""
    long_private_story = "My private personal story that should not be broadcasted..." * 20
    res = escalation_service.escalate(
        risk_level="HIGH",
        matched_trigger=long_private_story,
    )
    # The recorded trigger snippet should be strictly capped (e.g. 60 chars)
    audit_file = settings.escalation_log_file
    assert audit_file.exists()
    records = json.loads(audit_file.read_text(encoding="utf-8"))
    latest = records[-1]
    assert len(latest["trigger_category"]) <= 100
    assert long_private_story not in latest["trigger_category"]


def test_v_supervisor_escalation_invalid_mode(monkeypatch):
    """Category V: Verify invalid escalation mode safely falls back to demo mode."""
    service = SupervisorEscalationService()
    monkeypatch.setattr(settings, "escalation_mode", "unknown_invalid_mode")
    res = service.escalate(risk_level="HIGH")
    assert res["status"] == "logged_internally"
    assert "unsupported mode" in res["note"].lower()


def test_v_supervisor_escalation_low_risk_ignored():
    """Verify LOW risk calls do not produce escalation alerts."""
    res = escalation_service.escalate(risk_level="LOW")
    assert res["status"] == "ignored"
    assert "escalation_id" not in res
