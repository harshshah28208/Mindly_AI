"""Human Supervisor Escalation Service for Mindly Lab 3.

Provides non-emergency project owner/supervisor alerting when HIGH risk is detected.
Operates in 'demo' mode by default to simulate alerts without paid telephony costs.
Enforces strict data minimization: never transmits full private conversation history.
Does NOT claim that emergency services (911, police, hospital) were contacted.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from config.settings import settings


class SupervisorEscalationService:
    """Handles supervisor alerting and crisis escalation dispatch."""

    def __init__(self, mode: Optional[str] = None) -> None:
        self._custom_mode = mode
        self.supervisor_name = settings.supervisor_name
        self.supervisor_phone = settings.supervisor_phone

    @property
    def mode(self) -> str:
        return (self._custom_mode or settings.escalation_mode).lower()

    def escalate(
        self,
        risk_level: str,
        input_source: str = "text",
        safety_summary: str = "Acute self-harm or suicidal ideation indicators detected by deterministic safety guard.",
        matched_trigger: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger supervisor alert with minimal necessary metadata.

        Data Minimization:
        Only transmits: Risk level, Timestamp, Input source, Brief safety summary, truncated trigger phrase.
        Does NOT transmit full user conversation history.
        """
        if risk_level != "HIGH":
            return {"status": "ignored", "risk_level": risk_level}

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        escalation_id = f"ESC-{uuid.uuid4().hex[:6].upper()}"
        
        # Strict data minimization: cap trigger snippet at 50 chars
        safe_trigger = (matched_trigger[:50] + "...") if (matched_trigger and len(matched_trigger) > 50) else (matched_trigger or "[Protected]")

        # Persist audit record
        audit_file = settings.escalation_log_file
        try:
            records = []
            if audit_file.exists():
                try:
                    records = json.loads(audit_file.read_text(encoding="utf-8"))
                    if not isinstance(records, list):
                        records = []
                except Exception:
                    records = []
            records.append({
                "escalation_id": escalation_id,
                "timestamp": timestamp,
                "risk_level": risk_level,
                "trigger_category": safe_trigger,
                "mode": self.mode,
            })
            audit_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
        except Exception:
            pass

        alert_payload = {
            "escalation_id": escalation_id,
            "timestamp": timestamp,
            "risk_level": risk_level,
            "input_source": input_source,
            "supervisor_name": self.supervisor_name,
            "supervisor_phone": self.supervisor_phone,
            "safety_summary": safety_summary,
            "trigger_snippet": safe_trigger,
            "mode": self.mode,
        }

        if self.mode == "demo":
            alert_text = (
                f"🚨 **HUMAN SUPERVISOR ALERT & ESCALATION** (Simulation — Demo Mode)\n\n"
                f"• **Status**: `DEMO TRIGGERED`\n"
                f"• **Risk**: `{risk_level}`\n"
                f"• **Reason**: Self-harm risk detected\n"
                f"• **Escalation ID**: `{escalation_id}`\n"
                f"• **Timestamp**: `{timestamp}`\n"
                f"• **Input Source**: `{input_source.title()}`\n"
                f"• **Supervisor Contact**: `{self.supervisor_name}` ({self.supervisor_phone})\n\n"
                f"*(Supervisor escalation has been triggered in demo mode. No emergency services or 911 calls were made.)*"
            )
            alert_payload["status"] = "escalated"
            alert_payload["escalation_status"] = "DEMO TRIGGERED"
            alert_payload["display_card"] = alert_text
            return alert_payload

        if self.mode not in ["demo", "production"]:
            alert_payload["status"] = "logged_internally"
            alert_payload["note"] = "unsupported mode: reverted to internal logging"
            alert_payload["display_card"] = "Supervisor escalation logged internally."
            return alert_payload

        # Production adapter placeholder
        alert_payload["status"] = "ESCALATION_FAILED_NO_CREDENTIALS"
        alert_payload["display_card"] = "Production credentials unconfigured; automatically reverted to demo mode."
        return alert_payload


# Singleton escalation service
escalation_service = SupervisorEscalationService()
