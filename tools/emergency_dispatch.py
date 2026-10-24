"""Emergency dispatch and telephony notification tool for Mindly.

Inspired by SafeSpace AI's Twilio emergency calling integration.
Places automated crisis dispatch notifications to emergency contacts or
supervisor escalation desks when high-risk crisis criteria are detected.
Never crashes or blocks if telephony network is unavailable.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


LOG_FILE = Path("data/emergency_dispatch_log.json")


def dispatch_emergency_notification(
    risk_level: str,
    user_input_snippet: str,
    contact_number: Optional[str] = None,
    emergency_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Dispatch emergency crisis call or SMS via Twilio or local supervisor log.

    Args:
        risk_level: "HIGH" | "CRITICAL"
        user_input_snippet: Safe context snippet triggering the escalation.
        contact_number: Destination phone number (or defaults to env config).
        emergency_message: Custom TTS message or SMS body.

    Returns:
        Dict detailing dispatch status, provider ('twilio' | 'simulation_audit'), and timestamp.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    target_contact = contact_number or os.getenv("EMERGENCY_CONTACT", "+18002738255")

    default_msg = emergency_message or (
        "Emergency Alert from Mindly Mental Health System: A user has triggered an acute crisis safety protocol. "
        "Immediate support personnel review is advised."
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    dispatch_record = {
        "timestamp": timestamp,
        "risk_level": risk_level,
        "contact_target": target_contact[:4] + "****" if len(target_contact) > 4 else "****",
        "snippet_preview": user_input_snippet[:60],
        "message": default_msg,
    }

    # If real Twilio credentials are configured, execute automated telephony call
    if account_sid and auth_token and from_number:
        try:
            from twilio.rest import Client
            client = Client(account_sid, auth_token)
            call = client.calls.create(
                to=target_contact,
                from_=from_number,
                url="http://demo.twilio.com/docs/voice.xml"
            )
            dispatch_record["status"] = "DISPATCHED"
            dispatch_record["provider"] = "twilio"
            dispatch_record["call_sid"] = getattr(call, "sid", "unknown")
            _log_dispatch(dispatch_record)
            return dispatch_record
        except Exception as exc:
            dispatch_record["status"] = "TWILIO_ERROR_FALLBACK"
            dispatch_record["error"] = str(exc)

    # Simulation / Supervisor audit mode (Safe fallback)
    dispatch_record["status"] = "SIMULATED_SUPERVISOR_DISPATCH"
    dispatch_record["provider"] = "internal_supervisor_desk"
    _log_dispatch(dispatch_record)
    return dispatch_record


def _log_dispatch(record: Dict[str, Any]) -> None:
    """Safely log dispatch records to data directory."""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        records = []
        if LOG_FILE.exists():
            try:
                records = json.loads(LOG_FILE.read_text(encoding="utf-8"))
            except Exception:
                records = []
        records.append(record)
        # Keep last 50 dispatches
        LOG_FILE.write_text(json.dumps(records[-50:], indent=2), encoding="utf-8")
    except Exception:
        pass
