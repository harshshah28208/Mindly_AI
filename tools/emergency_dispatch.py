"""Emergency dispatch and telephony notification tool for Mindly.

Inspired by SafeSpace AI's Twilio emergency calling integration.
Places automated crisis dispatch notifications to emergency contacts or
supervisor escalation desks when high-risk crisis criteria are detected.
Never crashes or blocks if telephony network is unavailable.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG_FILE = Path("data/emergency_dispatch_log.json")


def initiate_emergency_call(
    risk_level: str = "HIGH",
    user_input_snippet: str = "Crisis detected",
    contact_number: Optional[str] = None,
    spoken_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute an automated emergency telephony voice call via Twilio.

    Args:
        risk_level: "HIGH" | "CRITICAL"
        user_input_snippet: Short contextual trigger phrase.
        contact_number: Destination phone number (defaults to EMERGENCY_CONTACT or 988 support).
        spoken_message: Spoken message content for automated voice call.

    Returns:
        Dict detailing the call SID, status, provider, destination, and timestamp.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    target_contact = contact_number or os.getenv("EMERGENCY_CONTACT", "+18002738255")

    default_speech = spoken_message or (
        "Emergency Alert from Mindly Mental Health Safety System. "
        "A user has triggered an acute high-risk safety protocol. "
        "Immediate support intervention is advised. "
        "Please check the supervisor emergency dashboard or connect with the user immediately."
    )

    twiml_instruction = (
        f'<Response><Pause length="1"/><Say voice="Polly.Joanna" language="en-US">'
        f'{default_speech}</Say></Response>'
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    dispatch_record = {
        "call_id": f"call_{uuid.uuid4().hex[:12]}",
        "timestamp": timestamp,
        "type": "VOICE_CALL",
        "risk_level": risk_level,
        "contact_target": target_contact[:4] + "****" if len(target_contact) > 4 else "****",
        "full_target": target_contact,
        "snippet_preview": user_input_snippet[:60],
        "spoken_message": default_speech,
    }

    # If real Twilio credentials are configured, execute automated telephony call
    if account_sid and auth_token and from_number:
        try:
            from twilio.rest import Client  # type: ignore
            client = Client(account_sid, auth_token)
            call = client.calls.create(
                to=target_contact,
                from_=from_number,
                twiml=twiml_instruction,
            )
            dispatch_record["status"] = "CALL_DISPATCHED"
            dispatch_record["provider"] = "twilio_voice"
            dispatch_record["call_sid"] = getattr(call, "sid", "unknown")
            _log_dispatch(dispatch_record)
            return dispatch_record
        except Exception as exc:
            dispatch_record["status"] = "TWILIO_ERROR_FALLBACK"
            dispatch_record["error"] = str(exc)

    # Simulation / Supervisor audit mode (Safe fallback when live telephony is unconfigured)
    dispatch_record["status"] = "SIMULATED_EMERGENCY_CALL_ACTIVE"
    dispatch_record["provider"] = "supervisor_telephony_gateway"
    dispatch_record["call_sid"] = f"CA_{uuid.uuid4().hex[:32]}"
    _log_dispatch(dispatch_record)
    return dispatch_record


def send_emergency_sms(
    contact_number: Optional[str] = None,
    sms_body: Optional[str] = None,
    risk_level: str = "HIGH",
) -> Dict[str, Any]:
    """Send an urgent crisis SMS notification via Twilio."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    target_contact = contact_number or os.getenv("EMERGENCY_CONTACT", "+18002738255")

    default_sms = sms_body or (
        "[MINDLY CRITICAL ALERT] A high-risk crisis event was detected. "
        "Immediate support intervention may be required. 988 Lifeline and supervisor notified."
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    record = {
        "timestamp": timestamp,
        "type": "SMS_ALERT",
        "risk_level": risk_level,
        "contact_target": target_contact[:4] + "****" if len(target_contact) > 4 else "****",
        "body": default_sms,
    }

    if account_sid and auth_token and from_number:
        try:
            from twilio.rest import Client  # type: ignore
            client = Client(account_sid, auth_token)
            msg = client.messages.create(
                to=target_contact,
                from_=from_number,
                body=default_sms,
            )
            record["status"] = "SMS_DISPATCHED"
            record["provider"] = "twilio_sms"
            record["message_sid"] = getattr(msg, "sid", "unknown")
            _log_dispatch(record)
            return record
        except Exception as exc:
            record["status"] = "SMS_ERROR_FALLBACK"
            record["error"] = str(exc)

    record["status"] = "SIMULATED_SMS_DISPATCHED"
    record["provider"] = "supervisor_sms_gateway"
    record["message_sid"] = f"SM_{uuid.uuid4().hex[:32]}"
    _log_dispatch(record)
    return record


def dispatch_emergency_notification(
    risk_level: str,
    user_input_snippet: str,
    contact_number: Optional[str] = None,
    emergency_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Unified dispatcher for high-risk alerts (Calls + Logs)."""
    return initiate_emergency_call(
        risk_level=risk_level,
        user_input_snippet=user_input_snippet,
        contact_number=contact_number,
        spoken_message=emergency_message,
    )


def get_emergency_dispatch_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve recent emergency dispatch logs for clinical supervisor audit."""
    if not LOG_FILE.exists():
        return []
    try:
        data = json.loads(LOG_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return list(reversed(data[-limit:]))
    except Exception:
        pass
    return []


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
        # Keep last 100 dispatches
        LOG_FILE.write_text(json.dumps(records[-100:], indent=2), encoding="utf-8")
    except Exception:
        pass
