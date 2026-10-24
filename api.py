"""Mindly Agentic AI — REST API Backend.

Provides FastAPI REST endpoints compatible with SafeSpace AI,
allowing external clients, web apps, and mobile frontends to query
the Mindly LangGraph StateGraph agent directly.
"""

import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph import send_message
from tools.rag import get_suggested_followups
from tools.therapist_locator import find_nearby_therapists


app = FastAPI(
    title="Mindly Agentic AI API",
    description="Empathetic Mental Wellness & Clinical RAG Multi-Agent API inspired by SafeSpace AI",
    version="2.0.0",
)

# Enable CORS for external frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskQuery(BaseModel):
    message: str = Field(..., description="User query or message")
    thread_id: Optional[str] = Field(None, description="Optional thread session ID")
    location: Optional[str] = Field(None, description="Optional city or region for therapist lookup")


class AskResponse(BaseModel):
    response: str
    tool_called: str


class ExtendedChatResponse(BaseModel):
    response: str
    tool_called: str
    risk_level: str
    detected_intent: Optional[str] = None
    detected_emotion: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    followups: List[str] = []
    escalation: Optional[str] = None
    thread_id: str


class TherapistQuery(BaseModel):
    location: str = Field(..., description="City or area to find licensed mental health professionals")


@app.get("/")
def root():
    return {
        "service": "Mindly Agentic AI Copilot",
        "version": "2.0.0",
        "status": "online",
        "architecture": "LangGraph StateGraph + Gale Encyclopedia of Medicine RAG",
        "features": [
            "Clinical Disease Q&A with Reason & Cure Plan",
            "SafeSpace-compatible /ask endpoint",
            "Licensed Therapist Locator Tool",
            "Automated Crisis Telephony Dispatch",
            "Gemini/ChatGPT Style Dynamic Follow-up Suggestions",
        ]
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "mindly-agentic-api"}


@app.post("/ask", response_model=AskResponse)
def ask_safespace_compatible(query: AskQuery):
    """SafeSpace-compatible endpoint returning response and tool_called.
    
    Compatible directly with SafeSpace AI frontend (frontend.py).
    """
    if not query.message or not query.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty",
        )

    t_id = query.thread_id or str(uuid.uuid4())
    result = send_message(
        user_text=query.message.strip(),
        thread_id=t_id,
        use_simulation=False,
    )

    tool_name = result.get("tool") or result.get("action", "direct_conversation")
    return AskResponse(
        response=result.get("response", ""),
        tool_called=str(tool_name),
    )


@app.post("/api/chat", response_model=ExtendedChatResponse)
def chat_extended(query: AskQuery):
    """Full-featured agent endpoint returning clinical sources, followups, and risk metadata."""
    if not query.message or not query.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty",
        )

    t_id = query.thread_id or str(uuid.uuid4())
    result = send_message(
        user_text=query.message.strip(),
        thread_id=t_id,
        use_simulation=False,
    )

    resp_text = result.get("response", "")
    followups = get_suggested_followups(resp_text)

    return ExtendedChatResponse(
        response=resp_text,
        tool_called=str(result.get("tool") or result.get("action", "direct_conversation")),
        risk_level=result.get("risk_level", "LOW"),
        detected_intent=result.get("intent"),
        detected_emotion=result.get("emotion"),
        sources=result.get("sources", []),
        followups=followups,
        escalation=result.get("escalation"),
        thread_id=t_id,
    )


@app.post("/api/therapists")
def find_therapists_endpoint(query: TherapistQuery):
    """Dedicated endpoint to locate licensed clinical psychologists and clinics."""
    if not query.location or not query.location.strip():
        raise HTTPException(status_code=400, detail="Location query cannot be empty")
    results_md = find_nearby_therapists(query.location.strip())
    return {"location": query.location, "directory_markdown": results_md}


class EmergencyCallRequest(BaseModel):
    contact_number: Optional[str] = Field(None, description="Target phone number")
    message: Optional[str] = Field(None, description="Spoken voice alert message")
    risk_level: str = Field("HIGH", description="Risk level")


@app.post("/api/emergency/call")
def trigger_emergency_call_endpoint(req: EmergencyCallRequest):
    """Trigger automated high-risk emergency telephony voice call via Twilio."""
    from tools.emergency_dispatch import initiate_emergency_call, get_emergency_dispatch_history
    result = initiate_emergency_call(
        risk_level=req.risk_level,
        user_input_snippet=req.message or "High-risk crisis trigger",
        contact_number=req.contact_number,
        spoken_message=req.message,
    )
    return {
        "status": "success",
        "dispatch": result,
        "recent_history": get_emergency_dispatch_history(limit=5),
    }

