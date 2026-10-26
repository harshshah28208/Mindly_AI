"""Agentic planner and hierarchical decision-making orchestrator for Mindly.

Implements the explicit 10-level routing priority model (Priorities 0-9):
- PRIORITY 0: SYSTEM / INVALID INPUT
- PRIORITY 1: HIGH-RISK SAFETY (Bypasses normal routing)
- PRIORITY 2: URGENT MEDICAL / HEALTH
- PRIORITY 3: GENERAL HEALTH INFORMATION (Gale RAG)
- PRIORITY 4: CRISIS-RELATED BUT NON-IMMEDIATE (Educational / Third-Person)
- PRIORITY 5: JOURNAL REQUEST
- PRIORITY 6: GOAL / PLANNING REQUEST
- PRIORITY 7: EXPLICIT GROUNDING REQUEST
- PRIORITY 8: COPING / EMOTIONAL SUPPORT
- PRIORITY 9: GENERAL CONVERSATION & SCOPE BOUNDARY

Demonstrates dynamic multimodal agentic behavior:
UNDERSTAND -> ANALYZE -> DECIDE -> PLAN -> SELECT TOOL -> EXECUTE -> EVALUATE
"""

import re
from typing import Any, Dict, Optional
from agent.intent import is_physical_health_query, is_out_of_scope
from agent.safety import assess_safety, is_educational_or_third_person, screen_urgent_medical



def plan_action(
    user_input: str,
    detected_intent: Optional[str] = None,
    detected_emotion: str = "neutral",
    risk_level: str = "LOW",
    safety_status: Optional[str] = None,
    health_priority: Optional[str] = None,
    input_type: str = "text",
    image_present: bool = False,
    previous_action: Optional[str] = None,
    evaluation_result: Optional[str] = None,
    retry_count: int = 0,
    confidence: Optional[float] = None,
    needs_clarification: Optional[bool] = None,
) -> Dict[str, str]:
    """Determine the optimal agentic action and tool for Mindly adhering strictly to priorities 0-9.

    Returns:
        Dict containing:
            - selected_action: 'DIRECT_RESPONSE' | 'USE_GROUNDING_TOOL' | 'USE_COPING_TOOL' |
                               'USE_RAG_TOOL' | 'USE_VISION_TOOL' | 'USE_JOURNAL_TOOL' |
                               'USE_GOAL_TOOL' | 'USE_MEMORY_TOOL' | 'CRISIS_RESPONSE'
            - tool_name: Specific sub-tool identifier
            - reasoning_summary: Safe, high-level non-private explanation
    """
    clean_input = (user_input or "").strip()
    if detected_intent is None:
        from agent.intent import classify_intent
        detected_intent = classify_intent(clean_input)
    lower = clean_input.lower()
    safety = assess_safety(clean_input)
    urgent_med = screen_urgent_medical(clean_input)

    # --------------------------------------------------------------------------
    # PRIORITY 0: SYSTEM / INVALID INPUT
    # --------------------------------------------------------------------------
    if not clean_input and not image_present and input_type != "image":
        return {
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "direct_conversation",
            "reasoning_summary": "Priority 0 System: received empty or invalid input; prompted user for clarification.",
        }

    # --------------------------------------------------------------------------
    # PRIORITY 1: HIGH-RISK SAFETY (Immediate override of all normal routing)
    # --------------------------------------------------------------------------
    if risk_level == "HIGH" or safety["risk_level"] == "HIGH" or detected_intent == "CRISIS_RELATED":
        # Ensure educational or third-person is not accidentally caught as HIGH here
        if not (safety["safety_status"] == "EDUCATIONAL_OR_THIRD_PERSON" or is_educational_or_third_person(clean_input)):
            return {
                "selected_action": "CRISIS_RESPONSE",
                "tool_name": "crisis_support_protocol",
                "reasoning_summary": "High-risk safety indicator detected; normal routing bypassed.",
            }

    # --------------------------------------------------------------------------
    # HANDLE REPLANNING / RETRY FROM EVALUATOR (Bounded to non-high-risk loops)
    # --------------------------------------------------------------------------
    if evaluation_result == "RETRY" and retry_count > 0:
        if urgent_med["is_urgent"] or health_priority == "URGENT" or detected_intent in ["URGENT_HEALTH", "URGENT_MEDICAL"]:
            return {
                "selected_action": "USE_RAG_TOOL",
                "tool_name": "urgent_medical_protocol",
                "reasoning_summary": "Priority 2 Urgent Medical: acute symptoms detected; delivering immediate clinical evaluation guidance.",
            }
        elif is_physical_health_query(clean_input) or health_priority in ["ROUTINE"]:
            return {
                "selected_action": "USE_RAG_TOOL",
                "tool_name": "gale_retriever",
                "reasoning_summary": "Re-evaluating clinical query: ensuring comprehensive medical response from Gale Encyclopedia of Medicine.",
            }
        elif previous_action == "USE_COPING_TOOL":
            return {
                "selected_action": "USE_COPING_TOOL",
                "tool_name": "coping_strategy",
                "reasoning_summary": "Replanning coping strategy: ensuring complete, fully numbered actionable guidance within token budget.",
            }
        elif previous_action == "USE_GROUNDING_TOOL":
            return {
                "selected_action": "USE_GROUNDING_TOOL",
                "tool_name": "box_breathing" if "breath" in lower else "54321_grounding",
                "reasoning_summary": "Replanning grounding tool: ensuring all somatic and sensory exercise steps are fully provided.",
            }
        elif previous_action == "USE_RAG_TOOL":
            return {
                "selected_action": "USE_RAG_TOOL",
                "tool_name": "gale_retriever",
                "reasoning_summary": "Replanning clinical inquiry: generating full structured clinical guide.",
            }
        else:
            return {
                "selected_action": "DIRECT_RESPONSE",
                "tool_name": "direct_conversation",
                "reasoning_summary": "Re-evaluating conversational need: providing complete, direct attuned guidance.",
            }

    # --------------------------------------------------------------------------
    # PRIORITY 2: URGENT MEDICAL / HEALTH
    # --------------------------------------------------------------------------
    if detected_intent == "URGENT_HEALTH" or health_priority == "URGENT" or urgent_med["is_urgent"]:
        return {
            "selected_action": "USE_RAG_TOOL",
            "tool_name": "urgent_medical_protocol",
            "reasoning_summary": "Priority 2 Urgent Medical: detected potentially acute symptoms; provided immediate emergency guidance and clinical assessment advisory without diagnosis.",
        }

    # --------------------------------------------------------------------------
    # MULTIMODAL VISION INPUT (When safe)
    # --------------------------------------------------------------------------
    if image_present or input_type == "image":
        return {
            "selected_action": "USE_VISION_TOOL",
            "tool_name": "vision_analyzer",
            "reasoning_summary": "Detected image input; routed to multimodal vision analyzer.",
        }

    # Multimodal image query without image attached
    if any(q in lower for q in [
        "what is in this picture", "what is in this image", "what's in this picture",
        "describe this picture", "analyze this image", "what do you see in this picture",
        "what is this picture", "what is this image"
    ]) and not image_present and input_type != "image":
        return {
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "missing_image_advisory",
            "reasoning_summary": "User asked about an image or picture, but no image was attached.",
        }

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # PRIORITY 3: GENERAL HEALTH INFORMATION (Gale RAG)
    # --------------------------------------------------------------------------
    if detected_intent in ["HEALTH_INFORMATION", "INFORMATION_REQUEST"]:
        reason = "Priority 3 General Health: identified health knowledge inquiry; routed to Gale Encyclopedia of Medicine retriever."
        if any(w in lower for w in ["scared", "stress", "stressed", "anxious", "nervous"]) or detected_emotion in ["anxious", "stressed", "fear", "sad", "overwhelmed"]:
            reason = "Priority 3 General Health: identified physical symptom with emotional context; routed to Gale Encyclopedia of Medicine retriever with supportive framing."
        return {
            "selected_action": "USE_RAG_TOOL",
            "tool_name": "gale_retriever",
            "reasoning_summary": reason,
        }

    # --------------------------------------------------------------------------
    # PRIORITY 4: CRISIS-RELATED BUT NON-IMMEDIATE (Educational / Third-Person)
    # --------------------------------------------------------------------------
    if (
        safety_status == "EDUCATIONAL_OR_THIRD_PERSON"
        or is_educational_or_third_person(clean_input)
    ):
        return {
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "third_person_support_protocol",
            "reasoning_summary": "Priority 4 Non-immediate crisis context: identified educational inquiry or third-person support discussion; provided compassionate helpline resources without high-risk escalation.",
        }

    # --------------------------------------------------------------------------
    # AMBIGUOUS / UNDERSPECIFIED INPUT (Clarification protocol instead of arbitrary tool dispatch)
    # --------------------------------------------------------------------------
    if detected_intent == "CLARIFICATION_NEEDED" or needs_clarification:
        return {
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "clarification_protocol",
            "reasoning_summary": "Intent ambiguous or underspecified; offering gentle clarifying options instead of arbitrary tool dispatch.",
        }

    # --------------------------------------------------------------------------
    # DOMAIN BOUNDARY GUARDRAIL: OUT-OF-SCOPE / NON-MEDICAL / MATH
    # --------------------------------------------------------------------------
    if detected_intent == "OUT_OF_SCOPE":
        return {
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "scope_boundary_protocol",
            "reasoning_summary": "Domain Scope Boundary: query is outside clinical medical & wellness scope; enforcing non-medical boundary.",
        }

    # --------------------------------------------------------------------------
    # THERAPIST LOCATOR REQUEST (Inspired by SafeSpace AI)
    # --------------------------------------------------------------------------
    if detected_intent == "THERAPIST_REQUEST":
        return {
            "selected_action": "USE_THERAPIST_TOOL",
            "tool_name": "therapist_locator",
            "reasoning_summary": "Identified request for professional human counseling or local clinic; routed to therapist locator.",
        }

    # --------------------------------------------------------------------------
    # PRIORITY 5: JOURNAL REQUEST
    # --------------------------------------------------------------------------
    if detected_intent == "JOURNAL_REQUEST":
        return {
            "selected_action": "USE_JOURNAL_TOOL",
            "tool_name": "journal_recorder",
            "reasoning_summary": "Priority 5 Journal: identified journaling instruction; routed to wellness reflection recorder.",
        }

    # --------------------------------------------------------------------------
    # PRIORITY 6: GOAL / PLANNING REQUEST
    # --------------------------------------------------------------------------
    if detected_intent in ["GOAL_REQUEST", "PLANNING_REQUEST"]:
        # If user explicitly asks about goals, habits, or setting/tracking goals
        if "goal" in lower or "habit" in lower or lower.startswith("/goal"):
            return {
                "selected_action": "USE_GOAL_TOOL",
                "tool_name": "goal_tracker",
                "reasoning_summary": "Priority 6 Goal: identified goal or habit request; routed to habits and goals tracker.",
            }
        # If user asks for study planning, focus, or concentration help without goal tracker
        return {
            "selected_action": "USE_COPING_TOOL",
            "tool_name": "coping_strategy",
            "reasoning_summary": "Priority 6 Planning: identified study concentration and planning support; selected actionable coping tool.",
        }

    # Long-term memory vault requests
    if detected_intent == "MEMORY_REQUEST":
        return {
            "selected_action": "USE_MEMORY_TOOL",
            "tool_name": "memory_vault",
            "reasoning_summary": "Identified persistent memory instruction; routed to memory vault.",
        }

    # --------------------------------------------------------------------------
    # NEGATIVE CONSTRAINT: USER DOES NOT WANT AN EXERCISE / JUST LISTEN
    # --------------------------------------------------------------------------
    if bool(re.search(r"\b(?:don't|dont|do\s+not|no)\s+(?:want|need|give\s+me)\s+(?:an?\s+)?(?:exercise|grounding|breathing)\b", lower)) or "just listen" in lower:
        return {
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "direct_conversation",
            "reasoning_summary": "User explicitly requested listening without active somatic exercises; selected open empathetic listening.",
        }

    # --------------------------------------------------------------------------
    # EDUCATIONAL / FACTUAL DEFINITIONS OF MENTAL WELLNESS CONCEPTS
    # --------------------------------------------------------------------------
    if bool(re.search(r"^(?:what\s+(?:is|are)|define|explain|meaning\s+of|tell\s+me\s+about)\s+(?:anxiety|depression|stress|mental\s+health|burnout|panic)\b", lower)):
        return {
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "direct_conversation",
            "reasoning_summary": "Identified educational definition inquiry; selected direct educational explanation.",
        }

    # --------------------------------------------------------------------------
    # PRIORITY 7: EXPLICIT GROUNDING REQUEST
    # --------------------------------------------------------------------------
    if detected_intent == "GROUNDING_REQUEST":
        if detected_emotion == "anxious" or any(w in lower for w in ["panic", "racing", "breath", "breathe", "heart"]):
            tool = "box_breathing"
            summary = "Priority 7 Grounding: detected acute anxiety/panic; selected structured Box Breathing (4-4-4-4)."
        elif any(w in lower for w in ["5-4-3-2-1", "senses", "overwhelm", "drowning", "too much"]):
            tool = "54321_grounding"
            summary = "Priority 7 Grounding: detected sensory/cognitive overwhelm; selected 5-4-3-2-1 sensory grounding exercise."
        else:
            tool = "guided_reflection"
            summary = "Priority 7 Grounding: selected gentle guided mindful reflection."
        return {
            "selected_action": "USE_GROUNDING_TOOL",
            "tool_name": tool,
            "reasoning_summary": summary,
        }

    # --------------------------------------------------------------------------
    # PRIORITY 8: COPING / EMOTIONAL SUPPORT
    # --------------------------------------------------------------------------
    if detected_intent == "COPING_REQUEST":
        return {
            "selected_action": "USE_COPING_TOOL",
            "tool_name": "coping_strategy",
            "reasoning_summary": "Priority 8 Coping: identified request for actionable coping methods; selected adaptive coping tool.",
        }

    if detected_intent == "EMOTIONAL_SUPPORT":
        # Context 1: Academic, college, exam or viva stress -> Coping tool
        if any(w in lower for w in ["college", "workload", "study", "deadline", "pressure", "viva", "vivas", "exam", "exams", "interview", "presentation", "test", "assignment"]):
            return {
                "selected_action": "USE_COPING_TOOL",
                "tool_name": "coping_strategy",
                "reasoning_summary": "Priority 8 Coping: detected academic, exam, or viva stress; selected structured coping tool.",
            }
        # Context 2: Acute emotional anxiety or somatic distress -> Grounding tool
        if detected_emotion in ["anxious", "overwhelmed"] or any(w in lower for w in ["very anxious", "severe anxiety", "panic"]):
            return {
                "selected_action": "USE_GROUNDING_TOOL",
                "tool_name": "box_breathing" if "exam" in lower else "guided_reflection",
                "reasoning_summary": "Priority 8 Emotional Support: detected anxiety or overwhelm; selected calming exercise.",
            }
        # Context 3: General emotional sharing (sadness, loneliness, bad day, jealousy) -> Direct empathetic listening
        return {
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "direct_conversation",
            "reasoning_summary": "Priority 8 Emotional Support: identified general emotional sharing; selected open empathetic listening.",
        }

    # --------------------------------------------------------------------------
    # PRIORITY 9: GENERAL CONVERSATION
    # --------------------------------------------------------------------------
    if detected_intent == "GENERAL_CONVERSATION":
        return {
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "direct_conversation",
            "reasoning_summary": "Priority 9 General: identified casual conversation or greeting; selected direct response without unnecessary tools.",
        }

    # Safe default fallback
    return {
        "selected_action": "DIRECT_RESPONSE",
        "tool_name": "direct_conversation",
        "reasoning_summary": "Selected direct compassionate reflection.",
    }
