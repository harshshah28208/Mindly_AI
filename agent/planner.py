"""Agentic planner and hierarchical decision-making orchestrator for Mindly Lab 3.

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
- PRIORITY 9: GENERAL CONVERSATION

Demonstrates dynamic multimodal agentic behavior:
UNDERSTAND -> ANALYZE -> DECIDE -> PLAN -> SELECT TOOL -> EXECUTE -> EVALUATE
"""

import re
from typing import Any, Dict, Optional
from agent.intent import is_physical_health_query
from agent.safety import assess_safety, is_educational_or_third_person, screen_urgent_medical


def plan_action(
    user_input: str,
    detected_intent: str,
    detected_emotion: str,
    risk_level: str,
    safety_status: Optional[str] = None,
    health_priority: Optional[str] = None,
    input_type: str = "text",
    image_present: bool = False,
    previous_action: Optional[str] = None,
    evaluation_result: Optional[str] = None,
    retry_count: int = 0,
) -> Dict[str, str]:
    """Determine the optimal agentic action and tool for Mindly Lab 3 adhering strictly to priorities 0-9.

    Returns:
        Dict containing:
            - selected_action: 'DIRECT_RESPONSE' | 'USE_GROUNDING_TOOL' | 'USE_COPING_TOOL' |
                               'USE_RAG_TOOL' | 'USE_VISION_TOOL' | 'USE_JOURNAL_TOOL' |
                               'USE_GOAL_TOOL' | 'USE_MEMORY_TOOL' | 'CRISIS_RESPONSE'
            - tool_name: Specific sub-tool identifier
            - reasoning_summary: Safe, high-level non-private explanation
    """
    clean_input = (user_input or "").strip()
    lower = clean_input.lower()

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
    safety = assess_safety(clean_input)
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
        if previous_action == "USE_RAG_TOOL":
            return {
                "selected_action": "USE_COPING_TOOL",
                "tool_name": "coping_strategy",
                "reasoning_summary": "Re-evaluating user need: switching from knowledge retrieval to practical coping support.",
            }
        elif previous_action == "USE_GROUNDING_TOOL":
            return {
                "selected_action": "USE_COPING_TOOL",
                "tool_name": "coping_strategy",
                "reasoning_summary": "Re-evaluating user need: switching from grounding exercise to adaptive coping strategies.",
            }
        elif previous_action == "USE_COPING_TOOL":
            return {
                "selected_action": "DIRECT_RESPONSE",
                "tool_name": "direct_conversation",
                "reasoning_summary": "Re-evaluating user need: transitioning to direct compassionate listening.",
            }
        else:
            return {
                "selected_action": "USE_GROUNDING_TOOL",
                "tool_name": "guided_reflection",
                "reasoning_summary": "Re-evaluating user need: providing gentle mindful reflection.",
            }

    # --------------------------------------------------------------------------
    # PRIORITY 2: URGENT MEDICAL / HEALTH
    # --------------------------------------------------------------------------
    urgent_med = screen_urgent_medical(clean_input)
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

    # --------------------------------------------------------------------------
    # PRIORITY 3: GENERAL HEALTH INFORMATION (Gale RAG)
    # --------------------------------------------------------------------------
    if (
        detected_intent in ["HEALTH_INFORMATION", "INFORMATION_REQUEST"]
        or health_priority == "ROUTINE"
        or is_physical_health_query(clean_input)
        or any(
            k in lower for k in [
                "hypertension", "migraine", "symptoms of migraine", "symptoms of hypertension",
                "blood pressure", "headache symptoms", "what is insomnia", "what is dehydration",
                "causes of back pain", "causes of insomnia", "throat hurt", "coughing for three days",
                "iron deficiency", "symptoms associated with"
            ]
        )
    ):
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
    # PRIORITY 5: JOURNAL REQUEST
    # --------------------------------------------------------------------------
    if detected_intent == "JOURNAL_REQUEST" or any(k in lower for k in ["/journal", "journal this", "save to journal", "journal entry", "make a journal entry", "write in my journal", "write this in my journal", "show my journal", "show my recent journal entries", "open my journal"]):
        return {
            "selected_action": "USE_JOURNAL_TOOL",
            "tool_name": "journal_recorder",
            "reasoning_summary": "Priority 5 Journal: identified journaling instruction; routed to wellness reflection recorder.",
        }

    # --------------------------------------------------------------------------
    # PRIORITY 6: GOAL / PLANNING REQUEST
    # --------------------------------------------------------------------------
    if detected_intent == "GOAL_REQUEST" or any(k in lower for k in [
        "/goal", "create a goal", "create goal", "set a goal", "add a goal",
        "view goals", "list goals", "my goals", "complete goal", "mark goal",
        "mark my study goal", "mark my goal complete", "goal as completed", "show my goals", "help me set a study goal"
    ]):
        return {
            "selected_action": "USE_GOAL_TOOL",
            "tool_name": "goal_tracker",
            "reasoning_summary": "Priority 6 Goal: identified goal planning request; routed to habits and goals tracker.",
        }

    if detected_intent == "PLANNING_REQUEST" or any(k in lower for k in [
        "study plan", "study schedule", "plan my week", "plan my study",
        "routine", "time management", "organize my day", "plan my exam preparation",
        "make a plan for my exam", "help me plan my study schedule"
    ]):
        return {
            "selected_action": "USE_COPING_TOOL",
            "tool_name": "coping_strategy",
            "reasoning_summary": "Priority 6 Planning: identified study planning or routine request; routed to structured planning tool.",
        }

    # Long-term memory vault requests
    if detected_intent == "MEMORY_REQUEST" or any(k in lower for k in ["remember that", "what name should you call me", "what should you call me", "view memory"]):
        return {
            "selected_action": "USE_MEMORY_TOOL",
            "tool_name": "memory_vault",
            "reasoning_summary": "Identified persistent memory instruction; routed to memory vault.",
        }

    # --------------------------------------------------------------------------
    # PRIORITY 7: EXPLICIT GROUNDING REQUEST
    # --------------------------------------------------------------------------
    if detected_intent == "GROUNDING_REQUEST" or any(k in lower for k in [
        "help me calm down", "calm me down", "breathing exercise", "breathwork",
        "box breathing", "help me breathe", "5-4-3-2-1", "senses grounding",
        "guide me through breathing", "guide me through 5-4-3-2-1", "need a grounding exercise",
        "give me the 5-4-3-2-1", "want a breathing exercise"
    ]):
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
    if detected_intent == "COPING_REQUEST" or any(k in lower for k in [
        "how can i deal with", "how do i deal with", "deal with stress", "deal with exam",
        "coping strategies", "coping strategy", "give me coping strategies", "how can i handle",
        "how do i handle", "ways to manage", "how to cope", "manage stress", "manage college pressure",
        "give me some ways to manage", "overwhelmed with college work"
    ]):
        return {
            "selected_action": "USE_COPING_TOOL",
            "tool_name": "coping_strategy",
            "reasoning_summary": "Priority 8 Coping: identified request for actionable coping methods; selected adaptive coping tool.",
        }

    if detected_intent == "EMOTIONAL_SUPPORT":
        # Context 1: Academic or college workload stress -> Coping tool
        if any(w in lower for w in ["college", "workload", "study", "deadline", "pressure"]):
            return {
                "selected_action": "USE_COPING_TOOL",
                "tool_name": "coping_strategy",
                "reasoning_summary": "Priority 8 Coping: detected academic or college stress; selected structured coping tool.",
            }
        # Context 2: Acute emotional anxiety or somatic exam distress -> Grounding tool
        if detected_emotion in ["anxious", "overwhelmed"] or any(w in lower for w in ["very anxious", "anxiety", "panic"]):
            return {
                "selected_action": "USE_GROUNDING_TOOL",
                "tool_name": "box_breathing" if "exam" in lower else "guided_reflection",
                "reasoning_summary": "Priority 8 Emotional Support: detected anxiety or overwhelm; selected calming exercise.",
            }
        # Context 3: General emotional sharing (sadness, loneliness, bad day) -> Direct empathetic listening
        return {
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "direct_conversation",
            "reasoning_summary": "Priority 8 Emotional Support: identified general emotional sharing; selected open empathetic listening.",
        }

    # --------------------------------------------------------------------------
    # PRIORITY 9: GENERAL CONVERSATION
    # --------------------------------------------------------------------------
    if detected_intent == "GENERAL_CONVERSATION" or any(
        re.search(rf"\b{re.escape(phrase)}\b", lower) for phrase in [
            "hello", "hi", "hey", "who are you", "what is your name", "what can you do",
            "what is my", "what's my", "who am i", "do you remember", "good morning",
            "tell me a joke", "tell me something interesting"
        ]
    ):
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
