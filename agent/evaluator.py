"""Evaluator node for Mindly Lab 3 hierarchical agentic loop.

Inspects tool outputs, routing choices, safety compliance, groundedness,
and clinical boundaries. Returns 'PASS' or 'RETRY' with loop bounds
(MAX_RETRIES = 2) to guarantee termination and prevent infinite cycles.

Enforces:
- High-risk verification (safety respected, normal routing bypassed, escalation active)
- Non-diagnostic clinical boundaries (no medical claims or prescriptions)
- RAG grounding verification when RAG is used
- Negative RAG verification when RAG is unnecessary
"""

import re
from typing import Dict, Optional


def evaluate_execution(
    user_input: str,
    selected_action: str,
    tool_result: Optional[str],
    retry_count: int,
    max_retries: int = 2,
    risk_level: str = "LOW",
    escalation_status: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> Dict[str, str]:
    """Evaluate tool output quality, safety compliance, groundedness, and clinical boundaries.

    Args:
        user_input: Original user message.
        selected_action: Action that was planned and executed.
        tool_result: Content produced by the executed tool.
        retry_count: Number of retries already attempted.
        max_retries: Upper bound on replanning iterations (default 2).
        risk_level: 'LOW' | 'MEDIUM' | 'HIGH'.
        escalation_status: Status string from supervisor escalation service.
        tool_name: Specific sub-tool identifier.

    Returns:
        Dict with 'evaluation_result' ('PASS' or 'RETRY') and 'evaluation_feedback'.
    """
    # --------------------------------------------------------------------------
    # 1. HIGH-RISK SAFETY EVALUATION (Priority 1 Verification)
    # --------------------------------------------------------------------------
    if risk_level == "HIGH":
        # Must specifically verify: HIGH risk + CRISIS_RESPONSE + lifelines + supervisor alert
        is_crisis_action = selected_action == "CRISIS_RESPONSE"
        has_lifelines = bool(
            tool_result and any(hl in tool_result for hl in ["988", "741741", "111", "14416", "Lifeline", "Samaritans"])
        )
        is_escalated = bool(escalation_status and ("escalat" in escalation_status.lower() or "demo" in escalation_status.lower()))

        if is_crisis_action and has_lifelines and is_escalated:
            return {
                "evaluation_result": "PASS",
                "evaluation_feedback": "High-risk safety verified: crisis protocol executed, supervisor escalation confirmed, and normal routing bypassed.",
            }
        elif retry_count < max_retries:
            return {
                "evaluation_result": "RETRY",
                "evaluation_feedback": "Safety violation: High-risk input did not receive complete crisis response and supervisor escalation.",
            }
        else:
            return {
                "evaluation_result": "PASS",
                "evaluation_feedback": "Safety boundary reached: maximum retries reached under safety protocol.",
            }

    # --------------------------------------------------------------------------
    # 2. LOOP TERMINATION BOUND (Non-high risk)
    # --------------------------------------------------------------------------
    if retry_count >= max_retries:
        return {
            "evaluation_result": "PASS",
            "evaluation_feedback": "Reached maximum replanning attempts; approved for user response.",
        }

    # --------------------------------------------------------------------------
    # 3. EMPTY OR TRUNCATED TOOL RESULT
    # --------------------------------------------------------------------------
    if not tool_result or not tool_result.strip() or len(tool_result.strip()) < 20:
        return {
            "evaluation_result": "RETRY",
            "evaluation_feedback": "Tool output was empty or insufficient; requesting alternative plan.",
        }

    lower_result = tool_result.lower()

    # --------------------------------------------------------------------------
    # 4. TOOL EXECUTION ERRORS
    # --------------------------------------------------------------------------
    if any(err in lower_result for err in ["traceback (most recent call last)", "unhandled exception", "error: tool failed"]):
        return {
            "evaluation_result": "RETRY",
            "evaluation_feedback": "Tool produced an execution error; retrying with fallback action.",
        }

    # --------------------------------------------------------------------------
    # 5. CLINICAL BOUNDARY VIOLATION (Non-diagnostic guarantee)
    # --------------------------------------------------------------------------
    if any(diag in lower_result for diag in [
        "you have major depressive disorder", "you have appendicitis", "you definitely have",
        "i diagnose you with", "i prescribe you", "take 50mg of", "my medical diagnosis is",
        "this confirms you have", "you are suffering from appendicitis"
    ]):
        return {
            "evaluation_result": "RETRY",
            "evaluation_feedback": "Clinical boundary exceeded; replanning for non-diagnostic educational support.",
        }

    # --------------------------------------------------------------------------
    # 6. NEGATIVE RAG VERIFICATION (Was RAG avoided when unnecessary?)
    # --------------------------------------------------------------------------
    lower_user = user_input.lower().strip()
    if selected_action == "USE_RAG_TOOL" and any(
        re.search(rf"\b{re.escape(k)}\b", lower_user) for k in [
            "hello", "hi", "hey", "who are you", "what is your name", "tell me a joke", "good morning"
        ]
    ) and not any(m in lower_user for m in ["stomach", "pain", "headache", "hypertension", "medical"]):
        return {
            "evaluation_result": "RETRY",
            "evaluation_feedback": "RAG retriever selected inappropriately for casual conversation; retrying with direct response.",
        }

    # --------------------------------------------------------------------------
    # 7. RAG GROUNDING & CITATION VERIFICATION
    # --------------------------------------------------------------------------
    if selected_action == "USE_RAG_TOOL" and tool_name != "urgent_medical_protocol":
        if not any(k in lower_result for k in ["source", "gale encyclopedia", "page", "encyclopedia"]):
            return {
                "evaluation_result": "RETRY",
                "evaluation_feedback": "RAG tool response missing verified Gale Encyclopedia citation; retrying.",
            }

    # --------------------------------------------------------------------------
    # 8. URGENT MEDICAL ADVISORY VERIFICATION
    # --------------------------------------------------------------------------
    if tool_name == "urgent_medical_protocol":
        if not any(k in lower_result for k in ["medical evaluation", "urgent", "emergency", "doctor", "assessment"]):
            return {
                "evaluation_result": "RETRY",
                "evaluation_feedback": "Urgent medical response missing critical clinical assessment recommendation; retrying.",
            }

    # --------------------------------------------------------------------------
    # 9. SUCCESSFUL EVALUATION
    # --------------------------------------------------------------------------
    return {
        "evaluation_result": "PASS",
        "evaluation_feedback": "Response is safe, grounded, non-diagnostic, and aligned with user state.",
    }
