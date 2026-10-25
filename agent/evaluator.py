"""Evaluator node for Mindly hierarchical agentic loop.

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
from typing import Any, Dict, Optional


def evaluate_execution(
    user_input: str,
    selected_action: str,
    tool_result: Optional[str],
    retry_count: int,
    max_retries: int = 2,
    risk_level: str = "LOW",
    escalation_status: Optional[str] = None,
    tool_name: Optional[str] = None,
    is_truncated: bool = False,
) -> Dict[str, Any]:
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
        is_truncated: Boolean flag indicating if LLM stopped due to token limit.

    Returns:
        Dict with 'evaluation_result' ('PASS' or 'RETRY'), 'evaluation_feedback', and 'is_truncated'.
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
    # 3. TRUNCATION & TOKEN-LIMIT INSPECTION (Phase 8 Requirement)
    # --------------------------------------------------------------------------
    if not tool_result or not tool_result.strip() or len(tool_result.strip()) < 20:
        return {
            "evaluation_result": "RETRY",
            "evaluation_feedback": "Tool output was empty or insufficient (< 20 characters); requesting alternative plan.",
            "is_truncated": True,
        }

    trimmed_result = tool_result.strip()
    has_abrupt_dangling_ending = bool(
        re.search(r"[,:;\-\(\[\{]\s*$", trimmed_result)
        or re.search(r"\b(?:and|or|the|with|to|in|of|that|because|step)\s*$", trimmed_result, re.IGNORECASE)
    )

    if is_truncated or has_abrupt_dangling_ending:
        return {
            "evaluation_result": "RETRY",
            "evaluation_feedback": "Generation stopped abruptly or reached token limit; retrying with expanded token budget and complete structure.",
            "is_truncated": True,
        }

    # Verify requested multi-step completeness (e.g. "give me 5 steps")
    steps_match = re.search(r"\b(\d+)\s+steps\b", user_input.lower())
    if steps_match:
        requested_count = int(steps_match.group(1))
        # Count numbered steps in output (e.g. "1. ", "Step 1", "1)")
        numbered_steps = len(re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]|Step\s+\d+)", trimmed_result))
        if requested_count > 1 and numbered_steps < requested_count:
            return {
                "evaluation_result": "RETRY",
                "evaluation_feedback": f"User requested {requested_count} steps, but only {numbered_steps} were generated; retrying to provide all requested steps.",
                "is_truncated": True,
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
