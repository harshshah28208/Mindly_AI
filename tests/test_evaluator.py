"""Tests for Evaluator Quality & Safety Inspection (Category W)."""

import pytest
from agent.evaluator import evaluate_execution


def test_w_evaluator_passes_substantive_output():
    """Verify substantive coping/grounding output passes evaluation."""
    res = evaluate_execution(
        user_input="I'm feeling anxious about exams.",
        selected_action="USE_GROUNDING_TOOL",
        tool_result="Here is the 4-4-4-4 box breathing technique:\n1. Inhale for 4 seconds...",
        retry_count=0,
    )
    assert res["evaluation_result"] == "PASS"


def test_w_evaluator_retries_empty_output():
    """Verify empty or whitespace-only tool output triggers RETRY."""
    res = evaluate_execution(
        user_input="Help me relax.",
        selected_action="USE_COPING_TOOL",
        tool_result="   ",
        retry_count=0,
    )
    assert res["evaluation_result"] == "RETRY"


def test_w_evaluator_retries_tool_error():
    """Verify execution error string triggers RETRY."""
    res = evaluate_execution(
        user_input="I need advice.",
        selected_action="USE_COPING_TOOL",
        tool_result="Error: tool failed - Ollama connection timeout",
        retry_count=0,
    )
    assert res["evaluation_result"] == "RETRY"


def test_w_evaluator_flags_rag_missing_citation():
    """Verify RAG tool result without source citation is flagged for RETRY when retries remain."""
    res = evaluate_execution(
        user_input="What is hypertension?",
        selected_action="USE_RAG_TOOL",
        tool_result="Hypertension is when your blood pressure is high. Take some medicine.",
        retry_count=0,
    )
    # Since citation is absent, it should issue RETRY
    assert res["evaluation_result"] == "RETRY"


def test_w_evaluator_passes_rag_with_citation():
    """Verify RAG tool result containing Gale Encyclopedia citation PASSES."""
    good_rag = (
        "Hypertension is high arterial blood pressure.\n\n"
        "📚 Sources Consulted:\n- Gale Encyclopedia of Medicine (Page 1)\n\n"
        "⚠️ Educational Disclaimer: For educational purposes only."
    )
    res = evaluate_execution(
        user_input="What is hypertension?",
        selected_action="USE_RAG_TOOL",
        tool_result=good_rag,
        retry_count=0,
    )
    assert res["evaluation_result"] == "PASS"
