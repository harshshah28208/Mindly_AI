"""Tests for Replanning and Retry Limit Bounding (Categories X and Y)."""

import pytest
from agent.planner import plan_action
from agent.evaluator import evaluate_execution
from config.settings import settings


def test_x_planner_switches_action_on_retry():
    """Category X: Verify planner selects an alternative action when previous action resulted in RETRY."""
    decision = plan_action(
        user_input="I am feeling very anxious and stressed.",
        detected_intent="EMOTIONAL_SUPPORT",
        detected_emotion="anxious",
        risk_level="LOW",
        previous_action="USE_GROUNDING_TOOL",
        evaluation_result="RETRY",
        retry_count=1,
    )
    # The planner should dynamically choose an alternative rather than repeating the failing action
    assert decision["selected_action"] != "USE_GROUNDING_TOOL"
    assert decision["selected_action"] in ["USE_COPING_TOOL", "DIRECT_RESPONSE"]


def test_y_retry_limit_forces_pass():
    """Category Y: Verify evaluator forces PASS once retry_count reaches max_retries limit, preventing infinite loops."""
    max_retries = settings.max_eval_retries
    res = evaluate_execution(
        user_input="Help me with anxiety.",
        selected_action="USE_GROUNDING_TOOL",
        tool_result="Error: tool failed - simulated network partition",
        retry_count=max_retries,
        max_retries=max_retries,
    )
    # Even though tool_result is an error, retry_count >= max_retries MUST force PASS
    assert res["evaluation_result"] == "PASS"


def test_y_graph_execution_terminates_under_failure(simulated_graph, monkeypatch):
    """Category Y: Verify graph execution terminates safely without infinite recursion under repeated failures."""
    import tools.grounding

    def broken_tool(*args, **kwargs):
        raise RuntimeError("Persistent crash simulated")

    monkeypatch.setattr(tools.grounding, "interactive_grounding_tool", broken_tool)

    config = {"configurable": {"thread_id": "test_retry_bound_thread"}}
    state = simulated_graph.invoke({
        "user_input": "Give me a grounding exercise.",
        "input_type": "text",
        "messages": [],
    }, config=config)

    assert state.get("final_response") is not None
    # Retry count must not exceed settings.max_eval_retries
    assert state.get("retry_count", 0) <= settings.max_eval_retries
