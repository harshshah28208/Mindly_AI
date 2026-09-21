"""Tests for Edge Cases, Stress Inputs, and Concurrency (Categories AC and AG)."""

import pytest
from graph import build_mindly_graph


@pytest.mark.parametrize("edge_input", [
    "",
    "    ",
    "\n\t  \n",
    "a",
    "?",
    "!@#$%^&*()_+`~|}{[]:;?><,./",
    "🌿🌬️🧘‍♂️🧠💫❤️✨",
    "1234567890 99999",
    "https://example.com/articles/mental-health?id=123&track=true",
    "DROP TABLE memories; SELECT * FROM users;",
    "<script>alert('xss')</script>",
])
def test_ac_edge_inputs_do_not_crash_graph(simulated_graph, edge_input):
    """Category AC: Verify various malformed, empty, symbolic, or injection inputs do not crash the StateGraph."""
    config = {"configurable": {"thread_id": "test_edge_thread"}}
    state = simulated_graph.invoke({
        "user_input": edge_input,
        "input_type": "text",
        "messages": [],
    }, config=config)
    assert isinstance(state, dict)
    assert state.get("final_response") is not None
    assert len(state.get("final_response")) > 0


def test_ac_very_long_input_handling(simulated_graph):
    """Category AC: Verify input with 5,000+ characters executes safely without memory exhaustion."""
    long_text = "I have been feeling stressed and writing about it. " * 150  # ~7500 chars
    config = {"configurable": {"thread_id": "test_long_input_thread"}}
    state = simulated_graph.invoke({
        "user_input": long_text,
        "input_type": "text",
        "messages": [],
    }, config=config)
    assert state.get("final_response") is not None


def test_ag_repeated_interactions_no_leakage(simulated_graph):
    """Category AG: Run 15 consecutive varied turns on a thread and verify clean execution."""
    config = {"configurable": {"thread_id": "test_repeated_thread"}}
    prompts = [
        "Hello Mindly",
        "I feel stressed about my exams.",
        "Give me a quick breathing exercise.",
        "What can I do to sleep better?",
        "What does medical literature say about insomnia?",
        "Thank you Mindly",
        "/journal Felt better after studying.",
        "/goal Drink water regularly",
        "Remember that I like mornings.",
        "What time of day do I like?",
        "Hello again",
        "Give me 5-4-3-2-1 grounding.",
        "What is hypertension?",
        "Goodnight Mindly",
        "See you tomorrow",
    ]
    for p in prompts:
        state = simulated_graph.invoke({
            "user_input": p,
            "input_type": "text",
            "messages": [],
        }, config=config)
        assert state.get("final_response") is not None
        assert state.get("selected_action") is not None
