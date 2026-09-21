"""Tests for Dynamic Agentic Routing Matrix (Category Z & H)."""

import pytest
from agent.planner import plan_action
from graph import build_mindly_graph


def test_routing_matrix_direct_response():
    """Verify greetings route to DIRECT_RESPONSE and NOT to RAG, Coping, or Grounding."""
    decision = plan_action(
        user_input="Hello Mindly, how are you today?",
        detected_intent="GENERAL_CONVERSATION",
        detected_emotion="neutral",
        risk_level="LOW",
    )
    assert decision["selected_action"] == "DIRECT_RESPONSE"


def test_routing_matrix_grounding():
    """Verify explicit grounding request routes to USE_GROUNDING_TOOL."""
    decision = plan_action(
        user_input="Can you guide me through a 5-4-3-2-1 exercise?",
        detected_intent="GROUNDING_REQUEST",
        detected_emotion="anxious",
        risk_level="LOW",
    )
    assert decision["selected_action"] == "USE_GROUNDING_TOOL"


def test_routing_matrix_coping():
    """Verify coping queries route to USE_COPING_TOOL."""
    decision = plan_action(
        user_input="What are some ways to manage study stress?",
        detected_intent="COPING_REQUEST",
        detected_emotion="stressed",
        risk_level="LOW",
    )
    assert decision["selected_action"] == "USE_COPING_TOOL"


def test_routing_matrix_rag():
    """Category G: Verify medical query routes to USE_RAG_TOOL."""
    decision = plan_action(
        user_input="What does the medical encyclopedia say about hypertension?",
        detected_intent="INFORMATION_REQUEST",
        detected_emotion="neutral",
        risk_level="LOW",
    )
    assert decision["selected_action"] == "USE_RAG_TOOL"


@pytest.mark.parametrize("casual_input", [
    "Hello Mindly.",
    "I am stressed.",
    "Help me calm down.",
    "Good morning.",
    "Thank you.",
    "I want to journal.",
    "Create a goal.",
])
def test_routing_matrix_rag_negative(casual_input):
    """Category H: Negative RAG test - verify casual, grounding, journal, or emotional queries DO NOT select RAG."""
    decision = plan_action(
        user_input=casual_input,
        detected_intent="GENERAL_CONVERSATION" if "Hello" in casual_input or "morning" in casual_input or "Thank" in casual_input
        else ("EMOTIONAL_SUPPORT" if "stressed" in casual_input
        else ("GROUNDING_REQUEST" if "calm" in casual_input
        else ("JOURNAL_REQUEST" if "journal" in casual_input
        else "GOAL_REQUEST"))),
        detected_emotion="neutral",
        risk_level="LOW",
    )
    assert decision["selected_action"] != "USE_RAG_TOOL", f"Input '{casual_input}' incorrectly selected RAG"


def test_routing_matrix_vision():
    """Verify image presence routes to USE_VISION_TOOL."""
    decision = plan_action(
        user_input="Look at my drawing.",
        detected_intent="GENERAL_CONVERSATION",
        detected_emotion="neutral",
        risk_level="LOW",
        image_present=True,
    )
    assert decision["selected_action"] == "USE_VISION_TOOL"


def test_routing_matrix_journal():
    """Verify journal command routes to USE_JOURNAL_TOOL."""
    decision = plan_action(
        user_input="/journal I felt much better after meditating.",
        detected_intent="JOURNAL_REQUEST",
        detected_emotion="calm",
        risk_level="LOW",
    )
    assert decision["selected_action"] == "USE_JOURNAL_TOOL"


def test_routing_matrix_goal():
    """Verify goal command routes to USE_GOAL_TOOL."""
    decision = plan_action(
        user_input="/goal Read 10 pages before bed",
        detected_intent="GOAL_REQUEST",
        detected_emotion="neutral",
        risk_level="LOW",
    )
    assert decision["selected_action"] == "USE_GOAL_TOOL"


def test_routing_matrix_crisis():
    """Verify HIGH risk unconditionally maps to CRISIS_RESPONSE."""
    decision = plan_action(
        user_input="I want to end my life.",
        detected_intent="CRISIS_RELATED",
        detected_emotion="distressed",
        risk_level="HIGH",
    )
    assert decision["selected_action"] == "CRISIS_RESPONSE"


@pytest.mark.parametrize("symptom_query", [
    "I am having pain in my stomach.",
    "My stomach hurts.",
    "Why does my stomach hurt?",
    "I have a headache.",
    "I have back pain.",
    "What causes nausea?",
    "What are symptoms of migraine?",
    "What is hypertension?",
])
def test_routing_matrix_physical_symptoms(symptom_query):
    """Sections 1 & 2: Verify physical symptom queries route to USE_RAG_TOOL (gale_retriever)."""
    decision = plan_action(
        user_input=symptom_query,
        detected_intent="HEALTH_INFORMATION",
        detected_emotion="neutral",
        risk_level="LOW",
    )
    assert decision["selected_action"] == "USE_RAG_TOOL"
    assert decision["tool_name"] == "gale_retriever"


def test_routing_matrix_stomach_with_stress_emotional_context():
    """Section 3: Emotional context must not override physical symptoms."""
    decision = plan_action(
        user_input="My stomach hurts because I am stressed.",
        detected_intent="HEALTH_INFORMATION",
        detected_emotion="stressed",
        risk_level="LOW",
    )
    assert decision["selected_action"] == "USE_RAG_TOOL"
    assert decision["tool_name"] == "gale_retriever"

    decision_b = plan_action(
        user_input="My stomach hurts badly and I am scared.",
        detected_intent="HEALTH_INFORMATION",
        detected_emotion="fear",
        risk_level="LOW",
    )
    assert decision_b["selected_action"] == "USE_RAG_TOOL"
    assert decision_b["tool_name"] == "gale_retriever"


def test_routing_matrix_explicit_grounding_preserved():
    """Section 4: Explicit grounding request must route to grounding tool, not medical RAG."""
    decision = plan_action(
        user_input="I am anxious and I want a breathing exercise.",
        detected_intent="GROUNDING_REQUEST",
        detected_emotion="anxious",
        risk_level="LOW",
    )
    assert decision["selected_action"] == "USE_GROUNDING_TOOL"
    assert decision["tool_name"] in ["box_breathing", "guided_reflection", "54321_grounding"]


def test_routing_matrix_idiom_not_rag():
    """Section 5 Test D: Metaphorical butterflies in stomach must not route to RAG."""
    decision = plan_action(
        user_input="I feel butterflies in my stomach because I am nervous.",
        detected_intent="EMOTIONAL_SUPPORT",
        detected_emotion="anxious",
        risk_level="LOW",
    )
    assert decision["selected_action"] != "USE_RAG_TOOL"
