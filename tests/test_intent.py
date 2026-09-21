"""Tests for Intent Classification across conversational, emotional, RAG, and tool domains."""

import pytest
from agent.intent import classify_intent


@pytest.mark.parametrize("phrase", [
    "Hello Mindly",
    "Hi",
    "What is your name?",
    "Tell me something about yourself.",
    "Thank you",
    "Good morning",
    "How are you doing today?",
])
def test_general_conversation_intent(phrase):
    """Category C: Verify conversational greetings and identity queries map to GENERAL_CONVERSATION."""
    intent = classify_intent(phrase)
    assert intent == "GENERAL_CONVERSATION"


@pytest.mark.parametrize("phrase", [
    "I am stressed about college.",
    "I feel overwhelmed.",
    "I feel sad today.",
    "I am anxious about my exam.",
    "I am frustrated with my studies.",
    "I don't know what to do.",
])
def test_emotional_support_intent(phrase):
    """Category D: Verify distress and emotional sharing map to EMOTIONAL_SUPPORT."""
    intent = classify_intent(phrase)
    assert intent in ["EMOTIONAL_SUPPORT", "COPING_REQUEST", "GROUNDING_REQUEST"]


@pytest.mark.parametrize("phrase", [
    "Help me calm down.",
    "I feel anxious, can you guide me through a breathing exercise?",
    "Give me a grounding exercise.",
    "Can you help me breathe?",
    "I need a 5-4-3-2-1 exercise.",
])
def test_grounding_request_intent(phrase):
    """Category E: Verify grounding requests map to GROUNDING_REQUEST."""
    intent = classify_intent(phrase)
    assert intent == "GROUNDING_REQUEST"


@pytest.mark.parametrize("phrase", [
    "What can I do to manage my stress?",
    "Give me ways to handle exam stress.",
    "How can I cope with feeling overwhelmed?",
    "What should I do when I feel emotionally exhausted?",
])
def test_coping_request_intent(phrase):
    """Category F: Verify coping strategy queries map to COPING_REQUEST."""
    intent = classify_intent(phrase)
    assert intent == "COPING_REQUEST"


@pytest.mark.parametrize("phrase", [
    "What is hypertension?",
    "What is migraine?",
    "Explain insomnia and sleep hygiene.",
    "What does medical literature say about anxiety?",
    "What are clinical causes of blood pressure spikes?",
])
def test_information_medical_intent(phrase):
    """Category G: Verify medical and physiological queries map to HEALTH_INFORMATION (or alias INFORMATION_REQUEST)."""
    intent = classify_intent(phrase)
    assert intent in ["HEALTH_INFORMATION", "INFORMATION_REQUEST"]


@pytest.mark.parametrize("phrase", [
    "I have stomach pain.",
    "My stomach hurts.",
    "Why does my stomach hurt?",
    "I have a headache.",
    "I have back pain.",
    "What causes nausea?",
    "What are symptoms of migraine?",
    "What is hypertension?",
    "I am having pain in my stomach.",
])
def test_health_information_physical_symptoms(phrase):
    """Section 1: Verify physical health symptoms map to HEALTH_INFORMATION."""
    intent = classify_intent(phrase)
    assert intent == "HEALTH_INFORMATION"


def test_distinguish_physical_from_emotional_intents():
    """Section 5: Verify distinction between physical symptoms, urgent/scared context, grounding, and idioms."""
    # Test A
    assert classify_intent("I am having pain in my stomach.") == "HEALTH_INFORMATION"
    # Test B: physical symptom with fear/emotional context
    assert classify_intent("My stomach hurts badly and I am scared.") == "HEALTH_INFORMATION"
    # Test C: explicit anxiety & calming down
    assert classify_intent("I feel anxious and need help calming down.") in ["GROUNDING_REQUEST", "COPING_REQUEST"]
    # Test D: idiomatic emotional metaphor ("butterflies in my stomach")
    assert classify_intent("I feel butterflies in my stomach because I am nervous.") == "EMOTIONAL_SUPPORT"


@pytest.mark.parametrize("phrase", [
    "Journal this: Today was productive.",
    "/journal I felt calm during my evening walk.",
    "I want to write in my journal.",
    "Show my journal entries.",
])
def test_journal_request_intent(phrase):
    """Verify journaling actions map to JOURNAL_REQUEST."""
    intent = classify_intent(phrase)
    assert intent == "JOURNAL_REQUEST"


@pytest.mark.parametrize("phrase", [
    "Create a goal to study for two hours daily.",
    "/goal Drink 2 liters of water",
    "Show my goals.",
    "Mark my study goal as completed.",
])
def test_goal_request_intent(phrase):
    """Verify goal management actions map to GOAL_REQUEST."""
    intent = classify_intent(phrase)
    assert intent == "GOAL_REQUEST"


@pytest.mark.parametrize("phrase", [
    "Remember that I prefer being called Alex.",
    "What is my preferred name?",
    "What did I ask you to remember?",
])
def test_memory_request_intent(phrase):
    """Verify personal memory storage/retrieval actions map to MEMORY_REQUEST or GENERAL_CONVERSATION."""
    intent = classify_intent(phrase)
    assert intent in ["MEMORY_REQUEST", "GENERAL_CONVERSATION"]
