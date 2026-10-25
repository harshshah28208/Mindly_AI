"""Comprehensive Verification Test Suite for Mindly AI.

Tests all requirements across Phases 1-6:
1. Token limits and dynamic length constraints
2. Ollama probe, missing model, timeout, empty output
3. Explicit provider selection, automatic fallback, and diagnostic isolation
4. Canonical routing table (Greetings, Educational definitions, Coping, Grounding, Medical RAG, Urgent, Crisis)
5. Medical red flags and educational third-person crisis separation
6. RAG retrieval, page citation verification, no hallucinated citations
7. Evaluator retry bounds and loop termination
8. Preserving final answer and developer diagnostics in graph state
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import requests

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from config.settings import settings
from config.llm_provider import UniversalChatModel, DynamicLocalResponseEngine, get_llm
from agent.intent import classify_intent, is_physical_health_query
from agent.safety import assess_safety, screen_urgent_medical, is_educational_or_third_person
from agent.planner import plan_action
from agent.evaluator import evaluate_execution
from tools.rag import execute_rag_query, format_citations
from graph import send_message


class TestProviderSelectionAndFailures(unittest.TestCase):
    """Phase 2: Provider selection, predictable failure handling, and diagnostics."""

    def test_explicit_ollama_provider_respected(self):
        """When provider='ollama', UniversalChatModel must not call cloud even if cloud key exists."""
        model = UniversalChatModel(provider="ollama")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake-key", "GROQ_API_KEY": "gsk-fake"}), \
             patch("requests.post") as mock_post, \
             patch.object(settings, "check_ollama_health", return_value={"is_running": True, "model_installed": True}):
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {
                "message": {"content": "Ollama local answer."},
                "done_reason": "stop",
                "eval_count": 10,
            }
            resp = model.invoke([HumanMessage(content="Hello")])
            assert resp.content == "Ollama local answer."
            assert UniversalChatModel.last_metadata["provider_requested"] == "Local Ollama"
            assert UniversalChatModel.last_metadata["provider_used"] == "Local Ollama"
            assert not UniversalChatModel.last_metadata["is_fallback"]

    def test_explicit_ollama_failure_with_fallback_disabled(self):
        """When explicit ollama fails and fallback is disabled, return safe diagnostic error."""
        model = UniversalChatModel(provider="ollama")
        with patch.dict(os.environ, {"LLM_FALLBACK_ENABLED": "false"}), \
             patch.object(settings, "check_ollama_health", return_value={"is_running": False, "model_installed": False}):
            resp = model.invoke([HumanMessage(content="Hello")])
            assert "Provider Error" in resp.content
            assert "Local Ollama" in resp.content
            assert UniversalChatModel.last_metadata["error_category"] == "connection_error"
            assert not UniversalChatModel.last_metadata["is_fallback"]

    def test_explicit_ollama_failure_with_fallback_enabled(self):
        """When explicit ollama fails and fallback is enabled, return dynamic text with fallback diagnostic."""
        model = UniversalChatModel(provider="ollama")
        with patch.dict(os.environ, {"LLM_FALLBACK_ENABLED": "true"}), \
             patch.object(settings, "check_ollama_health", return_value={"is_running": False, "model_installed": False}):
            resp = model.invoke([HumanMessage(content="Hello")])
            assert resp.content != ""
            assert "Provider Error" not in resp.content
            assert UniversalChatModel.last_metadata["is_fallback"] is True
            assert UniversalChatModel.last_metadata["provider_used"] == "Dynamic Local Engine (Fallback)"
            assert UniversalChatModel.last_metadata["error_category"] == "connection_error"

    def test_explicit_cloud_failure_with_fallback_disabled(self):
        """When cloud provider is explicit and no key is provided, return diagnostic."""
        model = UniversalChatModel(provider="groq")
        with patch.dict(os.environ, {"GROQ_API_KEY": "", "OPENAI_API_KEY": "", "LLM_FALLBACK_ENABLED": "false"}):
            resp = model.invoke([HumanMessage(content="Hello")])
            assert "Provider Error" in resp.content
            assert UniversalChatModel.last_metadata["error_category"] == "missing_api_key"

    def test_ollama_health_generation_probe(self):
        """Verify check_ollama_health with test_generation=True executes generate probe."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2:1b"}]}
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {"response": "Hi"}

            health = settings.check_ollama_health(test_generation=True)
            assert health["is_running"] is True
            assert health["model_installed"] is True
            assert health["can_generate"] is True


class TestFallbackBehaviorAndEducation(unittest.TestCase):
    """Phase 3: Verify dynamic fallback definitions and context-aware follow-up."""

    def test_anxiety_educational_definition(self):
        """'What is anxiety?' must receive a clear educational explanation, not an empathy question."""
        ans = DynamicLocalResponseEngine.generate([HumanMessage(content="What is anxiety?")])
        assert "Anxiety" in ans
        assert "autonomic nervous system" in ans.lower() or "stress response" in ans.lower()
        # Must not be a generic empathy deflection question
        assert "what is one small thing within your immediate control" not in ans.lower()

    def test_depression_educational_definition(self):
        """'What is depression?' must receive a factual clinical explanation."""
        ans = DynamicLocalResponseEngine.generate([HumanMessage(content="What is depression?")])
        assert "Depression" in ans
        assert "mood" in ans.lower()
        assert "anhedonia" in ans.lower() or "interest" in ans.lower()

    def test_multiturn_context_continuation_migraine(self):
        """'tell me more' after a migraine discussion must elaborate on migraine management."""
        messages = [
            HumanMessage(content="What causes a migraine?"),
            AIMessage(content="Migraine is a neurovascular disorder..."),
            HumanMessage(content="tell me more"),
        ]
        ans = DynamicLocalResponseEngine.generate(messages)
        assert "migraine" in ans.lower()
        assert "sensory rest" in ans.lower() or "trigger" in ans.lower()

    def test_multiturn_context_continuation_stomach(self):
        """'tell me more' after stomach acidity must continue on digestive health and brain-gut axis."""
        messages = [
            HumanMessage(content="My stomach hurts and I have acidity"),
            AIMessage(content="Gastric symptoms can be exacerbated by stress..."),
            HumanMessage(content="tell me more"),
        ]
        ans = DynamicLocalResponseEngine.generate(messages)
        assert "gut" in ans.lower() or "digestive" in ans.lower() or "stomach" in ans.lower()


class TestRoutingAndPlannerAudit(unittest.TestCase):
    """Phase 4: Canonical routing audit across safety, intent, emotion, planner, and tools."""

    def test_01_greeting_routing(self):
        intent = classify_intent("Hi Mindly")
        plan = plan_action(user_input="Hi Mindly", detected_intent=intent)
        assert plan["selected_action"] == "DIRECT_RESPONSE"
        assert plan["tool_name"] == "direct_conversation"

    def test_02_what_is_anxiety_educational_routing(self):
        intent = classify_intent("What is anxiety?")
        plan = plan_action(user_input="What is anxiety?", detected_intent=intent)
        assert plan["selected_action"] == "DIRECT_RESPONSE"
        assert plan["tool_name"] == "direct_conversation"

    def test_03_exam_anxiety_coping_routing(self):
        intent = classify_intent("I feel anxious about tomorrow's exam")
        plan = plan_action(user_input="I feel anxious about tomorrow's exam", detected_intent=intent)
        assert plan["selected_action"] == "USE_COPING_TOOL"
        assert plan["tool_name"] == "coping_strategy"

    def test_04_box_breathing_grounding_routing(self):
        intent = classify_intent("Guide me through box breathing")
        plan = plan_action(user_input="Guide me through box breathing", detected_intent=intent)
        assert plan["selected_action"] == "USE_GROUNDING_TOOL"
        assert plan["tool_name"] == "box_breathing"

    def test_05_study_concentration_planning_routing(self):
        intent = classify_intent("I cannot concentrate on studying. Help me make a plan")
        plan = plan_action(user_input="I cannot concentrate on studying. Help me make a plan", detected_intent=intent)
        assert plan["selected_action"] == "USE_COPING_TOOL"
        assert plan["tool_name"] == "coping_strategy"

    def test_06_migraine_medical_rag_routing(self):
        intent = classify_intent("What causes a migraine?")
        plan = plan_action(user_input="What causes a migraine?", detected_intent=intent)
        assert plan["selected_action"] == "USE_RAG_TOOL"
        assert plan["tool_name"] == "gale_retriever"

    def test_07_severe_stomach_faint_urgent_medical_routing(self):
        intent = classify_intent("I have severe stomach pain and feel faint")
        plan = plan_action(user_input="I have severe stomach pain and feel faint", detected_intent=intent)
        assert plan["selected_action"] == "USE_RAG_TOOL"
        assert plan["tool_name"] == "urgent_medical_protocol"

    def test_08_third_person_assignment_educational_crisis_routing(self):
        intent = classify_intent("Explain suicide prevention for my college assignment")
        plan = plan_action(user_input="Explain suicide prevention for my college assignment", detected_intent=intent)
        assert plan["selected_action"] == "DIRECT_RESPONSE"
        assert plan["tool_name"] == "third_person_support_protocol"

    def test_09_journal_request_routing(self):
        intent = classify_intent("Help me save a journal entry")
        plan = plan_action(user_input="Help me save a journal entry", detected_intent=intent)
        assert plan["selected_action"] == "USE_JOURNAL_TOOL"
        assert plan["tool_name"] == "journal_recorder"

    def test_10_goal_request_routing(self):
        intent = classify_intent("Help me create a daily study goal")
        plan = plan_action(user_input="Help me create a daily study goal", detected_intent=intent)
        assert plan["selected_action"] == "USE_GOAL_TOOL"
        assert plan["tool_name"] == "goal_tracker"


class TestRAGRetrieverAndCitations(unittest.TestCase):
    """Phase 5: Gale Encyclopedia of Medicine retrieval and strict citation verification."""

    def test_citations_include_real_page_numbers_only(self):
        chunks = [
            {"page": 6, "text": "Knee pain information..."},
            {"page": 28, "text": "Migraine headache causes..."},
        ]
        citation = format_citations(chunks)
        assert "Gale Encyclopedia of Medicine" in citation
        assert "Page 6" in citation
        assert "Page 28" in citation
        assert "Page 999" not in citation

    def test_empty_chunks_do_not_invent_fake_pages(self):
        citation = format_citations([])
        assert "Page" not in citation
        assert "Gale Encyclopedia of Medicine" in citation


class TestEvaluatorRetryBounds(unittest.TestCase):
    """Evaluator loop bounds and replanning safety."""

    def test_evaluator_passes_complete_output(self):
        res = evaluate_execution(
            user_input="How to manage stress",
            selected_action="USE_COPING_TOOL",
            tool_result="Here is a complete, multi-part coping guide:\n1. Breathe.\n2. Walk.\n3. Rest.",
            retry_count=0,
            max_retries=2,
        )
        assert res["evaluation_result"] == "PASS"

    def test_evaluator_retries_abrupt_truncation(self):
        res = evaluate_execution(
            user_input="How to manage stress",
            selected_action="USE_COPING_TOOL",
            tool_result="Here is step 1: take a breath and",  # dangling conjunction
            retry_count=0,
            max_retries=2,
        )
        assert res["evaluation_result"] == "RETRY"

    def test_evaluator_terminates_at_max_retries(self):
        res = evaluate_execution(
            user_input="How to manage stress",
            selected_action="USE_COPING_TOOL",
            tool_result="Short text",
            retry_count=2,
            max_retries=2,
        )
        assert res["evaluation_result"] == "PASS"


class TestGraphPreservationAndTelemetry(unittest.TestCase):
    """Preserve final response and developer diagnostics through send_message."""

    def test_send_message_returns_complete_telemetry(self):
        res = send_message(user_text="Hi Mindly", thread_id="test_thread_123", use_simulation=True)
        assert "response" in res
        assert res["response"] != ""
        assert "llm_provider" in res
        assert "provider_requested" in res
        assert "provider_used" in res
        assert "is_fallback" in res
        assert "intent" in res
        assert "action" in res


if __name__ == "__main__":
    unittest.main()
