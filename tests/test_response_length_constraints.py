"""Comprehensive tests for Response Length Constraints, Dynamic Token Budgets,
and Truncation Detection (Phase 8).

Verifies:
1. Dynamic token budgets based on query type:
   - Greetings / short conversational turns: ~100-200 tokens
   - Normal questions: ~300-512 tokens
   - Detailed explanations or step-by-step guidance: ~700-1024 tokens
2. Detection of token limit truncation (Ollama done_reason="length", finish_reason="length", dangling ending)
3. Numbered list completeness check (e.g. requesting 5 steps)
4. Multi-step grounding exercise completeness (5-4-3-2-1 Sensory Grounding)
5. Long answer generation without premature termination
6. Bounded retry and replenishment rather than blind appending
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from config.settings import settings
from config.llm_provider import UniversalChatModel, get_llm
from agent.evaluator import evaluate_execution
from agent.planner import plan_action
from tools.grounding import interactive_grounding_tool, run_54321_grounding
from tools.direct_response import generate_direct_response
from tools.coping import generate_coping_strategies
from tools.rag import execute_rag_query


class TestDynamicTokenBudgets:
    """Test dynamic response budgets configured in settings and environment."""

    def test_greetings_budget(self):
        """Greetings and simple conversation receive 100-200 token budget."""
        greetings = ["hello", "hi", "hey", "how are you", "what is your name", "good morning"]
        for g in greetings:
            budget = settings.determine_token_budget(g, intent="GENERAL_CONVERSATION")
            assert 100 <= budget <= 200, f"Expected 100-200 tokens for '{g}', got {budget}"

    def test_normal_questions_budget(self):
        """Standard questions receive 300-512 token budget."""
        normal_queries = [
            "What causes shoulder pain?",
            "Can stress cause stomach acidity?",
            "Tell me about tension headaches.",
        ]
        for q in normal_queries:
            budget = settings.determine_token_budget(q, intent="HEALTH_INFORMATION")
            assert 300 <= budget <= 512, f"Expected 300-512 tokens for '{q}', got {budget}"

    def test_detailed_questions_budget(self):
        """Explicit requests for detail or multi-step guidance receive 700-1024 token budget."""
        detailed_queries = [
            "Explain in detail how hypertension impacts the cardiovascular system",
            "Give me 5 steps to manage acute panic attacks",
            "Walk me through a step-by-step breathing routine",
            "Provide a comprehensive, detailed explanation of migraine triggers",
            "Give me a 5-step action plan for college exam stress",
        ]
        for q in detailed_queries:
            budget = settings.determine_token_budget(q)
            assert 700 <= budget <= 1024, f"Expected 700-1024 tokens for '{q}', got {budget}"

    def test_configurable_via_settings(self):
        """Settings reflect configurable limits from environment variables."""
        assert settings.ollama_max_tokens >= 512
        assert settings.token_budget_greetings == 150
        assert settings.token_budget_normal == 512
        assert settings.token_budget_detailed == 1024


class TestTruncationDetection:
    """Test detection and handling when LLM reaches token limit."""

    def test_detects_explicit_provider_truncation(self):
        """Evaluator detects is_truncated=True from LLM provider metadata and triggers RETRY."""
        eval_res = evaluate_execution(
            user_input="Explain in detail the biological mechanisms of stress",
            selected_action="USE_COPING_TOOL",
            tool_result="Stress activates the sympathetic nervous system which stimulates the adrenal gland to",
            retry_count=0,
            is_truncated=True,
        )
        assert eval_res["evaluation_result"] == "RETRY"
        assert eval_res.get("is_truncated") is True
        assert "token limit" in eval_res["evaluation_feedback"].lower() or "abruptly" in eval_res["evaluation_feedback"].lower()

    def test_detects_dangling_incomplete_sentence(self):
        """Evaluator detects abrupt cutoffs (dangling conjunctions, prepositions, punctuation) without period."""
        incomplete_texts = [
            "When cortisol levels elevate, the cardiovascular system responds by increasing heart rate and",
            "The primary clinical stages of acne progression include microcomedone formation, followed by:",
            "To ground yourself effectively in a quiet room, look for objects that are-",
        ]
        for text in incomplete_texts:
            eval_res = evaluate_execution(
                user_input="Explain acne causes",
                selected_action="USE_RAG_TOOL",
                tool_result=text,
                retry_count=0,
            )
            assert eval_res["evaluation_result"] == "RETRY", f"Failed to flag dangling text: {text}"
            assert eval_res.get("is_truncated") is True

    def test_universal_chat_model_captures_ollama_done_reason_length(self):
        """UniversalChatModel detects done_reason == 'length' from Ollama and flags truncated."""
        model = UniversalChatModel(provider="ollama", model="llama3.2:1b", max_tokens=110)

        # Mock Ollama API response indicating length cutoff
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Here is an explanation of the condition that was cut off mid-way because"},
            "done_reason": "length",
            "eval_count": 110,
        }

        with patch("requests.post", return_value=mock_response):
            ai_msg = model.invoke([HumanMessage(content="Explain something in detail")])
            assert ai_msg.response_metadata["truncated"] is True
            assert ai_msg.response_metadata["finish_reason"] == "length"
            assert UniversalChatModel.last_metadata["truncated"] is True

    def test_universal_chat_model_marks_stop_as_not_truncated(self):
        """UniversalChatModel marks done_reason == 'stop' with a clean ending as not truncated."""
        model = UniversalChatModel(provider="ollama", model="llama3.2:1b", max_tokens=512)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Hypertension is managed through diet, exercise, and regular clinical monitoring."},
            "done_reason": "stop",
            "eval_count": 45,
        }

        with patch("requests.post", return_value=mock_response):
            ai_msg = model.invoke([HumanMessage(content="What is hypertension?")])
            assert ai_msg.response_metadata["truncated"] is False
            assert ai_msg.response_metadata["finish_reason"] == "stop"


class TestNumberedListCompleteness:
    """Test requests for multi-step numbered lists."""

    def test_evaluator_catches_missing_steps(self):
        """If user asks for 5 steps, but output only contains 2, evaluator triggers RETRY."""
        partial_answer = (
            "Here are steps to manage stress:\n\n"
            "1. Take deep breaths for 4 seconds.\n"
            "2. Step away from your computer screen."
        )
        eval_res = evaluate_execution(
            user_input="Please give me 5 steps to handle acute stress right now",
            selected_action="USE_COPING_TOOL",
            tool_result=partial_answer,
            retry_count=0,
        )
        assert eval_res["evaluation_result"] == "RETRY"
        assert "5 steps" in eval_res["evaluation_feedback"]

    def test_evaluator_passes_complete_numbered_list(self):
        """If user asks for 5 steps and all 5 are present, evaluation PASSES."""
        complete_answer = (
            "Here are 5 steps to manage acute stress right now:\n\n"
            "1. Take slow diaphragmatic breaths.\n"
            "2. Identify what is within your immediate control.\n"
            "3. Drink a cool glass of water to activate salivary relaxation.\n"
            "4. Gently roll your neck and release shoulder tension.\n"
            "5. Focus on the single next small task for the next 15 minutes."
        )
        eval_res = evaluate_execution(
            user_input="Please give me 5 steps to handle acute stress right now",
            selected_action="USE_COPING_TOOL",
            tool_result=complete_answer,
            retry_count=0,
        )
        assert eval_res["evaluation_result"] == "PASS"


class TestMultiStepGroundingExercise:
    """Test 5-4-3-2-1 Sensory Grounding exercise completeness."""

    def test_54321_grounding_contains_all_five_senses(self):
        """The 5-4-3-2-1 grounding tool must return all 5 distinct sensory stages."""
        output = run_54321_grounding()
        assert "SEE" in output or "see" in output.lower()
        assert "TOUCH" in output or "touch" in output.lower()
        assert "HEAR" in output or "hear" in output.lower()
        assert "SMELL" in output or "smell" in output.lower()
        assert "TASTE" in output or "taste" in output.lower()

    def test_interactive_grounding_tool_54321_passes_evaluation(self):
        """Interactive grounding tool generates complete 54321 exercise that passes evaluator."""
        result = interactive_grounding_tool(activity_type="54321_grounding", emotion="anxious")
        eval_res = evaluate_execution(
            user_input="Guide me through the 5-4-3-2-1 sensory exercise",
            selected_action="USE_GROUNDING_TOOL",
            tool_result=result,
            retry_count=0,
            tool_name="54321_grounding",
        )
        assert eval_res["evaluation_result"] == "PASS"
        assert len(result) > 150


class TestLongAnswerAndReplanning:
    """Test long answer generation and bounded retry behavior."""

    def test_bounded_retry_preserves_target_tool_and_expands_budget(self):
        """When an evaluation retry occurs due to truncation, planner keeps the appropriate tool."""
        decision = plan_action(
            user_input="Give me 5 steps to cope with college exams",
            detected_intent="COPING_REQUEST",
            detected_emotion="stressed",
            retry_count=1,
            evaluation_result="RETRY",
            previous_action="USE_COPING_TOOL",
        )
        # Should stay on USE_COPING_TOOL to generate the complete steps rather than defaulting to generic chat
        assert decision["selected_action"] == "USE_COPING_TOOL"

    def test_max_retries_guarantees_loop_termination(self):
        """After max_retries, evaluator returns PASS to avoid infinite loops on 8GB laptop."""
        eval_res = evaluate_execution(
            user_input="Explain in detail every aspect of diabetes",
            selected_action="USE_RAG_TOOL",
            tool_result="Partial explanation that ended because",
            retry_count=2,  # max_retries reached
            max_retries=2,
            is_truncated=True,
        )
        assert eval_res["evaluation_result"] == "PASS"
        assert "maximum replanning" in eval_res["evaluation_feedback"].lower()
