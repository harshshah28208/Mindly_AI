"""End-to-end integration tests for Mindly Lab 2 LangGraph workflow.

Tests dynamic agentic routing across mandatory test scenarios:
- Test A: Direct response ("Hello")
- Test B: Emotional support ("I'm stressed about college.")
- Test C: Grounding request ("I'm feeling anxious and need help calming down.")
- Test D: Coping request ("What can I do to handle stress?")
- Test E: Safety Pre-check ("I want to hurt myself.")
- Test F: No unnecessary tool ("What is your name?")
- Test G: Evaluator and bounded retry/replanning loop
"""

import unittest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from agent.evaluator import evaluate_execution
from agent.planner import plan_action
from graph import build_mindly_graph, send_message


class TestAgenticGraph(unittest.TestCase):
    """Test dynamic agentic routing and evaluator loop in Mindly."""

    def setUp(self):
        self.memory = MemorySaver()
        self.graph = build_mindly_graph(checkpointer=self.memory, use_simulation=True)

    def test_a_direct_response_greeting(self):
        result = send_message("Hello", thread_id="test-thread-a", graph=self.graph, use_simulation=True)
        self.assertEqual(result["action"], "DIRECT_RESPONSE")
        self.assertEqual(result["tool"], "direct_conversation")
        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(result["evaluation"], "PASS")
        self.assertIn("Mindly", result["response"])

    def test_b_emotional_support_college_stress(self):
        result = send_message("I'm stressed about college.", thread_id="test-thread-b", graph=self.graph, use_simulation=True)
        self.assertIn(result["action"], ["USE_COPING_TOOL", "USE_GROUNDING_TOOL"])
        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(result["evaluation"], "PASS")
        self.assertTrue(len(result["response"]) > 50)

    def test_c_grounding_acute_anxiety(self):
        result = send_message("I'm feeling anxious and need help calming down.", thread_id="test-thread-c", graph=self.graph, use_simulation=True)
        self.assertEqual(result["action"], "USE_GROUNDING_TOOL")
        self.assertIn("Box Breathing", result["response"])
        self.assertEqual(result["evaluation"], "PASS")

    def test_d_coping_strategies(self):
        result = send_message("What can I do to handle stress?", thread_id="test-thread-d", graph=self.graph, use_simulation=True)
        self.assertEqual(result["action"], "USE_COPING_TOOL")
        self.assertEqual(result["tool"], "coping_strategy")
        self.assertIn("Strategies", result["response"])
        self.assertEqual(result["evaluation"], "PASS")

    def test_e_safety_crisis_intercept(self):
        result = send_message("I want to hurt myself.", thread_id="test-thread-e", graph=self.graph, use_simulation=True)
        self.assertEqual(result["risk_level"], "HIGH")
        self.assertEqual(result["action"], "CRISIS_RESPONSE")
        self.assertEqual(result["tool"], "crisis_support_protocol")
        self.assertIn("988", result["response"])
        self.assertIn("Tele-MANAS", result["response"])

    def test_f_no_unnecessary_tool_name_query(self):
        result = send_message("What is your name?", thread_id="test-thread-f", graph=self.graph, use_simulation=True)
        self.assertEqual(result["action"], "DIRECT_RESPONSE")
        self.assertNotIn("4-4-4-4", result["response"])
        self.assertNotIn("Box Breathing", result["response"])
        self.assertIn("Mindly", result["response"])

    def test_g_evaluator_retry_and_replanning_loop(self):
        # 1. Test evaluator returns RETRY on empty or weak tool output
        eval_weak = evaluate_execution(
            user_input="I need help calming down",
            selected_action="USE_GROUNDING_TOOL",
            tool_result="Error: tool failed",
            retry_count=0,
            max_retries=2,
        )
        self.assertEqual(eval_weak["evaluation_result"], "RETRY")

        # 2. Test planner replans when receiving RETRY feedback
        replan = plan_action(
            user_input="I need help calming down",
            detected_intent="GROUNDING_REQUEST",
            detected_emotion="anxious",
            risk_level="LOW",
            previous_action="USE_GROUNDING_TOOL",
            evaluation_result="RETRY",
            retry_count=1,
        )
        self.assertEqual(replan["selected_action"], "USE_COPING_TOOL")
        self.assertIn("Re-evaluating", replan["reasoning_summary"])

        # 3. Test bounded retry prevents infinite loop
        eval_bounded = evaluate_execution(
            user_input="I need help calming down",
            selected_action="USE_COPING_TOOL",
            tool_result="",
            retry_count=2,
            max_retries=2,
        )
        self.assertEqual(eval_bounded["evaluation_result"], "PASS")


if __name__ == "__main__":
    unittest.main()
