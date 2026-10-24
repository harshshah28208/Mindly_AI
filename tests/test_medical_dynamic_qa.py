"""Comprehensive test suite for dynamic medical intent detection and structured clinical RAG responses."""

import unittest
from agent.intent import is_physical_health_query, classify_intent
from agent.planner import plan_action
from tools.rag import execute_rag_query, get_suggested_followups, screen_medical_urgency


class TestDynamicMedicalQA(unittest.TestCase):
    """Verify dynamic medical intent detection, structured clinical plans, and follow-up generation."""

    def test_01_dynamic_intent_recognition_symptoms(self):
        """Verify dynamic natural language detection for varied bodily complaints."""
        test_queries = [
            "i am having acidity what to do?",
            "how to cure acidity",
            "what is the reason for acidity",
            "i have gas and bloating",
            "why do i have headache",
            "how to treat cough",
            "what to do for fever",
            "what is the cure for gerd",
            "my stomach is burning and hurts",
            "i am suffering from acid reflux",
            "what medicine for migraine",
            "how to relieve stomach cramps",
        ]
        for q in test_queries:
            with self.subTest(query=q):
                self.assertTrue(
                    is_physical_health_query(q),
                    f"Expected is_physical_health_query=True for '{q}'",
                )
                intent = classify_intent(q)
                self.assertIn(
                    intent,
                    ["HEALTH_INFORMATION", "URGENT_HEALTH"],
                    f"Expected HEALTH_INFORMATION for '{q}', got {intent}",
                )

    def test_02_planner_routes_acidity_to_rag_tool(self):
        """Verify 'i am having acidity what to do?' routes to USE_RAG_TOOL and never guided_reflection."""
        plan = plan_action(user_input="i am having acidity what to do?")
        self.assertEqual(plan["selected_action"], "USE_RAG_TOOL")
        self.assertEqual(plan["tool_name"], "gale_retriever")
        self.assertNotIn("guided_reflection", plan.get("tool_name", ""))
        self.assertNotIn("mindful", plan.get("reasoning_summary", "").lower())

    def test_03_acidity_response_contains_reason_and_proper_cure_plan(self):
        """Verify acidity response contains What Is Happening, Reason, Immediate Plan, and Cure Plan."""
        answer, chunks = execute_rag_query("i am having acidity what to do?")
        self.assertTrue(len(chunks) > 0)

        # Check required structured sections requested by user
        self.assertIn("Clinical Overview", answer)
        self.assertIn("What Is the Reason", answer)
        self.assertIn("Immediate Action Plan", answer)
        self.assertIn("How to Cure & Prevent", answer)
        self.assertIn("Warning Signs", answer)
        self.assertIn("Gale Encyclopedia of Medicine", answer)
        self.assertIn("Page 5", answer)

        # Verify educational and practical content comes before disclaimer
        idx_reason = answer.find("What Is the Reason")
        idx_plan = answer.find("Immediate Action Plan")
        idx_disclaimer = answer.find("Educational Disclaimer")
        self.assertTrue(idx_reason > 0)
        self.assertTrue(idx_plan > idx_reason)
        self.assertTrue(idx_disclaimer > idx_plan, "Educational information must appear before disclaimer")

    def test_04_chatgpt_gemini_followup_questions(self):
        """Verify dynamic suggested follow-up questions tailored to medical conditions."""
        followups = get_suggested_followups("i am having acidity what to do?")
        self.assertIsInstance(followups, list)
        self.assertGreaterEqual(len(followups), 3)
        self.assertTrue(any("acidity" in f.lower() or "acid" in f.lower() or "heartburn" in f.lower() for f in followups))

    def test_05_replanning_preserves_medical_rag_routing(self):
        """Verify evaluation retry on medical query stays on USE_RAG_TOOL instead of switching to guided reflection."""
        retry_plan = plan_action(
            user_input="i am having acidity what to do?",
            evaluation_result="RETRY",
            retry_count=1,
            previous_action="USE_RAG_TOOL",
        )
        self.assertEqual(retry_plan["selected_action"], "USE_RAG_TOOL")
        self.assertEqual(retry_plan["tool_name"], "gale_retriever")


if __name__ == "__main__":
    unittest.main()
