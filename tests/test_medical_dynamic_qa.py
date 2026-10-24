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

    def test_06_knee_pain_rag_clinical_response_and_reason_cure(self):
        """Verify 'i am having pain in my knee' returns What Is The Reason and Proper Plan How To Cure."""
        answer, chunks = execute_rag_query("i am having pain in my knee")
        self.assertTrue(len(chunks) > 0)
        self.assertIn("Clinical Overview", answer)
        self.assertIn("What Is the Reason", answer)
        self.assertIn("Immediate Action Plan", answer)
        self.assertIn("How to Cure & Prevent", answer)
        self.assertIn("Warning Signs", answer)
        self.assertIn("Gale Encyclopedia of Medicine", answer)
        self.assertIn("Page 6", answer)
        # Verify no canned refusal was emitted
        self.assertNotIn("i can't provide a diagnosis", answer.lower())
        self.assertNotIn("i cannot provide medical advice", answer.lower())

    def test_07_canned_refusal_override(self):
        """Verify that small LLM canned refusal is rejected and replaced by structured clinical report."""
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "I can't provide a diagnosis or medical advice. If you are experiencing persistent or "
            "severe pain in your knee, I recommend that you consult a qualified healthcare professional "
            "for a proper evaluation and treatment plan. Can I help you with any other questions or concerns?"
        )
        answer, chunks = execute_rag_query("i am having pain in my knee", llm=mock_llm)
        self.assertTrue(len(chunks) > 0)
        self.assertIn("Clinical Overview", answer)
        self.assertIn("What Is the Reason", answer)
        self.assertIn("How to Cure & Prevent", answer)
        self.assertNotIn("can't provide a diagnosis or medical advice", answer)

    def test_08_dynamic_unlisted_symptom(self):
        """Verify dynamic clinical synthesis for novel unlisted symptoms like shoulder strain."""
        answer, chunks = execute_rag_query("my shoulder hurts after lifting weights")
        self.assertTrue(len(chunks) > 0)
        self.assertIn("Clinical Overview", answer)
        self.assertIn("What Is the Reason", answer)
        self.assertIn("Immediate Action Plan", answer)
        self.assertIn("How to Cure & Prevent", answer)
        self.assertIn("Warning Signs", answer)

    def test_09_knee_pain_suggested_followups(self):
        """Verify dynamic suggested follow-ups for knee and joint complaints."""
        followups = get_suggested_followups("i am having pain in my knee")
        self.assertIsInstance(followups, list)
        self.assertGreaterEqual(len(followups), 3)
        self.assertTrue(any("r.i.c.e." in f.lower() or "quadriceps" in f.lower() or "cold" in f.lower() or "orthopedic" in f.lower() for f in followups))


if __name__ == "__main__":
    unittest.main()

