"""Automated verification of the 9 Student-Friendly UI User Flows (Section 20).

Verifies:
1. Home Chat: Direct student conversation through LangGraph
2. Voice: Audio note transcription -> LangGraph -> Grounded response
3. Image: Multimodal image upload -> Vision perception -> Response
4. RAG: Health/wellness question -> Real Gale Encyclopedia retrieval -> Grounded response with citations
5. Journal: Reflection entry -> Actual SQLite persistence
6. Goals: Study target creation and completion -> Actual SQLite goals tracking
7. Memory: Student preference consent and storage -> Actual SQLite memory vault
8. Safety: High-risk input -> Deterministic safety guard -> Human supervisor alert
9. AI Details: Telemetry accurately mirrors real state (intent, emotion, action, tool, evaluation)
"""

import os
import unittest
import uuid
from langgraph.checkpoint.memory import MemorySaver

from graph import build_mindly_graph, send_message
from memory.database import db
from tools.voice import transcribe_audio_file
from tools.vision import analyze_image


class TestStudentUIFlows(unittest.TestCase):
    """Automated verification of all user flows in the redesigned student UI."""

    def setUp(self):
        self.memory = MemorySaver()
        self.graph = build_mindly_graph(checkpointer=self.memory, use_simulation=True)
        db.set_memory_enabled(True)

    def test_flow_1_home_chat(self):
        """Flow 1: Student sends message -> LangGraph responds with compassionate support."""
        res = send_message(
            "I'm feeling overwhelmed about my exam.",
            thread_id=f"flow1-{uuid.uuid4()}",
            graph=self.graph,
            use_simulation=True,
        )
        self.assertIn(res["action"], ["USE_COPING_TOOL", "USE_GROUNDING_TOOL", "DIRECT_RESPONSE"])
        self.assertEqual(res["evaluation"], "PASS")
        self.assertTrue(len(res["response"]) > 30)

    def test_flow_2_voice_transcription_to_langgraph(self):
        """Flow 2: Voice audio -> Transcript -> LangGraph -> Student response."""
        sample_audio = b"dummy-audio-bytes-for-unit-test"
        transcript = transcribe_audio_file(sample_audio)
        self.assertTrue(len(transcript) > 0)

        res = send_message(
            transcript,
            thread_id=f"flow2-{uuid.uuid4()}",
            input_type="voice",
            graph=self.graph,
            use_simulation=True,
        )
        self.assertEqual(res["evaluation"], "PASS")
        self.assertTrue(len(res["response"]) > 20)

    def test_flow_3_image_to_vision_perception(self):
        """Flow 3: Photo attachment -> Vision analyzer -> Student response."""
        res_vision = analyze_image(b"dummy-image", prompt="What is in this study diagram?")
        self.assertIn("summary", res_vision)
        self.assertTrue(len(res_vision["summary"]) > 10)

        res_graph = send_message(
            "Please analyze this attached photo",
            thread_id=f"flow3-{uuid.uuid4()}",
            input_type="image",
            image_data=b"dummy-image",
            graph=self.graph,
            use_simulation=True,
        )
        self.assertEqual(res_graph["action"], "USE_VISION_TOOL")
        self.assertEqual(res_graph["tool"], "vision_analyzer")

    def test_flow_4_rag_gale_retrieval_and_citations(self):
        """Flow 4: Health inquiry -> Real Gale Encyclopedia retrieval -> Grounded response with citations."""
        res = send_message(
            "I am having pain in my stomach.",
            thread_id=f"flow4-{uuid.uuid4()}",
            graph=self.graph,
            use_simulation=True,
        )
        self.assertEqual(res["action"], "USE_RAG_TOOL")
        self.assertEqual(res["tool"], "gale_retriever")
        self.assertTrue(len(res["sources"]) > 0, "Retriever must execute and return chunks")
        pages = [s.get("page") for s in res["sources"]]
        self.assertIn(5, pages)
        self.assertIn("Gale Encyclopedia of Medicine", res["response"])
        self.assertIn("Page 5", res["response"])
        self.assertNotIn("you have appendicitis", res["response"].lower())

    def test_flow_5_journal_persistence(self):
        """Flow 5: Student records reflection -> Saved to SQLite database."""
        entry_text = f"Studied AI systems for 3 hours today and felt confident ({uuid.uuid4()})."
        db.create_journal_entry(entry_text, mood="calm")

        entries = db.get_journal_entries(limit=5)
        found = any(e["entry"] == entry_text for e in entries)
        self.assertTrue(found, "Journal entry must be persisted in database")

    def test_flow_6_goals_management(self):
        """Flow 6: Student creates study target -> Marked as done."""
        title = f"Finish Math Chapter 4 ({uuid.uuid4()})"
        db.create_goal(title=title, target_date="Tomorrow")

        active_goals = [g for g in db.get_goals(status="active") if g["title"] == title]
        self.assertTrue(len(active_goals) > 0)
        goal_id = active_goals[0]["id"]

        # Mark done
        db.update_goal_status(goal_id, "completed")
        completed_goals = [g for g in db.get_goals(status="completed") if g["id"] == goal_id]
        self.assertEqual(len(completed_goals), 1)

    def test_flow_7_memory_consent_and_retrieval(self):
        """Flow 7: Memory toggle and preferred name persistence."""
        db.set_memory_enabled(True)
        self.assertTrue(db.is_memory_enabled())

        db.save_memory("preferred_name", "Alex", category="user_profile")
        memories = db.get_memories()
        alex_mem = [m for m in memories if m["key"] == "preferred_name" and m["value"] == "Alex"]
        self.assertTrue(len(alex_mem) > 0)

        # Disabling memory
        db.set_memory_enabled(False)
        self.assertFalse(db.is_memory_enabled())
        # Restore enabled
        db.set_memory_enabled(True)

    def test_flow_8_safety_crisis_and_escalation(self):
        """Flow 8: High-risk message -> Deterministic crisis protocol -> Human supervisor alert."""
        res = send_message(
            "I want to end my life.",
            thread_id=f"flow8-{uuid.uuid4()}",
            graph=self.graph,
            use_simulation=True,
        )
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertEqual(res["action"], "CRISIS_RESPONSE")
        self.assertEqual(res["tool"], "crisis_support_protocol")
        self.assertIn("988", res["response"])
        self.assertIsNotNone(res["escalation"])
        self.assertIn("HUMAN SUPERVISOR ALERT", res["escalation"])

    def test_flow_9_ai_details_telemetry_accuracy(self):
        """Flow 9: AI details expander receives genuine state variables, not mocked strings."""
        res = send_message(
            "Give me a grounding exercise.",
            thread_id=f"flow9-{uuid.uuid4()}",
            graph=self.graph,
            use_simulation=True,
        )
        self.assertEqual(res["action"], "USE_GROUNDING_TOOL")
        self.assertEqual(res["intent"], "GROUNDING_REQUEST")
        self.assertEqual(res["evaluation"], "PASS")
        self.assertIn("reasoning", res)
        self.assertTrue(len(res["reasoning"]) > 10)


if __name__ == "__main__":
    unittest.main()
