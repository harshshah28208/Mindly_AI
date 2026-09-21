"""Comprehensive end-to-end integration tests for Mindly Lab 3.

Verifies the 9 Reference Demonstration Flows defined in Section 42:
- DEMO 1: Normal Greeting (Direct response, NO RAG)
- DEMO 2: Emotional Support (Emotion analysis, Coping/Grounding, Evaluator)
- DEMO 3: Knowledge RAG (Intent, Planner, Gale RAG, Retrieval, Grounded answer, Source display)
- DEMO 4: Voice Input & Output (Transcription -> Graph -> Response -> TTS synthesis)
- DEMO 5: Vision Understanding (Image upload -> Vision analyzer -> Agent response)
- DEMO 6: Wellness Journaling (Journal command -> Entry saved to SQLite)
- DEMO 7: Goals Management (Goal command -> Goal saved to SQLite)
- DEMO 8: Long-Term Memory (Remember preferred name -> Recall from database)
- DEMO 9: Crisis & Supervisor Escalation (Deterministic safety -> Crisis response -> Demo escalation)
"""

import os
import unittest
import uuid
from langgraph.checkpoint.memory import MemorySaver

from graph import build_mindly_graph, send_message
from memory.database import db
from tools.voice import synthesize_speech, transcribe_audio_file
from tools.vision import analyze_image


class TestMindlyLab3Integration(unittest.TestCase):
    """End-to-end integration test suite verifying the 9 demonstration flows."""

    def setUp(self):
        self.memory = MemorySaver()
        self.graph = build_mindly_graph(checkpointer=self.memory, use_simulation=True)
        db.set_memory_enabled(True)

    def test_demo_1_normal_greeting_no_rag(self):
        """DEMO 1: User says 'Hello Mindly' -> Direct response, NO RAG, no unnecessary tools."""
        res = send_message("Hello Mindly.", thread_id=f"d1-{uuid.uuid4()}", graph=self.graph, use_simulation=True)
        self.assertEqual(res["action"], "DIRECT_RESPONSE")
        self.assertEqual(res["tool"], "direct_conversation")
        self.assertEqual(res["risk_level"], "LOW")
        self.assertEqual(res["evaluation"], "PASS")
        self.assertEqual(len(res["sources"]), 0, "Normal greeting must NOT execute RAG")
        self.assertIn("Mindly", res["response"])

    def test_demo_2_emotional_support(self):
        """DEMO 2: 'I'm overwhelmed with college work.' -> Emotion analysis, Coping/Grounding, Evaluator."""
        res = send_message("I'm overwhelmed with college work.", thread_id=f"d2-{uuid.uuid4()}", graph=self.graph, use_simulation=True)
        self.assertIn(res["action"], ["USE_COPING_TOOL", "USE_GROUNDING_TOOL"])
        self.assertEqual(res["risk_level"], "LOW")
        self.assertEqual(res["evaluation"], "PASS")
        self.assertTrue(len(res["response"]) > 40)

    def test_demo_3_knowledge_rag_hypertension(self):
        """DEMO 3: 'What is hypertension?' -> Gale RAG, Retrieval, Grounded answer, Source citations."""
        res = send_message("What is hypertension?", thread_id=f"d3-{uuid.uuid4()}", graph=self.graph, use_simulation=True)
        self.assertEqual(res["action"], "USE_RAG_TOOL")
        self.assertEqual(res["tool"], "gale_retriever")
        self.assertTrue(len(res["sources"]) > 0, "RAG query must return retrieved sources")
        self.assertEqual(res["sources"][0]["source"], "Gale Encyclopedia of Medicine")
        self.assertEqual(res["sources"][0]["page"], 1)
        self.assertIn("Gale Encyclopedia of Medicine", res["response"])
        self.assertIn("hypertension", res["response"].lower())

    def test_demo_4_voice_stt_and_tts(self):
        """DEMO 4: Voice note transcription -> Graph response -> TTS synthesis."""
        sample_voice_bytes = b"fake-audio-bytes-for-test"
        transcript = transcribe_audio_file(sample_voice_bytes)
        self.assertTrue(len(transcript) > 0)

        res = send_message(transcript, thread_id=f"d4-{uuid.uuid4()}", input_type="voice", graph=self.graph, use_simulation=True)
        self.assertEqual(res["evaluation"], "PASS")

        # Synthesize audio TTS
        audio_path = synthesize_speech(res["response"])
        if audio_path:
            self.assertTrue(os.path.exists(audio_path))
            self.assertTrue(audio_path.endswith(".wav"))

    def test_demo_5_vision_understanding(self):
        """DEMO 5: Image upload -> Vision analysis -> Agent response."""
        res_vision = analyze_image(b"fake-image-bytes", prompt="Analyze calming elements")
        self.assertIn("summary", res_vision)
        self.assertIn("Educational Note", res_vision["disclaimer"])

        res_graph = send_message("Please analyze this attached photo", thread_id=f"d5-{uuid.uuid4()}", input_type="image", image_data=b"dummy", graph=self.graph, use_simulation=True)
        self.assertEqual(res_graph["action"], "USE_VISION_TOOL")
        self.assertEqual(res_graph["tool"], "vision_analyzer")

    def test_demo_6_wellness_journal(self):
        """DEMO 6: 'Journal this: Completed my AI lab today.' -> Journal tool, Entry saved."""
        res = send_message("Journal this: Today I completed my AI project and felt proud.", thread_id=f"d6-{uuid.uuid4()}", graph=self.graph, use_simulation=True)
        self.assertEqual(res["action"], "USE_JOURNAL_TOOL")
        self.assertEqual(res["tool"], "journal_recorder")
        self.assertIn("Journal Entry Saved", res["response"])

        entries = db.get_journal_entries(limit=1)
        self.assertTrue(len(entries) > 0)
        self.assertIn("completed my AI project", entries[0]["entry"])

    def test_demo_7_goal_management(self):
        """DEMO 7: 'Create a goal to study for 2 hours daily.' -> Goal tool, Goal saved."""
        res = send_message("Create a goal to study for 2 hours daily.", thread_id=f"d7-{uuid.uuid4()}", graph=self.graph, use_simulation=True)
        self.assertEqual(res["action"], "USE_GOAL_TOOL")
        self.assertEqual(res["tool"], "goal_tracker")
        self.assertIn("Goal Created Successfully", res["response"])

        goals = db.get_goals(status="active")
        self.assertTrue(len(goals) > 0)
        self.assertIn("study for 2 hours daily", goals[0]["title"])

    def test_demo_8_persistent_memory(self):
        """DEMO 8: 'Remember that I prefer being called Rahul.' -> Save; 'What name should you call me?' -> Retrieve."""
        res1 = send_message("Remember that my preferred name is Rahul.", thread_id=f"d8-{uuid.uuid4()}", graph=self.graph, use_simulation=True)
        self.assertEqual(res1["action"], "USE_MEMORY_TOOL")
        self.assertIn("Rahul", res1["response"])

        res2 = send_message("What name should you call me?", thread_id=f"d8-{uuid.uuid4()}", graph=self.graph, use_simulation=True)
        self.assertEqual(res2["action"], "USE_MEMORY_TOOL")
        self.assertIn("Rahul", res2["response"])

    def test_demo_9_crisis_and_supervisor_escalation(self):
        """DEMO 9: 'I want to hurt myself.' -> Deterministic safety, HIGH RISK, Crisis Response, Supervisor alert."""
        res = send_message("I want to hurt myself.", thread_id=f"d9-{uuid.uuid4()}", graph=self.graph, use_simulation=True)
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertEqual(res["action"], "CRISIS_RESPONSE")
        self.assertEqual(res["tool"], "crisis_support_protocol")
        self.assertIn("988", res["response"])
        self.assertIn("Tele-MANAS", res["response"])
        # Verify supervisor escalation triggered in demo mode
        self.assertIsNotNone(res["escalation"])
        self.assertIn("HUMAN SUPERVISOR ALERT", res["escalation"])
        self.assertIn("Simulation", res["escalation"])
        self.assertNotIn("I called 911", res["response"])


    def test_demo_10_medical_symptom_routing_stomach_pain(self):
        """Verify 'I am having pain in my stomach' routes to RAG, executes Gale retriever, and Evaluator passes."""
        res = send_message("I am having pain in my stomach.", thread_id=f"d10-{uuid.uuid4()}", graph=self.graph, use_simulation=True)
        self.assertEqual(res["action"], "USE_RAG_TOOL")
        self.assertEqual(res["tool"], "gale_retriever")
        self.assertTrue(len(res["sources"]) > 0, "Retriever must return Gale chunks")
        pages = [s.get("page") for s in res["sources"]]
        self.assertIn(5, pages, "Must retrieve Page 5 covering Abdominal Pain")
        self.assertEqual(res["evaluation"], "PASS")
        self.assertIn("Gale Encyclopedia of Medicine", res["response"])
        self.assertIn("Page 5", res["response"])
        self.assertNotIn("you have appendicitis", res["response"].lower())

    def test_demo_11_stomach_pain_with_stress_emotional_context(self):
        """Verify 'My stomach hurts because I am stressed' routes to RAG and includes brain-gut axis education."""
        res = send_message("My stomach hurts because I am stressed.", thread_id=f"d11-{uuid.uuid4()}", graph=self.graph, use_simulation=True)
        self.assertEqual(res["action"], "USE_RAG_TOOL")
        self.assertEqual(res["tool"], "gale_retriever")
        self.assertTrue(len(res["sources"]) > 0)
        self.assertIn("brain-gut axis", res["response"].lower())
        self.assertEqual(res["evaluation"], "PASS")

    def test_demo_12_explicit_grounding_not_rag(self):
        """Verify 'I am anxious and I want a breathing exercise' routes to grounding tool, NOT RAG."""
        res = send_message("I am anxious and I want a breathing exercise.", thread_id=f"d12-{uuid.uuid4()}", graph=self.graph, use_simulation=True)
        self.assertEqual(res["action"], "USE_GROUNDING_TOOL")
        self.assertEqual(len(res["sources"]), 0, "Grounding request must not execute RAG")
        self.assertEqual(res["evaluation"], "PASS")


if __name__ == "__main__":
    unittest.main()
