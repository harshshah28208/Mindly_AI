"""Unit and integration tests for Mindly Lab 3 Gale RAG pipeline."""

import unittest
from pathlib import Path
from config.settings import settings
from rag.ingestion import calculate_file_hash, extract_chunks_from_pdf, ingest_knowledge
from rag.retriever import retrieve_relevant_chunks
from rag.vectorstore import vector_store
from tools.rag import execute_rag_query, format_citations, screen_medical_urgency


class TestGaleRAGPipeline(unittest.TestCase):
    """Test PDF text extraction, chunking, indexing, retrieval, and grounding."""

    def setUp(self):
        self.pdf_path = settings.gale_pdf_path
        self.assertTrue(self.pdf_path.exists(), f"Gale PDF not found at {self.pdf_path}")

    def test_01_pdf_chunking_and_metadata(self):
        """Verify chunks contain required metadata: source, page, chunk_id, document_name."""
        chunks = extract_chunks_from_pdf(self.pdf_path)
        self.assertTrue(len(chunks) > 0)
        first = chunks[0]
        self.assertEqual(first["source"], "Gale Encyclopedia of Medicine")
        self.assertEqual(first["document_name"], "gale_encyclopedia.pdf")
        self.assertIn("page", first)
        self.assertIn("chunk_id", first)
        self.assertTrue(len(first["text"]) > 20)

    def test_02_delta_indexing_fingerprint(self):
        """Verify delta hashing prevents unnecessary re-indexing."""
        res1 = ingest_knowledge(force=False)
        self.assertTrue(res1["success"])

        # Second call without force should report unchanged
        res2 = ingest_knowledge(force=False)
        self.assertTrue(res2["success"])
        self.assertEqual(res2["status"], "unchanged")

    def test_03_vector_store_retrieval(self):
        """Verify vector search retrieves relevant Gale excerpts for hypertension."""
        results = retrieve_relevant_chunks("hypertension blood pressure", top_k=2)
        self.assertTrue(len(results) > 0)
        found_text = " ".join(r["text"] for r in results).lower()
        self.assertIn("hypertension", found_text)
        self.assertEqual(results[0]["page"], 1)

    def test_04_migraine_retrieval(self):
        """Verify retrieval accurately finds migraine headache symptoms on page 2."""
        results = retrieve_relevant_chunks("migraine headache symptoms photophobia", top_k=2)
        self.assertTrue(len(results) > 0)
        found_text = " ".join(r["text"] for r in results).lower()
        self.assertIn("migraine", found_text)
        self.assertEqual(results[0]["page"], 2)

    def test_05_grounded_answer_and_citations(self):
        """Verify execute_rag_query formats verified citations and educational disclaimer."""
        answer, chunks = execute_rag_query("What is hypertension?")
        self.assertTrue(len(chunks) > 0)
        self.assertIn("Gale Encyclopedia of Medicine", answer)
        self.assertIn("Source", answer)
        self.assertIn("Educational Disclaimer", answer)

    def test_06_insufficient_knowledge_handling(self):
        """Verify out-of-domain query reports insufficient information without hallucinating."""
        # Query totally absent from medical encyclopedia
        answer, chunks = execute_rag_query("What is quantum chromodynamics in string theory?", top_k=1)
        self.assertTrue(len(answer) > 20)

    def test_07_rag_failure_simulation_missing_pdf(self):
        """Category I: Verify extract_chunks_from_pdf returns empty list gracefully if PDF missing."""
        missing_path = Path("data/knowledge/non_existent_file.pdf")
        chunks = extract_chunks_from_pdf(missing_path)
        self.assertEqual(chunks, [])

    def test_08_rag_failure_simulation_retriever_exception(self):
        """Category I: Verify execute_rag_query catches retriever failure without crashing."""
        from unittest.mock import patch
        with patch("tools.rag.retrieve_relevant_chunks", side_effect=RuntimeError("Vector DB index corrupted")):
            answer, chunks = execute_rag_query("What is hypertension?")
            self.assertEqual(chunks, [])
            self.assertIn("could not access the Gale Encyclopedia", answer)

    def test_09_rag_empty_retrieval_no_fake_citations(self):
        """Category I & J: Verify empty retrieval produces no fabricated citations."""
        from unittest.mock import patch
        with patch("tools.rag.retrieve_relevant_chunks", return_value=[]):
            answer, chunks = execute_rag_query("Some rare condition")
            self.assertEqual(chunks, [])
            self.assertNotIn("Page:", answer)

    def test_10_stomach_pain_retrieval_and_page5_citation(self):
        """Verify stomach pain retrieves actual Page 5 content and avoids diagnosis."""
        answer, chunks = execute_rag_query("I am having pain in my stomach.")
        self.assertTrue(len(chunks) > 0)
        # Check actual retriever executed and got Page 5
        pages = [c.get("page") for c in chunks]
        self.assertIn(5, pages)
        # Check educational response avoids diagnosis
        self.assertNotIn("you have appendicitis", answer.lower())
        self.assertNotIn("i diagnose you with", answer.lower())
        # Check source citation and educational content
        self.assertIn("Gale Encyclopedia of Medicine", answer)
        self.assertIn("Page 5", answer)
        self.assertIn("Warning Signs", answer)
        self.assertIn("Educational Disclaimer", answer)

    def test_11_medical_safety_screening_red_flags(self):
        """Section 6: Verify lightweight medical safety screening flags urgent symptoms."""
        # Red flag: severe worsening abdominal pain and fainting
        advisory1 = screen_medical_urgency("severe worsening abdominal pain and fainting")
        self.assertIsNotNone(advisory1)
        self.assertIn("urgent medical assessment", advisory1.lower())

        # Red flag: vomiting blood
        advisory2 = screen_medical_urgency("I am vomiting blood")
        self.assertIsNotNone(advisory2)
        self.assertIn("vomiting blood", advisory2.lower())

        # Red flag: blood in stool
        advisory3 = screen_medical_urgency("I have blood in stool")
        self.assertIsNotNone(advisory3)
        self.assertIn("blood in stool", advisory3.lower())

        # Red flag: chest pain and difficulty breathing
        advisory4 = screen_medical_urgency("chest pain and difficulty breathing")
        self.assertIsNotNone(advisory4)
        self.assertIn("chest pain", advisory4.lower())

        # Red flag: rigid swollen abdomen
        advisory5 = screen_medical_urgency("I have a rigid swollen abdomen")
        self.assertIsNotNone(advisory5)
        self.assertIn("rigid or swollen", advisory5.lower())

        # Non-urgent: routine physical symptom
        non_urgent = screen_medical_urgency("I have mild stomach discomfort after eating")
        self.assertIsNone(non_urgent)

    def test_12_stomach_pain_with_emotional_context_brain_gut(self):
        """Section 3: Verify brain-gut axis education when stomach pain co-occurs with stress."""
        answer, chunks = execute_rag_query("My stomach hurts because I am stressed.", detected_emotion="stressed")
        self.assertTrue(len(chunks) > 0)
        self.assertIn("brain-gut axis", answer.lower())
        self.assertIn("Page 5", answer)

    def test_13_medical_screening_never_diagnoses(self):
        """Section 6: Ensure system never claims 'You have appendicitis' or any diagnosis."""
        answer, _ = execute_rag_query("My stomach hurts badly and I am scared.")
        self.assertNotIn("you have appendicitis", answer.lower())
        self.assertNotIn("i diagnose you with", answer.lower())


if __name__ == "__main__":
    unittest.main()

