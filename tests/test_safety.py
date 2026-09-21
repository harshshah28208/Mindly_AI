"""Unit tests for Mindly deterministic safety pre-check."""

import unittest
from agent.safety import assess_safety, generate_crisis_response


class TestSafetyGuard(unittest.TestCase):
    """Test deterministic safety guard classifications."""

    def test_high_risk_detection(self):
        high_risk_phrases = [
            "I want to hurt myself.",
            "I feel like killing myself today.",
            "I want to commit suicide.",
            "I just want to end it all.",
            "I don't want to live anymore.",
            "I'm thinking about cutting myself.",
            "I might take my own life.",
        ]
        for phrase in high_risk_phrases:
            result = assess_safety(phrase)
            self.assertEqual(
                result["risk_level"],
                "HIGH",
                f"Expected HIGH risk for phrase: '{phrase}', got {result['risk_level']}",
            )

    def test_medium_risk_detection(self):
        medium_risk_phrases = [
            "I can't take this anymore.",
            "I am feeling completely hopeless about everything.",
            "I feel like I'm breaking down.",
            "Having a severe panic attack right now.",
        ]
        for phrase in medium_risk_phrases:
            result = assess_safety(phrase)
            self.assertEqual(
                result["risk_level"],
                "MEDIUM",
                f"Expected MEDIUM risk for phrase: '{phrase}', got {result['risk_level']}",
            )

    def test_low_risk_normal_queries(self):
        low_risk_phrases = [
            "Hello Mindly",
            "I am feeling stressed about my college work.",
            "Can you help me with a breathing exercise?",
            "What is your name?",
            "I am feeling tired today.",
        ]
        for phrase in low_risk_phrases:
            result = assess_safety(phrase)
            self.assertEqual(
                result["risk_level"],
                "LOW",
                f"Expected LOW risk for phrase: '{phrase}', got {result['risk_level']}",
            )

    def test_crisis_response_content(self):
        response = generate_crisis_response()
        self.assertIn("988", response)
        self.assertIn("741741", response)
        self.assertIn("Tele-MANAS", response)
        # Verify no false emergency dispatch claims
        self.assertNotIn("I have called 911", response)
        self.assertNotIn("I contacted the police", response)


if __name__ == "__main__":
    unittest.main()
