"""Unit tests for SafeSpace-inspired features: Therapist Locator, Emergency Dispatch, and FastAPI API."""

import unittest
from fastapi.testclient import TestClient

from api import app
from agent.intent import classify_intent
from agent.planner import plan_action
from tools.therapist_locator import find_nearby_therapists, extract_location
from tools.emergency_dispatch import dispatch_emergency_notification


class TestSafeSpaceFeatures(unittest.TestCase):
    """Test SafeSpace AI therapist integration: locator, dispatch, and REST API."""

    def setUp(self):
        self.client = TestClient(app)

    def test_01_therapist_location_extraction(self):
        """Verify location extractor detects city names from user queries."""
        loc1 = extract_location("Find therapists in Mumbai")
        self.assertEqual(loc1, "mumbai")

        loc2 = extract_location("Can you recommend psychologists near New York?")
        self.assertEqual(loc2, "new york")

        loc3 = extract_location("Therapist near London please")
        self.assertEqual(loc3, "london")

    def test_02_therapist_locator_output(self):
        """Verify find_nearby_therapists returns verified clinical psychologists and helplines."""
        res_mumbai = find_nearby_therapists("therapists near Mumbai")
        self.assertIn("Ayesha Kapoor", res_mumbai)
        self.assertIn("Tele-MANAS", res_mumbai)

        res_ny = find_nearby_therapists("psychologists in New York")
        self.assertIn("Sarah Jenkins", res_ny)
        self.assertIn("988", res_ny)

    def test_03_therapist_intent_and_planner_routing(self):
        """Verify therapist inquiry detects THERAPIST_REQUEST and routes to USE_THERAPIST_TOOL."""
        intent = classify_intent("Can you find a therapist near Mumbai for me?")
        self.assertEqual(intent, "THERAPIST_REQUEST")

        plan = plan_action(
            user_input="Can you find a therapist near Mumbai for me?",
            detected_intent=intent,
            detected_emotion="neutral",
            risk_level="LOW",
        )
        self.assertEqual(plan["selected_action"], "USE_THERAPIST_TOOL")
        self.assertEqual(plan["tool_name"], "therapist_locator")

    def test_04_emergency_dispatch_telephony(self):
        """Verify automated crisis dispatch executes safely and logs audit trail."""
        dispatch = dispatch_emergency_notification(
            risk_level="HIGH",
            user_input_snippet="I want to hurt myself",
        )
        self.assertIn("status", dispatch)
        self.assertIn(dispatch["status"], ["DISPATCHED", "CALL_DISPATCHED", "SIMULATED_SUPERVISOR_DISPATCH", "SIMULATED_EMERGENCY_CALL_ACTIVE"])
        self.assertIn("timestamp", dispatch)

    def test_05_fastapi_safespace_ask_endpoint(self):
        """Verify SafeSpace-compatible POST /ask endpoint returns response and tool_called."""
        response = self.client.post("/ask", json={"message": "Can you recommend a therapist in Mumbai?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertIn("tool_called", data)
        self.assertTrue(len(data["response"]) > 20)

    def test_06_fastapi_extended_chat_endpoint(self):
        """Verify POST /api/chat returns full clinical metadata and followups."""
        response = self.client.post("/api/chat", json={"message": "I am having acidity what to do?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertIn("sources", data)
        self.assertIn("followups", data)
        self.assertTrue(len(data["followups"]) >= 3)

    def test_07_emergency_voice_call_high_risk(self):
        """Verify initiate_emergency_call creates valid telephony metadata and logs history."""
        from tools.emergency_dispatch import initiate_emergency_call, get_emergency_dispatch_history
        res = initiate_emergency_call(
            risk_level="HIGH",
            user_input_snippet="I feel like giving up and ending it",
            contact_number="+18002738255",
        )
        self.assertIn("call_sid", res)
        self.assertIn("status", res)
        self.assertEqual(res["risk_level"], "HIGH")
        self.assertEqual(res["type"], "VOICE_CALL")

        history = get_emergency_dispatch_history(limit=5)
        self.assertTrue(len(history) >= 1)
        self.assertEqual(history[0]["risk_level"], "HIGH")

    def test_08_fastapi_emergency_call_endpoint(self):
        """Verify POST /api/emergency/call triggers automated telephony and returns audit history."""
        resp = self.client.post(
            "/api/emergency/call",
            json={
                "contact_number": "+18002738255",
                "message": "Emergency alert test for high risk",
                "risk_level": "HIGH",
            },
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["status"], "success")
        self.assertIn("dispatch", payload)
        self.assertIn("call_sid", payload["dispatch"])
        self.assertIn("recent_history", payload)


if __name__ == "__main__":
    unittest.main()

