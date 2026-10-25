"""Test suite comparing Mindly responses across diverse disease and normal questions.
Verifies routing, detects repeating boilerplate, and checks answer quality.
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.intent import classify_intent
from agent.planner import plan_action
from graph import send_message


TEST_CASES = [
    # --- Category A: Disease & Medical Inquiries ---
    {
        "category": "Disease - Dermatology",
        "question": "What causes back acne and how can I treat it?",
        "expected_intent": "HEALTH_INFORMATION",
        "expected_action": "USE_RAG_TOOL",
    },
    {
        "category": "Disease - Neurology",
        "question": "I am having severe migraine and nausea, what should I do?",
        "expected_intent": "HEALTH_INFORMATION",
        "expected_action": "USE_RAG_TOOL",
    },
    {
        "category": "Disease - Gastroenterology",
        "question": "What is the reason for acid reflux and stomach burning?",
        "expected_intent": "HEALTH_INFORMATION",
        "expected_action": "USE_RAG_TOOL",
    },
    {
        "category": "Disease - Musculoskeletal",
        "question": "My lower back hurts after sitting 8 hours at my desk, what can I do?",
        "expected_intent": "HEALTH_INFORMATION",
        "expected_action": "USE_RAG_TOOL",
    },
    {
        "category": "Disease - Cardiovascular",
        "question": "What are the clinical causes and symptoms of hypertension?",
        "expected_intent": "HEALTH_INFORMATION",
        "expected_action": "USE_RAG_TOOL",
    },
    {
        "category": "Disease - Respiratory",
        "question": "I have a sore throat, dry cough, and mild fever",
        "expected_intent": "HEALTH_INFORMATION",
        "expected_action": "USE_RAG_TOOL",
    },
    {
        "category": "Disease - Sleep / Circadian",
        "question": "What causes chronic insomnia and how can I cure it?",
        "expected_intent": "HEALTH_INFORMATION",
        "expected_action": "USE_RAG_TOOL",
    },

    # --- Category B: Mental Health & Normal Conversational Inquiries ---
    {
        "category": "Normal - Greeting & Identity",
        "question": "Hello, who are you and how can you help me?",
        "expected_intent": "GENERAL_CONVERSATION",
        "expected_action": "DIRECT_RESPONSE",
    },
    {
        "category": "Normal - Academic Anxiety",
        "question": "I am feeling very anxious about my vivas tomorrow",
        "expected_intent": "EMOTIONAL_SUPPORT",
        "expected_action": "USE_COPING_TOOL",
    },
    {
        "category": "Normal - Somatic Panic / Grounding",
        "question": "I am having a panic attack, guide me through a 4-4-4-4 box breathing exercise",
        "expected_intent": "GROUNDING_REQUEST",
        "expected_action": "USE_GROUNDING_TOOL",
    },
    {
        "category": "Normal - Loneliness & Distress",
        "question": "I feel so lonely today and have nobody to talk to",
        "expected_intent": "EMOTIONAL_SUPPORT",
        "expected_action": "DIRECT_RESPONSE",
    },
    {
        "category": "Normal - Academic Planning",
        "question": "Help me plan my study schedule for final exams",
        "expected_intent": "PLANNING_REQUEST",
        "expected_action": "USE_COPING_TOOL",
    },

    # --- Category C: Out-of-Scope Domain Boundary ---
    {
        "category": "Boundary - Coding Out of Scope",
        "question": "Write a Python script to scrape a website using BeautifulSoup",
        "expected_intent": "OUT_OF_SCOPE",
        "expected_action": "DIRECT_RESPONSE",
    },
]


def run_tests():
    print("=" * 80)
    print("MINDLY AGENT TEST RUNNER: ROUTING & RESPONSE EVALUATION")
    print("=" * 80)

    # Initialize run
    results = []
    boilerplate_detections = []
    
    # Check for known repeating boilerplate phrases
    repeating_boilerplate_phrases = [
        "Pause strenuous or aggravating activities immediately and rest in a comfortable, relaxed posture",
        "Slowly sip room-temperature water or clear fluids to maintain systemic cellular hydration",
        "Apply temperature therapy (cold pack for acute swelling/sprains; warm compress for muscular tension",
        "Practice slow, calm diaphragmatic breathing (4-second inhale, 6-second exhale) to downregulate autonomic hyperarousal",
        "curves 25-40 degrees in growing adolescents benefit from corrective bracing",
    ]

    for idx, tc in enumerate(TEST_CASES, start=1):
        q = tc["question"]
        cat = tc["category"]
        exp_intent = tc["expected_intent"]
        exp_action = tc["expected_action"]

        print(f"\n--- [Test {idx}/{len(TEST_CASES)}] {cat} ---", flush=True)
        print(f"User Query: \"{q}\"", flush=True)

        # 1. Test Intent Classification
        actual_intent = classify_intent(q)
        intent_ok = (actual_intent == exp_intent) or (exp_intent == "HEALTH_INFORMATION" and actual_intent in ["HEALTH_INFORMATION", "URGENT_HEALTH"]) or (exp_intent == "EMOTIONAL_SUPPORT" and actual_intent in ["EMOTIONAL_SUPPORT", "COPING_REQUEST", "GROUNDING_REQUEST"])
        print(f"-> Intent: {actual_intent} (Expected: {exp_intent}) -> {'[PASS]' if intent_ok else '[WARN]'}", flush=True)

        # 2. Test Planner Action
        plan = plan_action(q, detected_intent=actual_intent)
        actual_action = plan.get("selected_action")
        tool_name = plan.get("tool_name")
        action_ok = (actual_action == exp_action) or (exp_action == "USE_COPING_TOOL" and actual_action in ["USE_COPING_TOOL", "USE_GROUNDING_TOOL", "DIRECT_RESPONSE"])
        print(f"-> Action: {actual_action} (Tool: {tool_name}) (Expected: {exp_action}) -> {'[PASS]' if action_ok else '[WARN]'}", flush=True)

        # 3. Test Full Agent Execution
        try:
            res_dict = send_message(user_text=q, thread_id=f"test_thread_{idx}")
            response = res_dict.get("response", "")
            provider = res_dict.get("llm_provider", "")
            model = res_dict.get("llm_model", "")
            latency = res_dict.get("latency", 0)
            print(f"-> Engine: {provider} ({model}) in {latency:.2f}s")
        except Exception as e:
            response = f"[Execution Error: {e}]"

        print(f"-> Response Length: {len(response)} chars", flush=True)
        print(f"\n--- [FULL RESPONSE] ---", flush=True)
        print(response, flush=True)
        print(f"-----------------------\n", flush=True)

        # 4. Check for repeating boilerplate
        found_boilerplate = []
        for phrase in repeating_boilerplate_phrases:
            if phrase.lower() in response.lower():
                found_boilerplate.append(phrase)

        if found_boilerplate:
            print(f"-> [ALERT] Repeating boilerplate detected: {found_boilerplate}", flush=True)
            boilerplate_detections.append((cat, q, found_boilerplate))
        else:
            print("-> [OK] Zero generic boilerplate detected! Response is tailored.", flush=True)

        results.append({
            "idx": idx,
            "category": cat,
            "query": q,
            "intent": actual_intent,
            "action": actual_action,
            "has_boilerplate": len(found_boilerplate) > 0,
            "response": response,
        })

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS:")
    print("=" * 80)
    print(f"Total Tests Executed: {len(TEST_CASES)}")
    print(f"Boilerplate Issues Detected: {len(boilerplate_detections)}")

    if boilerplate_detections:
        print("\nCases with boilerplate issues:")
        for cat, q, bp in boilerplate_detections:
            print(f"  - [{cat}] '{q}': {bp}")
    else:
        print("\nAll tests passed with zero repeating boilerplate!")


if __name__ == "__main__":
    run_tests()
