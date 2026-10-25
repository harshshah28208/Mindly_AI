import sys
sys.path.insert(0, ".")

from agent.intent import classify_intent
from agent.planner import plan_action
from graph import build_mindly_graph

print("=" * 60)
print("TEST 1: INTENT CLASSIFICATION & PLANNER ROUTING")
print("=" * 60)

test_cases = [
    {
        "query": "how to handle skin acnes",
        "expected_intent": "HEALTH_INFORMATION",
        "expected_action": "USE_RAG_TOOL"
    },
    {
        "query": "how to stop being jealous",
        "expected_intent": "EMOTIONAL_SUPPORT",
        "expected_action": "DIRECT_RESPONSE"
    },
    {
        "query": "how can i manage college stress",
        "expected_intent": "COPING_REQUEST",
        "expected_action": "USE_COPING_TOOL"
    },
    {
        "query": "guide me through a 4-4-4-4 box breathing exercise",
        "expected_intent": "GROUNDING_REQUEST",
        "expected_action": "USE_GROUNDING_TOOL"
    },
    {
        "query": "what are the causes of hypertension",
        "expected_intent": "HEALTH_INFORMATION",
        "expected_action": "USE_RAG_TOOL"
    },
    {
        "query": "hello who are you",
        "expected_intent": "GENERAL_CONVERSATION",
        "expected_action": "DIRECT_RESPONSE"
    }
]

all_passed = True
for tc in test_cases:
    q = tc["query"]
    intent = classify_intent(q)
    plan = plan_action(user_input=q, detected_intent=intent)
    action = plan["selected_action"]
    tool = plan["tool_name"]

    intent_ok = (intent == tc["expected_intent"])
    action_ok = (action == tc["expected_action"])
    status = "PASS" if (intent_ok and action_ok) else "FAIL"
    if not (intent_ok and action_ok):
        all_passed = False

    print(f"[{status}] Query: '{q}'")
    print(f"       Intent: {intent} (Expected: {tc['expected_intent']})")
    print(f"       Action: {action} | Tool: {tool} (Expected: {tc['expected_action']})")
    print("-" * 60)

print(f"\nOverall Routing Passed: {all_passed}")

print("\n" + "=" * 60)
print("TEST 2: FULL GRAPH EXECUTION (DISEASE vs MENTAL SUPPORT)")
print("=" * 60)

app = build_mindly_graph()

# Test Disease Execution
disease_q = "how to handle skin acnes"
print(f"\nExecuting Disease Query: '{disease_q}'")
result_disease = app.invoke({"user_input": disease_q, "messages": []}, config={"configurable": {"thread_id": "test_disease_1"}})
resp_disease = result_disease.get("final_response", "")
print(f"Detected Intent: {result_disease.get('detected_intent')}")
print(f"Selected Action: {result_disease.get('selected_action')}")
print(f"Response Preview:\n{resp_disease[:400]}...\n")
assert "rotator cuff" not in resp_disease.lower(), "ERROR: Found rotator cuff in acne response!"
assert "artery" not in resp_disease.lower(), "ERROR: Found artery in acne response!"

# Test Emotional/Mental Support Execution
mental_q = "how to stop being jealous"
print(f"\nExecuting Mental Support Query: '{mental_q}'")
result_mental = app.invoke({"user_input": mental_q, "messages": []}, config={"configurable": {"thread_id": "test_mental_1"}})
resp_mental = result_mental.get("final_response", "")
print(f"Detected Intent: {result_mental.get('detected_intent')}")
print(f"Selected Action: {result_mental.get('selected_action')}")
print(f"Response Preview:\n{resp_mental[:400]}...\n")
assert "gale encyclopedia" not in resp_mental.lower(), "ERROR: Gale RAG was called for emotional jealousy!"
assert "rotator cuff" not in resp_mental.lower(), "ERROR: Found rotator cuff in jealousy response!"
assert "artery" not in resp_mental.lower(), "ERROR: Found artery in jealousy response!"

print("=" * 60)
print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
print("=" * 60)
