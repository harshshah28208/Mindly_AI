import sys
sys.path.insert(0, ".")

from agent.intent import classify_intent
from agent.planner import plan_action
from agent.safety import assess_safety, screen_urgent_medical

queries = [
    "Hi Mindly",
    "What is anxiety?",
    "I feel anxious about tomorrow's exam",
    "Guide me through box breathing",
    "I cannot concentrate on studying. Help me make a plan",
    "What causes a migraine?",
    "I have severe stomach pain and feel faint",
    "Explain suicide prevention for my college assignment",
    "Help me save a journal entry",
    "Help me create a daily study goal",
    "What is 2 + 2?",
    "Explain React hooks",
    "how to handle shoulder pain"
]

print("=" * 80)
print("TESTING 13 CANONICAL QUERIES ACROSS INTENT & PLANNER")
print("=" * 80)

for q in queries:
    intent = classify_intent(q)
    plan = plan_action(user_input=q, detected_intent=intent)
    print(f"QUERY: '{q}'")
    print(f"  -> INTENT: {intent}")
    print(f"  -> ACTION: {plan.get('selected_action')} | TOOL: {plan.get('tool_name')}")
    print(f"  -> REASON: {plan.get('reasoning_summary')}")
    print("-" * 80)
