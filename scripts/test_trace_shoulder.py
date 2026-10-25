import sys
sys.path.insert(0, ".")

from agent.intent import classify_intent
from agent.planner import plan_action

query = "how to handle shoulder pain"
intent = classify_intent(query)
plan = plan_action(user_input=query, detected_intent=intent)

print(f"Query: '{query}'")
print(f"Intent: {intent}")
print(f"Plan: {plan}")
