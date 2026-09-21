import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from graph import build_mindly_graph, send_message
from agent.intent import classify_intent
from agent.planner import plan_action

g = build_mindly_graph(use_simulation=True)

test_queries = [
    ("Test A (Physical Symptom)", "I am having pain in my stomach."),
    ("Test B (Physical + Emotional Fear)", "My stomach hurts badly and I am scared."),
    ("Test C (Grounding/Coping)", "I feel anxious and need help calming down."),
    ("Test D (Emotional Metaphor/Idiom)", "I feel butterflies in my stomach because I am nervous."),
    ("Urgent Red Flag 1", "I have severe worsening abdominal pain and I am fainting."),
    ("Urgent Red Flag 2", "I am vomiting blood with intense stomach pain."),
]

for name, q in test_queries:
    intent = classify_intent(q)
    plan = plan_action(q, detected_intent=intent, detected_emotion="neutral", risk_level="LOW")
    res = send_message(q, thread_id=f"verify-{uuid.uuid4()}", graph=g, use_simulation=True)
    
    print(f"================ {name} ================")
    print(f"Query: \"{q}\"")
    print(f"Intent: {intent}")
    print(f"Action: {res.get('action')} | Tool: {res.get('tool')}")
    print(f"Evaluator: {res.get('evaluation')}")
    print(f"Sources retrieved: {len(res.get('sources', []))}")
    resp = res.get('response', '').strip()
    first_lines = "\n".join(resp.split("\n")[:4])
    print(f"Response snippet:\n{first_lines}\n")
