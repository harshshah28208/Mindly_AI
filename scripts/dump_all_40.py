import sys
import json

sys.stdout.reconfigure(encoding="utf-8")
data = json.load(open('test_40_results.json', encoding='utf-8'))

for d in data[:9]:
    resp_clean = d['response'].replace('\n', ' ')
    print(f"ID: {d['id']:02d}")
    print(f"SEC: {d['section']}")
    print(f"INPUT: {d['input']}")
    print(f"ACTION: {d['action']}")
    print(f"TOOL: {d['tool']}")
    print(f"REASONING: {d.get('reasoning')}")
    print(f"RESPONSE: {resp_clean[:200]}...")
    print("=" * 60)
