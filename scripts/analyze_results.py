import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

data = json.load(open('test_40_results.json', encoding='utf-8'))
print(f"Total tests: {len(data)}\n")

for d in data[:20]:
    t_id = d["id"]
    sec = d["section"]
    inp = d["input"]
    act = d["action"]
    tool = d["tool"]
    resp = d["response"].replace("\n", " ")[:90]
    print(f"Test {t_id:02d} [{sec}] | Route: {act} ({tool})")
    print(f"   In:   {inp}")
    print(f"   Out:  {resp}...")
    print(f"   Note: {d.get('reasoning', '')[:80]}")
    print("-" * 60)
