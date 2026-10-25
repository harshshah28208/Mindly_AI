import sys
sys.path.insert(0, ".")
from config.llm_provider import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from agent.intent import INTENT_PROMPT

llm = get_llm()
test_queries = [
    'how to handle skin acnes',
    'how to stop being jealous',
    'what can i do to manage stress',
]

print("--- TESTING AI INTENT CLASSIFICATION ---", flush=True)
for q in test_queries:
    resp = llm.invoke([SystemMessage(content=INTENT_PROMPT), HumanMessage(content=f'User message: "{q}"\nIntent:')])
    print(f"'{q}' -> {resp.content.strip()}", flush=True)
