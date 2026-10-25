import requests

prompt_template = """Classify the user message into EXACTLY ONE category:
- HEALTH_INFORMATION (physical medical symptoms, diseases, skin conditions, acne, headaches, pain, nausea, fever)
- EMOTIONAL_SUPPORT (emotions, jealousy, loneliness, anxiety, sadness, stress, relationships)
- COPING_REQUEST (practical strategies to handle stress, habits, emotions, mindset)
- GROUNDING_REQUEST (breathing exercises, 5-4-3-2-1, panic calming)
- PLANNING_REQUEST (study schedules, daily routines, exam prep)
- GENERAL_CONVERSATION (greetings, identity, casual chat)
- OUT_OF_SCOPE (coding, math, sports, non-wellness)

User message: "{query}"
Category:"""

queries = [
    "how to stop being jealous",
    "how to handle skin acnes",
    "Write a Python script to scrape a website using BeautifulSoup",
    "I am feeling very anxious about my vivas tomorrow",
    "guide me through a 4-4-4-4 box breathing exercise",
    "hello who are you"
]

for q in queries:
    p = prompt_template.replace("{query}", q)
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:1b", "prompt": p, "stream": False, "options": {"temperature": 0.0, "num_predict": 10}},
        timeout=30
    )
    cat = res.json()["response"].strip().replace("-", "").strip()
    print(f'"{q}" -> AI Decided: {cat}')
