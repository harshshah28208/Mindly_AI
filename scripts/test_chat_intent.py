import requests

queries = [
    "how to handle skin acnes",
    "how to stop being jealous",
    "what can i do to manage stress and burnout",
    "what are the causes of hypertension",
    "i have severe stomach ache",
    "guide me through a 4-4-4-4 box breathing exercise",
    "Write a Python script to scrape a website",
    "hello who are you"
]

sys_prompt = """Classify the user input into exactly ONE category:
HEALTH_INFORMATION: physical diseases, illnesses, medical conditions, acne, skin issues, pain, headaches, fever.
EMOTIONAL_SUPPORT: mental wellness, emotions, feelings, jealousy, sadness, loneliness, heartbreak.
COPING_REQUEST: practical coping strategies, stress management, study advice, burnout handling.
GROUNDING_REQUEST: breathing exercises, calming down, grounding techniques.
OUT_OF_SCOPE: computer programming, coding, math, stocks, non-wellness topics.
GENERAL_CONVERSATION: greetings, casual talk, questions about your identity.

Examples:
"how to treat acne" -> HEALTH_INFORMATION
"what causes back pain" -> HEALTH_INFORMATION
"how to stop being jealous" -> EMOTIONAL_SUPPORT
"i feel sad and lonely" -> EMOTIONAL_SUPPORT
"how do i cope with exam stress" -> COPING_REQUEST
"teach me box breathing" -> GROUNDING_REQUEST
"write code in python" -> OUT_OF_SCOPE
"hi how are you" -> GENERAL_CONVERSATION

Respond with ONLY the category name."""

print("--- TESTING FEW-SHOT INTENT CLASSIFICATION ---")
for q in queries:
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f'Input: "{q}"\nCategory:'}
    ]
    res = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2:1b",
            "messages": msgs,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 10}
        },
        timeout=30
    )
    ans = res.json()["message"]["content"].strip()
    print(f'"{q}" -> AI: {ans}', flush=True)
