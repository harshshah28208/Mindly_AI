import requests

few_shot_prompt = """You are a dialogue classifier for a wellness application. Output ONLY the exact category in brackets: [HEALTH_INFORMATION], [EMOTIONAL_SUPPORT], [GROUNDING_REQUEST], [PLANNING_REQUEST], [GENERAL_CONVERSATION], [OUT_OF_SCOPE].

Example: "my skin has acne" -> [HEALTH_INFORMATION]
Example: "how to stop feeling jealous of friends" -> [EMOTIONAL_SUPPORT]
Example: "write a python script" -> [OUT_OF_SCOPE]
Example: "guide me through 4-4-4-4 breathing" -> [GROUNDING_REQUEST]
Example: "make a study schedule" -> [PLANNING_REQUEST]
Example: "hello how are you" -> [GENERAL_CONVERSATION]

Input: "{query}"
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
    p = few_shot_prompt.replace("{query}", q)
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:1b", "prompt": p, "stream": False, "options": {"temperature": 0.0, "num_predict": 10}},
        timeout=30
    )
    ans = res.json()["response"].strip()
    print(f'"{q}" -> {ans}', flush=True)
