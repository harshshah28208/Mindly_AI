import sys
sys.path.insert(0, ".")
from config.llm_provider import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

INTENT_PROMPT = """You are an intent classification component for Mindly, an AI mental wellness and health assistant.
Classify the user's message into EXACTLY ONE of the following categories:
- HEALTH_INFORMATION (physical symptoms, medical questions, diseases, bodily pain, illness, acne, skin issues, stomach ache, headache, fever)
- EMOTIONAL_SUPPORT (mental wellness, feelings, sadness, jealousy, relationship worries, feeling down, loneliness, emotional distress)
- COPING_REQUEST (asking for practical coping strategies, advice to manage stress, burnout, pressure, study challenges, emotional regulation)
- GROUNDING_REQUEST (explicitly asking for immediate calming exercises, breathing techniques, box breathing, 5-4-3-2-1 sensory grounding)
- GENERAL_CONVERSATION (greetings, casual chatting, identity, capabilities)
- JOURNAL_REQUEST (asking to record or view personal journal reflections)
- GOAL_REQUEST (asking to set, track, view, or manage habits and goals)
- PLANNING_REQUEST (asking to create study plans, schedules, routines, time management)
- THERAPIST_REQUEST (asking to find or locate professional human therapists, counselors, clinics)
- URGENT_HEALTH (acute life-threatening medical emergencies like severe chest pain, vomiting blood, difficulty breathing)
- CRISIS_RELATED (suicidal thoughts, self-harm, severe crisis)
- OUT_OF_SCOPE (unrelated non-health/non-wellness topics like computer programming, stock market, sports scores, car mechanics)

Respond with ONLY the exact category name. Nothing else.
"""

llm = get_llm()

queries = [
    "how to handle skin acnes",
    "how to stop being jealous",
    "i am feeling sad and overwhelmed with exams",
    "what are the causes of hypertension",
    "can you write a python script",
    "hello how are you",
    "give me a breathing exercise",
    "i want to set a goal to meditate daily",
    "how can i manage stress"
]

for q in queries:
    msgs = [SystemMessage(content=INTENT_PROMPT), HumanMessage(content=f'User message: "{q}"\nIntent:')]
    resp = llm.invoke(msgs)
    print(f"'{q}' -> {resp.content.strip()}", flush=True)
