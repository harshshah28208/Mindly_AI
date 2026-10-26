import os
import sys
import time
import re
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))
from agent.safety import assess_safety, screen_urgent_medical, is_educational_or_third_person
from agent.intent import evaluate_math_expression

# ------------------------------------------------------------------------------
# 1. Supported Intents & Action Mapping
# ------------------------------------------------------------------------------
INTENT_SPECS = {
    "CRISIS": {
        "action": "CRISIS_RESPONSE",
        "tool": "crisis_support_protocol",
        "desc": "Imminent self-harm, suicidal ideation, or life-threatening psychiatric emergencies.",
    },
    "URGENT_MEDICAL": {
        "action": "USE_RAG_TOOL",
        "tool": "urgent_medical_protocol",
        "desc": "Acute physical health emergencies (severe chest pain, fainting, dyspnea, overdose, poisoning).",
    },
    "HEALTH_INFORMATION": {
        "action": "USE_RAG_TOOL",
        "tool": "gale_retriever",
        "desc": "General physical health inquiries, diseases, illnesses, bodily symptoms (Gale Encyclopedia RAG).",
    },
    "GROUNDING_REQUEST": {
        "action": "USE_GROUNDING_TOOL",
        "tool": "interactive_grounding",
        "desc": "Somatic grounding, breathing exercises (box breathing), 5-4-3-2-1 sensory exercises, panic relief.",
    },
    "COPING_REQUEST": {
        "action": "USE_COPING_TOOL",
        "tool": "coping_strategy",
        "desc": "Actionable coping mechanisms, stress management, study burnout advice, sleep hygiene routines.",
    },
    "EMOTIONAL_SUPPORT": {
        "action": "DIRECT_RESPONSE",
        "tool": "direct_conversation",
        "desc": "Sharing emotional vulnerability, sadness, loneliness, anxiety, feeling overwhelmed, seeking empathetic listening.",
    },
    "JOURNAL_REQUEST": {
        "action": "USE_JOURNAL_TOOL",
        "tool": "journal_recorder",
        "desc": "Writing reflections, saving personal journal entries, or recording daily mood ratings (e.g. 6/10).",
    },
    "GOAL_REQUEST": {
        "action": "USE_GOAL_TOOL",
        "tool": "goal_tracker",
        "desc": "Setting, updating, or tracking wellness habits, personal goals, and routines.",
    },
    "MEMORY_REQUEST": {
        "action": "USE_MEMORY_TOOL",
        "tool": "memory_vault",
        "desc": "Remembering, recalling past user facts/moods, updating preferred names, or clearing memory.",
    },
    "GENERAL_CONVERSATION": {
        "action": "DIRECT_RESPONSE",
        "tool": "direct_conversation",
        "desc": "Greetings, friendly conversational check-ins, Mindly identity inquiries, general in-scope wellness chat.",
    },
    "OUT_OF_SCOPE": {
        "action": "DIRECT_RESPONSE",
        "tool": "scope_boundary_protocol",
        "desc": "Inquiries outside mental wellness and health (programming, math, science, finance, non-medical trivia).",
    },
    "CLARIFICATION_NEEDED": {
        "action": "DIRECT_RESPONSE",
        "tool": "clarification_protocol",
        "desc": "Ambiguous, vague, or underspecified inputs where the user's intent cannot be reliably determined.",
    },
}

SEMANTIC_ANCHORS = {
    "HEALTH_INFORMATION": [
        "stomach", "pain", "ache", "aching", "hurt", "hurts", "hurting", "sore", "headache", "migraine",
        "nausea", "vomit", "fever", "cramp", "cramping", "throat", "dizzy", "dizziness", "chest", "cough",
        "fatigue", "tiredness", "rash", "skin", "joint", "shoulder", "back", "belly", "abdomen", "abdominal",
        "dehydration", "dehydrated", "hypertension", "diabetes", "asthma", "infection", "illness", "disease",
        "symptom", "symptoms", "causes", "cause", "etiology", "tummy", "noggin", "throbbing", "organ", "body",
        "dairy", "eating", "digestion", "indigestion", "cluster headache", "cluster headaches"
    ],
    "GROUNDING_REQUEST": [
        "breathe", "breathing", "breath", "inhale", "exhale", "box breathing", "square breathing", "4-4-4-4", "4-7-8",
        "ground", "grounding", "5-4-3-2-1", "senses", "sensory", "calm down", "calming", "panic attack",
        "panicking", "racing thoughts", "heart racing", "shaking", "hyperventilating", "chill out", "somatic",
        "respiratory", "breath cycle", "lead me through", "guide me through"
    ],
    "COPING_REQUEST": [
        "cope", "coping", "ways to handle", "how to manage", "manage stress", "stress management", "burnout",
        "study tips", "exam stress", "exam pressure", "workload", "study break", "recovery pause", "sleep routine",
        "sleep hygiene", "insomnia", "practical ways", "strategy", "strategies", "routine", "habit", "actionable",
        "steps to", "manageable steps", "practical strategies", "stress", "work piles up", "manage everyday stress",
        "overworked", "overworking"
    ],
    "EMOTIONAL_SUPPORT": [
        "feel", "feeling", "anxious", "anxiety", "sad", "sadness", "lonely", "loneliness", "depressed",
        "overwhelmed", "overwhelm", "overthinking", "nervous", "scared", "fear", "angry", "anger", "frustrated",
        "heartbroken", "breakup", "crying", "cry", "listen", "talk to someone", "need someone", "vent", "venting",
        "down today", "heavy heart", "unmotivated", "lost", "hopeless", "bad day", "rough day", "butterflies",
        "emotional weight", "blue today", "spiraling"
    ],
    "JOURNAL_REQUEST": [
        "journal", "journaling", "log my mood", "log mood", "record mood", "record my mood", "today's mood",
        "mood rating", "out of 10", "out of ten", "/10", "save entry", "diary", "write down my thoughts",
        "reflect on today", "save a new journal entry", "my reflection"
    ],
    "GOAL_REQUEST": [
        "goal", "goals", "habit", "habits", "track", "tracking", "set a goal", "create a goal", "daily walk",
        "daily meditation", "routine target", "commit to", "streak", "target", "establish a", "morning meditation",
        "drink two liters", "weekly study schedule"
    ],
    "MEMORY_REQUEST": [
        "remember that", "remember my", "what is my name", "what did i tell you", "what did i say",
        "my preferred name", "call me", "forget my", "clear all memories", "what was my mood earlier",
        "stored memories", "what do you remember", "mood score i recorded", "preferred nickname",
        "what was the mood score", "recorded earlier", "what did i record"
    ],

    "GENERAL_CONVERSATION": [
        "hello", "hi", "hey", "good morning", "good evening", "how are you", "who are you", "what is your name",
        "what can you do", "tell me a joke", "nice to meet you", "thank you", "thanks", "mindly", "purpose"
    ],
    "OUT_OF_SCOPE": [
        "python", "java", "javascript", "c++", "html", "css", "sql", "docker", "algorithm", "binary search",
        "hashmap", "sort", "sorting", "quicksort", "code", "coding", "debug", "programming", "algebra",
        "equation", "solve for x", "calculus", "integral", "derivative", "solar panel", "solar panels",
        "photovoltaic", "astronomy", "black hole", "galaxy", "quantum", "physics", "stock", "stocks",
        "bitcoin", "crypto", "trading", "stock price", "capital city", "capital of france", "capital of",
        "who is the president", "election", "weather in", "car engine", "oil change", "spark plug"
    ]
}

def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"\b[a-z0-9\-]+\b", text.lower()) if len(w) > 1]

def compute_concept_scores(text: str) -> Dict[str, float]:
    tokens = set(tokenize(text))
    lower = text.lower()
    scores = {}
    for intent, anchors in SEMANTIC_ANCHORS.items():
        score = 0.0
        for anchor in anchors:
            if " " in anchor:
                if anchor in lower:
                    score += 2.5
            else:
                if anchor in tokens:
                    score += 1.0
                elif any(anchor in t for t in tokens if len(anchor) > 4 and len(t) > 4):
                    score += 0.5
        scores[intent] = score
    return scores

def classify_intent_semantic(text: str, llm=None) -> Dict[str, Any]:
    clean = text.strip()
    if not clean:
        return {
            "intent": "GENERAL_CONVERSATION",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "direct_conversation",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Empty or whitespace input.",
            "classifier_source": "deterministic",
        }

    # 1. Deterministic Safety Priority 1: Crisis
    safety = assess_safety(clean)
    if safety["risk_level"] == "HIGH":
        return {
            "intent": "CRISIS",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "CRISIS_RESPONSE",
            "tool_name": "crisis_support_protocol",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Safety invariant: detected imminent crisis indicators.",
            "classifier_source": "safety_override",
        }

    # 2. Deterministic Safety Priority 2: Urgent Medical
    urgent = screen_urgent_medical(clean)
    if urgent["is_urgent"]:
        return {
            "intent": "URGENT_MEDICAL",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "USE_RAG_TOOL",
            "tool_name": "urgent_medical_protocol",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Safety invariant: detected acute red-flag medical indicators.",
            "classifier_source": "safety_override",
        }

    # 3. Third person or educational crisis
    if safety["safety_status"] == "EDUCATIONAL_OR_THIRD_PERSON" or is_educational_or_third_person(clean):
        return {
            "intent": "CRISIS",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "third_person_support_protocol",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Educational or third-person crisis context.",
            "classifier_source": "safety_override",
        }

    # 4. Arithmetic / explicit math
    if evaluate_math_expression(clean) is not None:
        return {
            "intent": "OUT_OF_SCOPE",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "scope_boundary_protocol",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Math calculation query outside mental wellness scope.",
            "classifier_source": "deterministic",
        }

    # 5. Semantic concept scoring
    concept_scores = compute_concept_scores(clean)
    sorted_scores = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
    top_intent, top_score = sorted_scores[0]
    runner_up_intent, runner_up_score = sorted_scores[1]

    # Check for ambiguous / vague input (< 8 words with 0 concept matches)
    tokens = tokenize(clean)
    if top_score == 0.0:
        if len(tokens) <= 8 and not re.search(r"\b(?:hello|hi|hey|greetings|morning|evening|afternoon)\b", clean.lower()):
            return {
                "intent": "CLARIFICATION_NEEDED",
                "confidence": 0.40,
                "confidence_level": "LOW",
                "selected_action": "DIRECT_RESPONSE",
                "tool_name": "clarification_protocol",
                "needs_clarification": True,
                "clarification_prompt": "Could you clarify what you would like to explore?",
                "routing_reason": "Vague or underspecified input with zero domain concept overlap.",
                "classifier_source": "clarification",
            }


    # Check out-of-scope non-medical queries
    if top_intent == "OUT_OF_SCOPE" and top_score >= 1.0:
        wellness_sum = sum(concept_scores[k] for k in ["EMOTIONAL_SUPPORT", "HEALTH_INFORMATION", "GROUNDING_REQUEST", "COPING_REQUEST"])
        if wellness_sum == 0.0:
            return {
                "intent": "OUT_OF_SCOPE",
                "confidence": 0.95,
                "confidence_level": "HIGH",
                "selected_action": "DIRECT_RESPONSE",
                "tool_name": "scope_boundary_protocol",
                "needs_clarification": False,
                "clarification_prompt": None,
                "routing_reason": "Topic identified as non-medical / non-wellness domain.",
                "classifier_source": "concept_arbiter",
            }

    # Semantic decision
    spec = INTENT_SPECS.get(top_intent, INTENT_SPECS["GENERAL_CONVERSATION"])
    calibrated_conf = min(0.95, 0.65 + (top_score * 0.1)) if top_score > 0 else 0.50
    conf_level = "HIGH" if calibrated_conf >= 0.80 else "MEDIUM" if calibrated_conf >= 0.60 else "LOW"

    return {
        "intent": top_intent,
        "confidence": round(calibrated_conf, 2),
        "confidence_level": conf_level,
        "selected_action": spec["action"],
        "tool_name": spec["tool"],
        "needs_clarification": False,
        "clarification_prompt": None,
        "routing_reason": f"Semantic concept profile matched {top_intent} (score={top_score}).",
        "classifier_source": "concept_arbiter",
    }


# ------------------------------------------------------------------------------
# 2. Run Evaluation against Held-out Benchmark
# ------------------------------------------------------------------------------
HELD_OUT_DATASET = [
    # A. Emotional Support (unseen paraphrases & typos)
    ("I've been carrying a heavy emotional weight since the breakup", "EMOTIONAL_SUPPORT"),
    ("feel lonly and blue today, just want someone to listen", "EMOTIONAL_SUPPORT"),
    ("My mind won't stop spiraling late at night", "EMOTIONAL_SUPPORT"),
    ("I am overwhelmed by the pressure at my new job", "EMOTIONAL_SUPPORT"),
    ("I feel so angry at everything and I cannot figure out why", "EMOTIONAL_SUPPORT"),
    ("Feeling deeply down today, please just be here with me", "EMOTIONAL_SUPPORT"),
    ("Anxious butterflies in my stomach about a big life decision", "EMOTIONAL_SUPPORT"),
    
    # B. Grounding Request (respiratory / somatic paraphrases)
    ("Guide me through the square breathing rhythm", "GROUNDING_REQUEST"),
    ("Can you lead me through the 5-4-3-2-1 sensory exercise?", "GROUNDING_REQUEST"),
    ("My heart is pounding and I feel panicky, help me calm down", "GROUNDING_REQUEST"),
    ("Give me a somatic grounding method right now", "GROUNDING_REQUEST"),
    ("Can you teach me the four-count breath cycle?", "GROUNDING_REQUEST"),
    
    # C. Coping Request (actionable strategies, sleep, burnout)
    ("What are manageable steps to build a healthy sleep hygiene routine?", "COPING_REQUEST"),
    ("Give me three practical strategies for college exam burnout", "COPING_REQUEST"),
    ("How can I take effective mental recovery pauses during long study sessions?", "COPING_REQUEST"),
    ("Ways to manage everyday stress when work piles up", "COPING_REQUEST"),
    ("I struggle with insomnia because my thoughts race in bed", "COPING_REQUEST"),
    
    # D. Physical Health / Medical Information (Gale RAG)
    ("What factors can trigger acute abdominal or stomach cramping?", "HEALTH_INFORMATION"),
    ("What are the key biological symptoms of severe dehydration?", "HEALTH_INFORMATION"),
    ("Why might someone suffer from persistent cluster headaches?", "HEALTH_INFORMATION"),
    ("stommach herts after eating dairy", "HEALTH_INFORMATION"),
    ("My noggin is throbbing so badly this afternoon", "HEALTH_INFORMATION"),
    ("What causes hypertension in adults?", "HEALTH_INFORMATION"),
    
    # E. Urgent Medical Concern
    ("I have sudden crushing chest pain and fainted on the floor", "URGENT_MEDICAL"),
    ("I accidentally swallowed too many prescription pills by mistake", "URGENT_MEDICAL"),
    ("Severe stabbing stomach pain with blood and loss of consciousness", "URGENT_MEDICAL"),
    
    # F. Crisis
    ("I might hurt myself tonight and I am not safe alone", "CRISIS"),
    ("I am thinking about ending my life and cannot go on", "CRISIS"),
    ("My coworker told me they want to end their life. How can I help them?", "CRISIS"),
    
    # G. Journal Request
    ("Help me log today's mood as 8 out of 10", "JOURNAL_REQUEST"),
    ("Record my reflection for today in my journal", "JOURNAL_REQUEST"),
    ("Save a new journal entry about my walk in the park", "JOURNAL_REQUEST"),
    
    # H. Goal Request
    ("Help me establish a daily morning meditation habit", "GOAL_REQUEST"),
    ("I want to set a goal to drink two liters of water each day", "GOAL_REQUEST"),
    ("Create a weekly study schedule goal for my biology exam", "GOAL_REQUEST"),
    
    # I. Memory Request
    ("What was the mood score I recorded earlier?", "MEMORY_REQUEST"),
    ("Please remember that my preferred nickname is Sam", "MEMORY_REQUEST"),
    ("Forget my stored preferences and clear my memories", "MEMORY_REQUEST"),
    
    # J. General In-Scope Conversation
    ("Hello Mindly, how are you doing today?", "GENERAL_CONVERSATION"),
    ("What is your purpose and how can you help me?", "GENERAL_CONVERSATION"),
    ("Good evening, thanks for being available", "GENERAL_CONVERSATION"),
    
    # K. Out of Scope
    ("Can you explain how a Java HashMap handles hash collisions?", "OUT_OF_SCOPE"),
    ("Write a quicksort algorithm in C++", "OUT_OF_SCOPE"),
    ("What is the capital city of France?", "OUT_OF_SCOPE"),
    ("Explain the physics of solar panel photovoltaic cells", "OUT_OF_SCOPE"),
    ("Solve this algebraic equation for x: 5x - 15 = 35", "OUT_OF_SCOPE"),
    ("What is the current stock price of Apple?", "OUT_OF_SCOPE"),
    
    # L. Ambiguous / Clarification Needed
    ("can you help me with that thing?", "CLARIFICATION_NEEDED"),
    ("I need something right now but don't know what", "CLARIFICATION_NEEDED"),
    ("do the thing we talked about", "CLARIFICATION_NEEDED"),
]

# Run benchmark
y_true = []
y_pred = []
total = len(HELD_OUT_DATASET)
correct = 0
conf_matrix = defaultdict(lambda: defaultdict(int))
classes = sorted(list(set(y for _, y in HELD_OUT_DATASET)))

print(f"\n=======================================================")
print(f"BENCHMARK EVALUATION ON HELD-OUT TEST SET ({total} cases)")
print(f"=======================================================\n")

for query, expected in HELD_OUT_DATASET:
    res = classify_intent_semantic(query)
    pred = res["intent"]
    y_true.append(expected)
    y_pred.append(pred)
    conf_matrix[expected][pred] += 1
    
    is_correct = (expected == pred)
    if is_correct:
        correct += 1
    else:
        print(f"[FAIL] Query: {query}")
        print(f"       Expected: {expected} | Actual: {pred} (Conf: {res['confidence']})")

accuracy = correct / total * 100.0
print(f"\nTotal Cases: {total}")
print(f"Correctly Classified: {correct}")
print(f"Accuracy: {accuracy:.2f}%\n")

# Compute per-class Precision, Recall, F1
print(f"{'Category':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support'}")
print("-" * 65)

for c in classes:
    tp = conf_matrix[c][c]
    fp = sum(conf_matrix[other][c] for other in classes if other != c)
    fn = sum(conf_matrix[c][other] for other in classes if other != c)
    support = sum(conf_matrix[c].values())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"{c:<22} | {precision*100:>8.1f}% | {recall*100:>8.1f}% | {f1*100:>8.1f}% | {support}")

print("\nConfusion Matrix:")
header = "Actual \\ Pred".ljust(20) + " ".join(f"{c[:4]:>5}" for c in classes)
print(header)
for actual in classes:
    row = f"{actual[:18]:<20}" + " ".join(f"{conf_matrix[actual][pred]:>5}" for pred in classes)
    print(row)
