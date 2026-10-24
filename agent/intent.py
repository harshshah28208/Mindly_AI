"""Intent classification agent node for Mindly.

Classifies user messages into standardized conversational, educational, wellness, and medical intents:
- GENERAL_CONVERSATION
- EMOTIONAL_SUPPORT
- GROUNDING_REQUEST
- COPING_REQUEST
- HEALTH_INFORMATION (physical symptoms, medical inquiries -> Gale RAG)
- INFORMATION_REQUEST (compatible alias for HEALTH_INFORMATION)
- JOURNAL_REQUEST
- GOAL_REQUEST
- MEMORY_REQUEST
- PLANNING_REQUEST
- CRISIS_RELATED
- OUT_OF_SCOPE (unrelated technical, financial, or non-medical queries)
"""

import ast
import operator
import re
from typing import Any, Dict, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from agent.safety import assess_safety, screen_urgent_medical, is_educational_or_third_person

MATH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def evaluate_math_expression(text: str) -> Optional[Dict[str, Any]]:
    """Safely parse and evaluate basic arithmetic expressions without eval."""
    clean = (text or "").strip()
    if not clean:
        return None
    m = re.match(r"^(?:what(?:\'s|\s+is)?\s+|calculate\s+|solve\s+)?([0-9\.\s\+\-\*\/\(\)\^\%]+)\??$", clean, re.IGNORECASE)
    if m:
        expr = m.group(1).strip()
        if any(op in expr for op in ["+", "-", "*", "/", "^", "%"]) and re.search(r"\d", expr):
            try:
                python_expr = expr.replace("^", "**")
                tree = ast.parse(python_expr, mode="eval")

                def _eval(node):
                    if isinstance(node, ast.Expression):
                        return _eval(node.body)
                    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                        return node.value
                    elif isinstance(node, ast.BinOp) and type(node.op) in MATH_OPS:
                        left = _eval(node.left)
                        right = _eval(node.right)
                        return MATH_OPS[type(node.op)](left, right)
                    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                        val = _eval(node.operand)
                        return -val if isinstance(node.op, ast.USub) else val
                    raise ValueError("Unsafe node")

                ans = _eval(tree)
                if isinstance(ans, float) and ans.is_integer():
                    ans = int(ans)
                return {"expr": expr, "ans": ans}
            except Exception:
                return None
    return None



INTENT_CATEGORIES = [
    "GENERAL_CONVERSATION",
    "EMOTIONAL_SUPPORT",
    "GROUNDING_REQUEST",
    "COPING_REQUEST",
    "HEALTH_INFORMATION",
    "URGENT_HEALTH",
    "CRISIS_RELATED",
    "JOURNAL_REQUEST",
    "GOAL_REQUEST",
    "PLANNING_REQUEST",
    "INFORMATION_REQUEST",  # Maintained as compatible alias
    "MEMORY_REQUEST",
    "OUT_OF_SCOPE",
]


def is_math_query(text: str) -> bool:
    """Detect whether user input is an arithmetic computation or math query."""
    clean = (text or "").strip().lower()
    if not clean:
        return False
    if evaluate_math_expression(clean) is not None:
        return True
    if re.search(r"\b(?:calculate|calc|math|arithmetic|solve\s+equation|evaluate)\b", clean) and re.search(r"[\d\+\-\*\/\^\=]", clean):
        return True
    if any(q in clean for q in ["can you do math", "calculate 2", "what is 2", "can you calculate", "solve this equation", "how much is 2"]):
        return True
    return False


def is_physical_health_query(user_text: str) -> bool:
    """Detect whether user input describes a physical health symptom, disease, or medical inquiry."""
    lower = user_text.lower().strip()
    if not lower:
        return False

    # 1. Exclude idiomatic or purely psychological metaphors
    if any(m in lower for m in [
        "butterflies in my stomach",
        "butterflies in the stomach",
        "butterflies in your stomach",
        "pains me to say",
        "heartbroken",
        "heart breaks",
        "pain in the neck" if not ("my neck" in lower or "neck hurts" in lower) else "",
    ]):
        return False

    # 2. Exclude somatic anxiety sensations tied specifically to exams/nervousness
    if ("feels weird" in lower or "feel weird" in lower or "weird feeling" in lower) and any(
        ctx in lower for ctx in ["exam", "test", "before exams", "anxious", "nervous"]
    ):
        return False

    # 3. Check for explicit grounding requests first (user is explicitly seeking a calming exercise)
    if any(k in lower for k in [
        "want a breathing exercise", "need a breathing exercise", "give me a grounding",
        "need a 5-4-3-2-1", "box breathing", "guide me to breathe", "help me calm down",
        "guide me through a breathing", "give me a breathing"
    ]):
        return False

    # 4. Dynamic Action / Cure / Remedy / Treatment inquiry patterns
    if re.search(
        r"\b(?:what\s+(?:to\s+do|can\s+i\s+do|should\s+i\s+(?:do|take|eat|drink))\s+for|how\s+to\s+(?:cure|treat|heal|relieve|reduce|stop|manage|handle)|how\s+can\s+i\s+(?:cure|treat|relieve|stop)|(?:cure|treatment|remedy|remedies|home\s+remedy|home\s+remedies|medicine|medication)\s+for|how\s+to\s+get\s+rid\s+of)\b",
        lower,
    ):
        return True

    # 5. Dynamic Reason, Etiology, and Cause inquiry patterns
    if re.search(
        r"\b(?:what\s+is\s+the\s+reason\s+(?:for|behind)|why\s+do\s+i\s+have|why\s+am\s+i\s+having|what\s+causes?|reasons?\s+for|causes?\s+of)\b",
        lower,
    ):
        return True

    # 6. Dynamic Symptom & Bodily Sensation complaints ("I am having acidity", "I have gas", etc.)
    if re.search(
        r"\b(?:having|have|got|feeling|suffering\s+from|experiencing|struggling\s+with)\s+(?:an?\s+)?(?:acute\s+|chronic\s+|mild\s+|severe\s+)?(?:acid(?:ity)?|heartburn|acid\s+reflux|gerd|gas|bloat(?:ing)?|indigestion|dyspepsia|nausea|vomit(?:ing)?|cramp(?:s|ing)?|burn(?:ing)?|pain(?:ing)?|ache|aching|fever|chills?|shivering|cough(?:ing)?|cold|congestion|sore\s+throat|diarrhea|constipation|dizz(?:y|iness)|vertigo|migraine|headache|weakness|fatigue|palpitation(?:s)?|rash(?:es)?|swelling|infection|stiffness|spasm(?:s)?|discomfort|sleeplessness|insomnia|itch(?:ing)?|shortness\s+of\s+breath)\b",
        lower,
    ):
        return True

    # 7. Dynamic Anatomical complaints ("my stomach is burning", "chest tightness", etc.)
    if re.search(
        r"\b(?:my\s+)?(?:stomach|abdomen|gut|belly|tummy|chest|head|throat|back|neck|shoulder|knee|joints?|eyes?|ears?|lungs?|bowels?|muscles?|skin|esophagus)\s+(?:hurts?|aches?|pains?|is\s+burning|is\s+upset|is\s+aching|is\s+cramping|is\s+swollen|is\s+stiff|is\s+paining|is\s+tight)\b",
        lower,
    ):
        return True
    if re.search(
        r"\b(?:burning|pain|cramp(?:s|ing)?|ache|aching|tightness|heaviness|discomfort|soreness)\s+in\s+(?:my\s+)?(?:stomach|abdomen|gut|belly|tummy|chest|head|throat|back|neck|shoulder|knee|joints?|eyes?|ears?|lungs?|bowels?|muscles?|body)\b",
        lower,
    ):
        return True

    # 8. Clinical disease suffixes (-itis, -algia, -emia, -pathy, -osis)
    if re.search(r"\b\w+(?:itis|algia|emia|opathy|osis)\b", lower):
        return True

    # 9. Specific diseases, clinical conditions, disorders, and pathology
    if any(k in lower for k in [
        "acidity", "acid reflux", "gerd", "heartburn", "indigestion", "gastritis", "dyspepsia",
        "hypertension", "high blood pressure", "low blood pressure", "hypotension",
        "migraine", "headache", "cluster headache", "tension headache",
        "diabetes", "diabetic", "blood sugar", "glucose", "insulin",
        "asthma", "bronchitis", "pneumonia", "wheezing", "respiratory", "lung", "lungs",
        "stomach pain", "abdominal pain", "pain in my stomach", "pain in stomach", "stomach hurts",
        "stomach hurt", "stomach ache", "stomachache", "belly ache", "tummy hurt",
        "ulcer", "peptic ulcer", "irritable bowel", "ibs", "colitis", "bloating",
        "insomnia", "sleep disorder", "sleep apnea", "restless legs", "circadian",
        "arthritis", "osteoarthritis", "rheumatoid", "joint pain", "back pain", "neck pain", "gout",
        "fever", "cough", "coughing", "sore throat", "throat hurt", "cold", "flu", "influenza",
        "infection", "viral infection", "bacterial infection", "strep",
        "anemia", "iron deficiency", "dehydration", "electrolyte",
        "thyroid", "hypothyroidism", "hyperthyroidism", "goiter",
        "heart disease", "cardiac", "chest pain", "angina", "palpitations", "arrhythmia", "cholesterol",
        "kidney", "kidney stone", "kidney stones", "uti", "urinary tract",
        "allergy", "allergies", "allergic", "eczema", "dermatitis", "hives", "rash",
        "depression", "clinical depression", "depressive disorder",
        "cancer", "tumor", "tumor cells", "biopsy", "oncology",
        "nausea", "nauseous", "vomiting", "vomit", "diarrhea", "constipation", "vertigo"
    ]):
        return True

    # 10. General medical and clinical vocabulary
    if any(k in lower for k in [
        "disease", "diseases", "disorder", "disorders", "syndrome", "symptom", "symptoms",
        "diagnosis", "prognosis", "etiology", "pathology", "clinical causes", "clinical management",
        "medical encyclopedia", "gale encyclopedia", "medical literature", "medical advice",
        "how to treat", "treatment for", "cure for", "medicine for", "management of",
        "goals of treatment", "therapeutic management", "clinical therapies", "remedies for"
    ]):
        return True

    return False


def is_out_of_scope(user_text: str) -> bool:
    """Detect whether user input is clearly outside the scope of mental wellness, health, and medical guidance."""
    lower = user_text.lower().strip()
    if not lower:
        return False

    # 1. If physical health or disease query, it is strictly IN SCOPE
    if is_physical_health_query(user_text):
        return False

    # 2. Math queries and arithmetic are strictly OUT_OF_SCOPE
    if is_math_query(user_text):
        return True

    # 3. Explicit out-of-scope domain patterns (checked before general wellness words to prevent "help" collisions)
    out_of_scope_patterns = [
        # Coding & Software Development
        r"\b(?:python|javascript|typescript|c\+\+|java|html|css|sql|react|angular|vue|docker|kubernetes|git\s+push|api\s+endpoint|algorithm|debug\s+code|write\s+a\s+script|regex|function\s*\(|class\s+\w+:|import\s+\w+|write\s+code|write\s+a\s+program|program\s+in|binary\s+search)\b",
        # Cryptocurrency, finance, trading, stock market
        r"\b(?:crypto|bitcoin|ethereum|dogecoin|stock\s+market|forex|day\s+trading|investing\s+in|mutual\s+fund|share\s+price|cryptocurrency|buy\s+stocks|stock\s+price)\b",
        # Automotive, mechanical, electrical
        r"\b(?:car\s+engine|transmission|spark\s+plug|oil\s+change|tire\s+pressure|brake\s+pads|fix\s+my\s+car|cylinder\s+head|fix\s+car)\b",
        # Advanced math calculation / complex algebra / calculus
        r"\b(?:solve\s+equation|integral\s+of|derivative\s+of|matrix\s+multiplication|eigenvalues|quadratic\s+formula|calculus|can\s+you\s+do\s+math)\b",
        # Video game cheats, betting, sports match scores
        r"\b(?:cheat\s+codes|gta\s+5\s+cheats|fifa\s+coins|fortnite\s+vbucks|cricket\s+score|nba\s+score|premier\s+league\s+table|betting\s+odds|who\s+won\s+the\s+(?:match|game|cricket|football))\b",
        # Legal litigation
        r"\b(?:legal\s+contract|sue\s+my|lawsuit|attorney\s+fees|court\s+hearing)\b",
        # General non-medical trivia / geopolitics
        r"\b(?:capital\s+of|election\s+results|weather\s+in|who\s+is\s+the\s+president|president\s+of\s+the)\b",
    ]
    for pat in out_of_scope_patterns:
        if re.search(pat, lower):
            return True

    # 4. Standard wellness, mental health, emotional words, conversational check-ins, or tools are IN scope
    wellness_keywords = [
        "stress", "anxious", "anxiety", "feel", "feeling", "sad", "happy", "overwhelm",
        "study", "exam", "college", "school", "breath", "breathing", "ground", "grounding",
        "cope", "coping", "journal", "goal", "remember", "mindly", "hello", "hi", "hey",
        "who are you", "what is your name", "what's your name", "how are you", "help",
        "thank", "tired", "sleep", "insomnia", "headache", "pain", "hurt", "ache",
        "medicine", "gale", "calm", "relax", "habit", "routine", "burnout", "lonely",
        "scared", "fear", "panic", "worry", "worried", "crying", "depressed", "mental",
        "health", "emotion", "thoughts", "unpack", "listen", "friend", "advice", "favorite"
    ]
    if any(k in lower for k in wellness_keywords):
        return False

    return False


def classify_intent_heuristic(user_text: str) -> str:
    """Heuristic rule-based intent classification adhering to hierarchical priorities (0-9)."""
    clean = user_text.strip()
    if not clean:
        return "GENERAL_CONVERSATION"

    lower = clean.lower()

    # PRIORITY 1: High-Risk Safety Guard Check (must override normal routing)
    safety = assess_safety(user_text)
    if safety["risk_level"] == "HIGH":
        return "CRISIS_RELATED"

    # PRIORITY 2: Urgent Medical / Acute Health Symptoms
    urgent = screen_urgent_medical(user_text)
    if urgent["is_urgent"]:
        return "URGENT_HEALTH"

    # PRIORITY 4: Crisis-related but educational or third-person (Non-immediate)
    if safety["safety_status"] == "EDUCATIONAL_OR_THIRD_PERSON":
        return "CRISIS_RELATED"

    # PRIORITY 3: General Health Information, Physical Symptoms & Disease Inquiries
    if is_physical_health_query(user_text):
        return "HEALTH_INFORMATION"

    # OUT-OF-SCOPE / MATH DOMAIN BOUNDARY CHECK (Must intercept non-medical queries before tools)
    if is_math_query(user_text) or is_out_of_scope(user_text):
        return "OUT_OF_SCOPE"

    # PRIORITY 5: Journaling requests
    if any(k in lower for k in [
        "/journal", "journal", "journaling", "journal entry", "write in my journal",
        "save to journal", "in my journal", "show my journal", "open my journal"
    ]):
        return "JOURNAL_REQUEST"

    # PRIORITY 6: Goal requests (NEVER for medical queries or disease treatments)
    if not is_physical_health_query(user_text):
        if lower.startswith("/goal") or (
            ("goal" in lower or "goals" in lower) and any(
                act in lower for act in [
                    "set", "create", "make", "add", "show", "view", "list", "mark",
                    "complete", "finish", "my goal", "my goals", "study goal", "help me"
                ]
            ) and not any(w in lower for w in [
                "disease", "treatment", "medicine", "doctor", "health", "cure", "pain",
                "symptom", "causes", "hypertension", "diabetes", "migraine", "asthma"
            ])
        ):
            return "GOAL_REQUEST"

    # PRIORITY 6: Planning requests
    if any(k in lower for k in [
        "study plan", "study schedule", "plan my week", "plan my study", "plan my exam",
        "routine", "time management", "organize my day", "exam preparation plan",
        "make a plan for my exam", "plan for studying", "help me plan my study schedule"
    ]):
        return "PLANNING_REQUEST"

    # PRIORITY 7: Explicit Grounding Exercise Requests
    if any(k in lower for k in [
        "want a breathing exercise", "give me a grounding", "breathing exercise", "breathwork",
        "box breathing", "help me breathe", "5-4-3-2-1", "senses grounding", "help me calm down",
        "need help calming down", "help calming down", "calm down", "guide me through breathing",
        "guide me through 5-4-3-2-1", "need a grounding exercise"
    ]):
        return "GROUNDING_REQUEST"

    # PRIORITY 8: Coping Requests
    if any(k in lower for k in [
        "cope", "coping", "deal with", "manage stress", "manage college", "manage pressure",
        "handle stress", "ways to manage", "ways to handle", "tips for stress", "advice for burnout",
        "overwhelmed with college", "overwhelmed with school", "overwhelmed with work",
        "how can i handle", "how do i handle", "how to handle", "what can i do to",
        "what should i do", "what can i do", "how to deal with", "give me ways"
    ]):
        return "COPING_REQUEST"

    # PRIORITY 8 (Cont): General Emotional Support
    if any(k in lower for k in [
        "stress", "stressed", "anxious", "anxiety", "sad", "depressed", "lonely",
        "overwhelm", "overwhelmed", "exhausted", "burnout", "scared", "worried",
        "crying", "upset", "angry", "feeling down", "hard day", "failing",
        "frustrated", "frustration", "don't know what to do", "feeling lost",
        "butterflies in my stomach", "butterflies", "bad day", "terrible day",
        "someone to talk to"
    ]):
        return "EMOTIONAL_SUPPORT"

    # Memory vault requests
    if any(k in lower for k in ["remember that", "what name should you call me", "what should you call me", "what is my preferred name", "view memory"]):
        return "MEMORY_REQUEST"

    # PRIORITY 9: General Conversation (Greetings, Identity, Casual)
    if any(k in lower for k in [
        "what is your name", "what's your name", "who are you",
        "tell me what you can do", "what can you do", "who made you",
        "who created you", "how are you", "hello", "hi mindly",
        "hey mindly", "good morning", "good evening", "good afternoon",
        "nice to meet you", "tell me a joke", "tell me something interesting",
        "thank you", "thanks"
    ]):
        return "GENERAL_CONVERSATION"

    return "GENERAL_CONVERSATION"


INTENT_PROMPT = """You are an intent classification component for Mindly, a mental wellness support assistant.
Classify the user's message into EXACTLY ONE of the following categories:
- GENERAL_CONVERSATION (greetings, casual conversation, questions about bot identity or capabilities)
- HEALTH_INFORMATION (physical symptoms, stomach pain, headaches, back pain, nausea, fever, cough, medical questions about conditions like hypertension, migraine, insomnia, dehydration)
- URGENT_HEALTH (potentially acute or life-threatening symptoms such as severe worsening pain, fainting, difficulty breathing, chest pain, vomiting blood, rigid abdomen)
- EMOTIONAL_SUPPORT (sharing emotional distress, sadness, loneliness, academic stress, feeling overwhelmed, nervous butterflies)
- GROUNDING_REQUEST (explicitly asking for help calming down right now, breathing exercises, 5-4-3-2-1 exercises)
- COPING_REQUEST (asking for practical strategies or methods to manage problems/stress/exams)
- JOURNAL_REQUEST (asking to record or save thoughts to the reflection journal)
- GOAL_REQUEST (asking to create, view, or complete goals and habits)
- PLANNING_REQUEST (asking for help organizing or study schedule routines)
- MEMORY_REQUEST (asking to remember a personal preference or asking what name to call the user)
- CRISIS_RELATED (mentions of self-harm, suicidal ideation, or acute danger)
- OUT_OF_SCOPE (questions completely unrelated to wellness, mental health, student wellbeing, or medical guidance, such as coding, stock market, sports scores, mechanical repair)

Important rule: If the message describes a physical health symptom (such as stomach pain, headache, nausea, back pain) even if stress is mentioned, classify it as HEALTH_INFORMATION.
If acute or severe red-flag symptoms are present, classify it as URGENT_HEALTH.
If the query is clearly an out-of-scope non-wellness/non-medical inquiry, classify it as OUT_OF_SCOPE.

Respond with ONLY the exact category name. Nothing else.
"""


def classify_intent(user_text: str, llm=None) -> str:
    """Classify user intent using hierarchical two-stage pipeline:
    Stage 1: Deterministic safety and priority rule filters.
    Stage 2: LLM classifier for ambiguous phrasing (constrained by safety).
    """
    clean = user_text.strip()
    if not clean:
        return "GENERAL_CONVERSATION"

    # Stage 1: Deterministic safety guard check (HIGH risk CANNOT be overridden by LLM)
    safety = assess_safety(user_text)
    if safety["risk_level"] == "HIGH":
        return "CRISIS_RELATED"

    # Stage 1 (Cont): Acute / Urgent Medical Symptoms Check
    urgent = screen_urgent_medical(user_text)
    if urgent["is_urgent"]:
        return "URGENT_HEALTH"

    # Stage 1 (Cont): Educational or Third-Person Crisis discussion
    if safety["safety_status"] == "EDUCATIONAL_OR_THIRD_PERSON":
        return "CRISIS_RELATED"

    # Stage 1 (Cont): Physical Health Symptoms / Conditions (takes priority over general emotional words)
    if is_physical_health_query(user_text):
        return "HEALTH_INFORMATION"

    # Arithmetic expressions (2+3, 2+2, etc.) are strictly OUT_OF_SCOPE
    if evaluate_math_expression(user_text) is not None:
        return "OUT_OF_SCOPE"

    # Check for clearly out-of-scope / unnecessary queries before passing to LLM
    if is_out_of_scope(user_text):
        return "OUT_OF_SCOPE"


    # Stage 1 (Cont): Explicit Tool Triggers & Conversational Filters
    heuristic_intent = classify_intent_heuristic(user_text)
    if heuristic_intent in [
        "JOURNAL_REQUEST", "GOAL_REQUEST", "PLANNING_REQUEST",
        "MEMORY_REQUEST", "GROUNDING_REQUEST", "COPING_REQUEST",
        "HEALTH_INFORMATION", "URGENT_HEALTH", "CRISIS_RELATED",
        "OUT_OF_SCOPE",
    ]:
        return heuristic_intent

    # Stage 2: LLM Classifier for Nuanced / Ambiguous Messages
    if llm is not None:
        try:
            messages = [
                SystemMessage(content=INTENT_PROMPT),
                HumanMessage(content=f"User message: \"{user_text}\"\nIntent:"),
            ]
            response = llm.invoke(messages)
            raw_category = str(response.content).strip().upper()
            for cat in INTENT_CATEGORIES:
                if cat in raw_category:
                    # Safety invariant: LLM can never downgrade high risk
                    if cat != "CRISIS_RELATED" and safety["risk_level"] == "HIGH":
                        return "CRISIS_RELATED"
                    return cat
        except Exception:
            pass

    return heuristic_intent

