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


def is_physical_health_query(user_text: str) -> bool:
    """Detect whether user input describes a physical health symptom or medical inquiry."""
    lower = user_text.lower().strip()

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

    # 4. Specific stomach / abdominal pain and digestive symptoms
    if any(k in lower for k in [
        "stomach pain", "pain in my stomach", "pain in stomach", "stomach hurts",
        "stomach hurt", "stomach ache", "stomachache", "abdominal pain", "abdomen pain",
        "pain in my abdomen", "cramps in my stomach", "stomach cramps", "belly ache",
        "tummy hurt", "hurts in my stomach", "why does my stomach hurt", "what causes stomach",
        "stomach hurts because", "stomach hurts badly", "acid reflux", "heartburn", "indigestion",
        "gastritis", "dyspepsia", "cramping in my stomach"
    ]):
        return True

    # 5. General bodily pains, common symptoms, conditions, and medical queries
    if any(k in lower for k in [
        "headache", "migraine", "back pain", "chest pain", "joint pain", "neck pain",
        "throat hurt", "sore throat", "why does my throat hurt", "coughing", "cough",
        "coughing for three days", "been coughing", "nausea", "nauseous", "vomiting",
        "vomit", "fever", "diarrhea", "hypertension", "blood pressure", "insomnia",
        "dehydration", "what is dehydration", "iron deficiency", "anemia",
        "symptoms associated with iron deficiency", "causes of back pain",
        "causes of insomnia", "common causes of", "what is anxiety", "sleep hygiene",
        "causes nausea", "what causes nausea", "symptoms of migraine", "symptoms of hypertension",
        "symptoms of", "what is hypertension", "medical encyclopedia", "gale encyclopedia",
        "medical literature", "clinical causes", "what does medical"
    ]):
        return True

    # 6. General regex: "I have/am having [pain/ache/cramp] in my [body part]"
    if re.search(r"\b(?:have|having|feeling|got)\s+(?:pain|ache|aching|cramps?|discomfort|soreness)\s+in\s+my\s+\w+", lower):
        return True
    if re.search(r"\bmy\s+(?:stomach|abdomen|head|back|chest|neck|throat|knee|shoulder|gut|belly)\s+(?:hurts?|aches?|pains?)\b", lower):
        return True

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

    # PRIORITY 3: General Health Information & Physical Symptoms
    if is_physical_health_query(user_text):
        return "HEALTH_INFORMATION"

    # PRIORITY 5: Journaling requests
    if any(k in lower for k in [
        "/journal", "journal", "journaling", "journal entry", "write in my journal",
        "save to journal", "in my journal", "show my journal", "open my journal"
    ]):
        return "JOURNAL_REQUEST"

    # PRIORITY 6: Goal requests
    if lower.startswith("/goal") or "/goal" in lower or (
        ("goal" in lower or "goals" in lower) and any(
            act in lower for act in [
                "set", "create", "make", "add", "show", "view", "list", "mark",
                "complete", "finish", "my goal", "my goals", "study goal", "help me"
            ]
        )
    ):
        return "GOAL_REQUEST"

    # PRIORITY 6: Planning requests
    if any(k in lower for k in [
        "study plan", "study schedule", "plan my week", "plan my study", "plan my exam",
        "routine", "time management", "organize my day", "exam preparation plan",
        "make a plan", "plan for my exam", "plan for studying", "help me plan"
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

    # Arithmetic expressions (2+2, etc.) are handled as direct conversation
    if evaluate_math_expression(user_text) is not None:
        return "GENERAL_CONVERSATION"

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

    # Out of scope query check
    if is_out_of_scope(user_text):
        return "OUT_OF_SCOPE"

    return "GENERAL_CONVERSATION"


def is_out_of_scope(user_text: str) -> bool:
    """Detect whether user input is clearly outside the scope of mental wellness, study support, and medical guidance."""
    lower = user_text.lower().strip()
    if not lower:
        return False

    # If physical symptoms or medical query, it is IN scope
    if is_physical_health_query(user_text):
        return False

    # Simple arithmetic is handled directly, not rejected
    if evaluate_math_expression(user_text) is not None:
        return False

    # Standard wellness, mental health, emotional words, conversational check-ins, or tools are IN scope
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

    # Out of scope domain patterns
    out_of_scope_patterns = [
        # Coding & Software Development
        r"\b(?:python|javascript|typescript|c\+\+|java|html|css|sql|react|angular|vue|docker|kubernetes|git\s+push|api\s+endpoint|algorithm|debug\s+code|write\s+a\s+script|regex|function\s*\(|class\s+\w+:|import\s+\w+|write\s+code)\b",
        # Cryptocurrency, finance, trading, stock market
        r"\b(?:crypto|bitcoin|ethereum|dogecoin|stock\s+market|forex|day\s+trading|investing\s+in|mutual\s+fund|share\s+price|cryptocurrency|buy\s+stocks)\b",
        # Automotive, mechanical, electrical
        r"\b(?:car\s+engine|transmission|spark\s+plug|oil\s+change|tire\s+pressure|brake\s+pads|fix\s+my\s+car|cylinder\s+head)\b",
        # Advanced math calculation / complex algebra / calculus
        r"\b(?:solve\s+equation|integral\s+of|derivative\s+of|matrix\s+multiplication|eigenvalues|quadratic\s+formula|calculus)\b",
        # Video game cheats, betting, sports match scores
        r"\b(?:cheat\s+codes|gta\s+5\s+cheats|fifa\s+coins|fortnite\s+vbucks|cricket\s+score|nba\s+score|premier\s+league\s+table|betting\s+odds)\b",
        # Legal litigation
        r"\b(?:legal\s+contract|sue\s+my|lawsuit|attorney\s+fees|court\s+hearing)\b",
        # General non-medical trivia / geopolitics
        r"\b(?:capital of|election results|weather in|who is the president|stock price)\b",
    ]
    for pat in out_of_scope_patterns:
        if re.search(pat, lower):
            return True
    return False


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

    # Arithmetic expressions are routed directly
    if evaluate_math_expression(user_text) is not None:
        return "GENERAL_CONVERSATION"

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

