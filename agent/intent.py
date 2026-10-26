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
from typing import Any, Dict, List, Optional, Tuple, TypedDict
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
    "THERAPIST_REQUEST",
    "OUT_OF_SCOPE",
]


def is_math_query(text: str) -> bool:
    """Detect whether user input is an arithmetic computation or math query."""
    clean = (text or "").strip().lower()
    if not clean:
        return False
    if evaluate_math_expression(clean) is not None:
        return True
    if re.search(r"\b(?:calculate|calc|math|arithmetic|algebra|equation|evaluate)\b", clean) and re.search(r"[\d\+\-\*\/\^\=\w]", clean):
        if any(w in clean for w in ["solve", "equation", "algebra", "calculate", "derivative", "integral", "x", "="]):
            return True
    if any(q in clean for q in ["can you do math", "calculate 2", "what is 2", "can you calculate", "solve this equation", "how much is 2"]):
        return True
    return False


def is_physical_health_query(user_text: str) -> bool:
    """Detect whether user input describes a physical health symptom, disease, or medical inquiry."""
    lower = user_text.lower().strip()
    if not lower:
        return False

    # 1. Strictly exclude emotional, psychological, and relational topics from physical health RAG
    if any(k in lower for k in [
        "jealous", "jealousy", "envy", "overthink", "overthinking",
        "procrastinat", "breakup", "heartbreak", "heart broken", "friendship",
        "relationship", "lonely", "loneliness", "feeling down", "feeling sad",
        "cry", "crying", "guilt", "shame", "imposter", "boredom",
        "butterflies in my stomach", "butterflies in the stomach", "pains me to say",
    ]):
        return False

    # 2. Strictly exclude coding, technical, and math topics
    if any(k in lower for k in [
        "python", "script", "beautifulsoup", "code", "coding", "software",
        "javascript", "html", "css", "database", "sql", "api", "function"
    ]):
        return False

    # 3. Exclude somatic anxiety sensations tied specifically to exams/nervousness
    if ("feels weird" in lower or "feel weird" in lower or "weird feeling" in lower) and any(
        ctx in lower for ctx in ["exam", "test", "before exams", "anxious", "nervous"]
    ):
        return False

    # 4. Check for explicit grounding requests first (user is explicitly seeking a calming exercise)
    if any(k in lower for k in [
        "want a breathing exercise", "need a breathing exercise", "give me a grounding",
        "need a 5-4-3-2-1", "box breathing", "guide me to breathe", "help me calm down",
        "guide me through a breathing", "give me a breathing"
    ]):
        return False

    # 5. Exclude generic pronoun continuations without physical symptoms
    if re.search(r"\b(?:what\s+(?:can\s+i\s+do|to\s+do|should\s+i\s+do)\s+for\s+(?:it|this|that))\b", lower) and not any(
        k in lower for k in ["pain", "ache", "fever", "cough", "migraine", "stomach", "headache", "throat", "injury", "sprain", "rash", "acidity", "nausea", "acne", "pimple"]
    ):
        return False

    # 6. Specific physical conditions, diseases, dermatological, and somatic pathology (word boundaries)
    physical_terms = [
        r"\bacne\b", r"\bacnes\b", r"\bpimple\b", r"\bpimples\b", r"\bblackhead\b", r"\bwhitehead\b", r"\bfolliculitis\b",
        r"\bacidity\b", r"\bacid\s+reflux\b", r"\bgerd\b", r"\bheartburn\b", r"\bindigestion\b", r"\bgastritis\b", r"\bdyspepsia\b",
        r"\bhypertension\b", r"\bhigh\s+blood\s+pressure\b", r"\blow\s+blood\s+pressure\b", r"\bhypotension\b",
        r"\bmigraine\b", r"\bheadache\b", r"\bcluster\s+headache\b", r"\btension\s+headache\b",
        r"\b(?:noggin|head)\s+(?:is\s+)?pounding\b", r"\bpounding\s+(?:head|noggin)\b", r"\bhead\s+hurts?\b",
        r"\bdiabetes\b", r"\bdiabetic\b", r"\bblood\s+sugar\b", r"\bglucose\b", r"\binsulin\b",
        r"\basthma\b", r"\bbronchitis\b", r"\bpneumonia\b", r"\bwheezing\b", r"\brespiratory\b", r"\blungs?\b",
        r"\bstomach\s+pain\b", r"\babdominal\s+pain\b", r"\bpain\s+in\s+(?:my\s+)?stomach\b", r"\bstomach\s+hurts?\b",
        r"\bstomach\s+ache\b", r"\bstomachache\b", r"\bbelly\s+ache\b", r"\btummy\s+hurt\b", r"\bstommach\s+herts?\b", r"\bstommach\b",
        r"\bulcer\b", r"\bpeptic\s+ulcer\b", r"\birritable\s+bowel\b", r"\bibs\b", r"\bcolitis\b", r"\bbloating\b",
        r"\binsomnia\b", r"\bsleep\s+disorder\b", r"\bsleep\s+apnea\b", r"\brestless\s+legs\b", r"\bcircadian\b",
        r"\barthritis\b", r"\bosteoarthritis\b", r"\brheumatoid\b", r"\bjoint\s+pain\b", r"\bback\s+pain\b", r"\bneck\s+pain\b", r"\bgout\b",
        r"\bfever\b", r"\bcough\b", r"\bcoughing\b", r"\bsore\s+throat\b", r"\bthroat\s+hurt\b", r"\bcold\b", r"\bflu\b", r"\binfluenza\b",
        r"\binfection\b", r"\bviral\s+infection\b", r"\bbacterial\s+infection\b", r"\bstrep\b",
        r"\banemia\b", r"\biron\s+deficiency\b", r"\bdehydration\b", r"\belectrolyte\b",
        r"\bthyroid\b", r"\bhypothyroidism\b", r"\bhyperthyroidism\b", r"\bgoiter\b",
        r"\bheart\s+disease\b", r"\bcardiac\b", r"\bchest\s+pain\b", r"\bangina\b", r"\bpalpitations?\b", r"\barrhythmia\b", r"\bcholesterol\b",
        r"\bkidney\b", r"\bkidney\s+stones?\b", r"\buti\b", r"\burinary\s+tract\b",
        r"\ballergy\b", r"\ballergies\b", r"\ballergic\b", r"\beczema\b", r"\bdermatitis\b", r"\bhives\b", r"\brash\b",
        r"\bcancer\b", r"\btumor\b", r"\btumor\s+cells\b", r"\bbiopsy\b", r"\boncology\b",
        r"\bnausea\b", r"\bnauseous\b", r"\bvomiting\b", r"\bvomit\b", r"\bdiarrhea\b", r"\bconstipation\b", r"\bvertigo\b"
    ]
    for pat in physical_terms:
        if re.search(pat, lower):
            return True

    # 7. Dynamic Action / Cure / Treatment inquiry ONLY if associated with physical body or condition
    has_cure_action = bool(re.search(
        r"\b(?:what\s+(?:to\s+do|can\s+i\s+do|should\s+i\s+(?:do|take|eat|drink))\s+for|how\s+to\s+(?:cure|treat|heal|relieve|reduce|stop|manage|handle)|how\s+can\s+i\s+(?:cure|treat|relieve|stop)|(?:cure|treatment|remedy|remedies|home\s+remedy|home\s+remedies|medicine|medication)\s+for|how\s+to\s+get\s+rid\s+of)\b",
        lower,
    ))
    has_body_or_symptom = bool(re.search(
        r"\b(?:acne|acnes|pimple|pimples|skin|face|back|spine|neck|shoulder|knee|joint|stomach|gut|belly|chest|head|throat|ear|eye|pain|ache|cough|fever|burn|rash|cramp|swelling|bleeding|nausea|reflux|gas)\b",
        lower,
    ))
    if has_cure_action and has_body_or_symptom:
        return True

    # 8. Dynamic Reason, Etiology, and Cause inquiry patterns associated with physical conditions
    has_cause_inquiry = bool(re.search(
        r"\b(?:what\s+is\s+the\s+reason\s+(?:for|behind)|why\s+do\s+i\s+have|why\s+am\s+i\s+having|what\s+causes?|reasons?\s+for|causes?\s+of)\b",
        lower,
    ))
    if has_cause_inquiry and has_body_or_symptom:
        return True

    # 9. Dynamic Symptom & Bodily Sensation complaints ("I am having acidity", "my back hurts", etc.)
    if re.search(
        r"\b(?:having|have|got|suffering\s+from|experiencing)\s+(?:an?\s+)?(?:acute\s+|chronic\s+|mild\s+|severe\s+)?(?:acid(?:ity)?|heartburn|acid\s+reflux|gerd|gas|bloat(?:ing)?|indigestion|dyspepsia|nausea|vomit(?:ing)?|cramp(?:s|ing)?|burn(?:ing)?|pain(?:ing)?|ache|aching|fever|chills?|cough(?:ing)?|cold|sore\s+throat|diarrhea|constipation|dizz(?:y|iness)|vertigo|migraine|headache|palpitation(?:s)?|rash(?:es)?|swelling|acne|pimples?)\b",
        lower,
    ):
        return True

    # 10. Medical or clinical literature inquiries
    if re.search(r"\b(?:medical\s+literature|clinical\s+literature|clinical\s+causes?)\b", lower):
        return True

    # 10. Dynamic Anatomical complaints ("my back hurts", "burning in my chest", etc.)
    if re.search(
        r"\b(?:my\s+)?(?:stomach|abdomen|gut|belly|tummy|chest|head|throat|back|neck|shoulder|knee|joints?|eyes?|ears?|lungs?|bowels?|muscles?|skin|esophagus)\s+(?:hurts?|aches?|pains?|is\s+burning|is\s+upset|is\s+aching|is\s+cramping|is\s+swollen|is\s+stiff|is\s+paining|is\s+tight)\b",
        lower,
    ):
        return True

    # 11. Clinical disease suffixes (-itis, -algia, -emia, -pathy)
    if re.search(r"\b\w+(?:itis|algia|emia|opathy)\b", lower):
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
        r"\b(?:algebra|algebraic|equation|solve\s+(?:\w+\s+)?equation|integral\s+of|derivative\s+of|matrix\s+multiplication|eigenvalues|quadratic\s+formula|calculus|can\s+you\s+do\s+math)\b",
        # Video game cheats, betting, sports match scores
        r"\b(?:cheat\s+codes|gta\s+5\s+cheats|fifa|fifa\s+coins|fortnite\s+vbucks|cricket\s+score|nba\s+score|premier\s+league\s+table|betting\s+odds|who\s+won\s+the\s+(?:match|game|cricket|football|fifa|world\s+cup|super\s+bowl|olympics)|world\s+cup|super\s+bowl|olympics)\b",
        # Legal litigation
        r"\b(?:legal\s+contract|sue\s+my|lawsuit|attorney\s+fees|court\s+hearing)\b",
        # General non-medical trivia / geopolitics
        r"\b(?:capital\s+of|election\s+results|weather\s+in|who\s+is\s+the\s+president|president\s+of\s+the)\b",
        # Non-medical science, energy, physics, astronomy
        r"\b(?:solar\s+panel|solar\s+panels|renewable\s+energy|quantum\s+physics|thermodynamics|black\s+hole|astronomy|chemical\s+reaction|nuclear\s+energy|how\s+do\s+airplanes|how\s+do\s+rockets|photosynthesis)\b",
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


def is_therapist_query(user_text: str) -> bool:
    """Detect whether user input is seeking professional human therapy, counseling, or clinic recommendations."""
    lower = (user_text or "").lower().strip()
    if not lower:
        return False
    professions = ["therapist", "therapists", "psychologist", "psychologists", "psychiatrist", "psychiatrists", "counselor", "counselors", "counseling", "therapy"]
    if any(p in lower for p in professions):
        # Exclude pure disease RAG questions like "therapy for hypertension" if not seeking a practitioner
        if any(w in lower for w in [
            "find", "recommend", "locate", "near", "in", "best", "contact", "list",
            "directory", "suggest", "look for", "search for", "address", "phone",
            "appointment", "session", "locator", "clinical", "licensed", "doctor",
            "where can i", "who can i", "around", "at", "close to"
        ]):
            return True
        # If user directly queries e.g. "therapists mumbai", "therapist locator"
        if any(w in lower for w in ["locator", "near me", "in my area", "directory"]):
            return True
        # General therapist inquiry
        if any(p in lower for p in ["licensed therapist", "clinical psychologist", "therapist locator", "find a counselor"]):
            return True
    return False


def classify_intent_heuristic(user_text: str) -> str:
    """Minimal safety-first fallback classification if AI LLM is completely unreachable."""
    clean = user_text.strip()
    if not clean:
        return "GENERAL_CONVERSATION"

    lower = clean.lower()

    # Safety
    safety = assess_safety(user_text)
    if safety["risk_level"] == "HIGH":
        return "CRISIS_RELATED"
    urgent = screen_urgent_medical(user_text)
    if urgent["is_urgent"]:
        return "URGENT_HEALTH"
    if safety["safety_status"] == "EDUCATIONAL_OR_THIRD_PERSON":
        return "CRISIS_RELATED"

    # Math or out-of-scope non-wellness
    if is_math_query(user_text) or is_out_of_scope(user_text):
        return "OUT_OF_SCOPE"

    # Physical symptom or medical query
    if is_physical_health_query(user_text):
        return "HEALTH_INFORMATION"

    # Therapist request
    if is_therapist_query(user_text):
        return "THERAPIST_REQUEST"

    # Slash and explicit tool commands
    if lower.startswith("/journal") or "record" in lower and "mood" in lower:
        return "JOURNAL_REQUEST"
    if lower.startswith("/goal") or ("create" in lower or "set" in lower) and "goal" in lower:
        return "GOAL_REQUEST"
    if "what did i tell you" in lower or "what is my preferred name" in lower:
        return "MEMORY_REQUEST"

    # Minimal fallback for basic wellness
    if any(w in lower for w in ["breathe", "breathing", "breath", "grounding", "box breathing"]):
        return "GROUNDING_REQUEST"
    if any(w in lower for w in ["cope", "stress", "anxious", "sad", "overwhelm", "overthinking", "nervous"]):
        return "EMOTIONAL_SUPPORT"

    return "GENERAL_CONVERSATION"


# ------------------------------------------------------------------------------
# STRUCTURED INTENT CLASSIFICATION SPECIFICATION & SCHEMAS
# ------------------------------------------------------------------------------
import logging
logger = logging.getLogger("mindly.routing")

class IntentClassificationResult(TypedDict):
    intent: str                  # Canonical intent string
    confidence: float            # Calibrated confidence score (0.0 to 1.0)
    confidence_level: str        # 'HIGH', 'MEDIUM', 'LOW'
    selected_action: str         # Associated LangGraph action
    tool_name: str               # Associated sub-tool identifier
    needs_clarification: bool    # True if ambiguous or underspecified
    clarification_prompt: Optional[str]  # Clarifying question if needed
    routing_reason: str          # Safe, non-private explanation
    raw_intent: Optional[str]    # Raw output from underlying neural model
    classifier_source: str       # 'safety_override' | 'semantic_llm' | 'concept_arbiter' | 'clarification'


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
    "VISION_REQUEST": {
        "action": "USE_VISION_TOOL",
        "tool": "vision_analyzer",
        "desc": "Analyzing, describing, or perceiving uploaded images and visual attachments.",
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
        "stomach", "stommach", "stomache", "herts", "pain", "ache", "aching", "hurt", "hurts", "hurting", "sore", "headache", "migraine",
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
        "burnt out", "burned out", "burn out",
        "sleep", "asleep", "falling asleep", "trouble sleeping", "trouble falling asleep", "trouble with sleep",
        "study", "studying", "mental break", "study break", "break",
        "study tips", "exam stress", "exam pressure", "workload", "recovery pause", "sleep routine",
        "sleep hygiene", "insomnia", "practical ways", "strategy", "strategies", "routine", "habit", "actionable",
        "steps to", "manageable steps", "practical strategies", "stress", "work piles up", "manage everyday stress",
        "overworked", "overworking"
    ],
    "EMOTIONAL_SUPPORT": [
        "feel", "feeling", "feels", "anxious", "anxiety", "sad", "sadness", "lonely", "loneliness", "depressed",
        "overwhelmed", "overwhelm", "overthinking", "nervous", "scared", "fear", "angry", "anger", "frustrated",
        "heartbroken", "breakup", "crying", "cry", "listen", "talk to someone", "need someone", "vent", "venting",
        "down today", "heavy heart", "unmotivated", "lost", "hopeless", "bad day", "rough day", "tough day",
        "terrible", "horrible", "awful", "worst", "terrible day", "horrible day", "awful day", "worst day",
        "miserable", "struggling", "struggle", "drained", "exhausted", "exhasted", "exhaust", "exhaustion",
        "burnt out", "burned out", "burn out", "burnout", "scrambled", "on my plate",
        "hurting", "heartache", "painful",
        "butterflies", "emotional weight", "blue today", "spiraling"
    ],
    "JOURNAL_REQUEST": [
        "journal", "journaling", "log my mood", "log mood", "record mood", "record my mood", "today's mood",
        "mood rating", "out of 10", "out of ten", "/10", "save entry", "diary", "write down my thoughts",
        "reflect on today", "save a new journal entry", "my reflection"
    ],
    "GOAL_REQUEST": [
        "goal", "goals", "habit", "habits", "track", "tracking", "set a goal", "create a goal", "daily walk",
        "daily meditation", "routine target", "commit to", "streak", "target", "establish a", "morning meditation",
        "drink two liters", "weekly study schedule", "plan", "planning", "schedule", "study schedule", "plan my week",
        "exam preparation", "study goal", "study goals", "mark my goal", "mark goal"
    ],
    "MEMORY_REQUEST": [
        "remember that", "remember my", "what is my name", "what did i tell you", "what did i say",
        "my preferred name", "call me", "forget my", "clear all memories", "what was my mood earlier",
        "stored memories", "what do you remember", "mood score i recorded", "preferred nickname",
        "what was the mood score", "recorded earlier", "what did i record"
    ],
    "GENERAL_CONVERSATION": [
        "hello", "hi", "hey", "good morning", "good evening", "how are you", "who are you", "what is your name",
        "what can you do", "tell me a joke", "nice to meet you", "thank you", "thanks", "mindly", "purpose",
        "interesting", "tell me something", "tell me a story", "tell me something interesting", "story", "joke",
        "chat", "conversation", "talk", "what's up", "whats up", "howdy", "good afternoon", "good day"
    ],
    "OUT_OF_SCOPE": [
        "python", "java", "javascript", "c++", "html", "css", "sql", "docker", "algorithm", "binary search",
        "hashmap", "sort", "sorting", "quicksort", "code", "coding", "debug", "programming", "algebra",
        "equation", "solve for x", "calculus", "integral", "derivative", "solar panel", "solar panels",
        "photovoltaic", "astronomy", "black hole", "galaxy", "quantum", "physics", "stock", "stocks",
        "bitcoin", "crypto", "trading", "stock price", "capital city", "capital of france", "capital of",
        "who is the president", "election", "weather in", "car engine", "oil change", "spark plug",
        "fifa", "world cup", "super bowl", "olympics"
    ]
}

FUNCTION_WORDS = {
    "what", "should", "could", "would", "have", "been", "from", "with",
    "that", "this", "they", "them", "then", "there", "some", "thing",
    "your", "help", "please", "about", "which", "whose", "where", "when"
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
                elif any(t.startswith(anchor[:5]) for t in tokens if len(anchor) >= 6 and len(t) >= 5 and t not in FUNCTION_WORDS):
                    score += 0.5
                elif any(t.startswith(anchor[:4]) for t in tokens if len(anchor) == 5 and len(t) >= 4 and t not in FUNCTION_WORDS):
                    score += 0.5
        scores[intent] = score
    return scores


INTENT_PROMPT = """Classify the user message into exactly ONE category from this list:
- CRISIS: Imminent self-harm, suicide, life danger.
- URGENT_MEDICAL: Acute emergencies (chest pain, fainting, difficulty breathing, poisoning, overdose).
- HEALTH_INFORMATION: Physical symptoms, illness, body ache, headaches, stomach pain (Gale Medical RAG).
- GROUNDING_REQUEST: Breathing exercises (box breathing), 5-4-3-2-1 sensory grounding, acute panic relief.
- COPING_REQUEST: Actionable techniques for stress management, sleep hygiene, exam pressure, study burnout.
- EMOTIONAL_SUPPORT: Venting, sadness, loneliness, heartbreak, feeling overwhelmed, wanting someone to listen.
- JOURNAL_REQUEST: Writing reflections, logging daily mood scores (e.g. 6/10).
- GOAL_REQUEST: Setting, building, or tracking personal goals and habits.
- MEMORY_REQUEST: Recalling past shared information, updating preferred name, forgetting memories.
- GENERAL_CONVERSATION: Greetings, conversational check-ins, Mindly identity, general wellness chat.
- OUT_OF_SCOPE: Non-wellness topics: coding (Java/Python), math/algebra, solar panels, science, finance, trivia.
- CLARIFICATION_NEEDED: Extremely vague, ambiguous, or incomplete input where intent is genuinely unclear.

Respond with ONLY the exact category name. Nothing else."""


def classify_intent_structured(user_text: str, llm=None, image_data=None) -> IntentClassificationResult:
    """Perform robust semantic intent classification returning structured, validated output with calibrated confidence."""
    clean = (user_text or "").strip()
    lower_clean = clean.lower()

    # Pre-screen 1: Image attachment
    if image_data:
        res: IntentClassificationResult = {
            "intent": "VISION_REQUEST",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "USE_VISION_TOOL",
            "tool_name": "vision_analyzer",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Visual image attachment provided; routed to multimodal vision analyzer.",
            "raw_intent": "VISION_REQUEST",
            "classifier_source": "deterministic",
        }
        logger.info("RoutingDecision | intent=%s | action=%s | conf=%.2f | source=%s", res["intent"], res["selected_action"], res["confidence"], res["classifier_source"])
        return res

    # Pre-screen 2: Empty input
    if not clean:
        res: IntentClassificationResult = {
            "intent": "GENERAL_CONVERSATION",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "direct_conversation",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Received empty or whitespace input.",
            "raw_intent": "GENERAL_CONVERSATION",
            "classifier_source": "deterministic",
        }
        logger.info("RoutingDecision | intent=%s | action=%s | conf=%.2f | source=%s", res["intent"], res["selected_action"], res["confidence"], res["classifier_source"])
        return res

    # Stage 1: Deterministic Safety Priority 1 - High Risk Crisis
    safety = assess_safety(clean)
    if safety["risk_level"] == "HIGH":
        res: IntentClassificationResult = {
            "intent": "CRISIS",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "CRISIS_RESPONSE",
            "tool_name": "crisis_support_protocol",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Safety invariant: detected imminent crisis indicators; normal routing bypassed.",
            "raw_intent": "CRISIS",
            "classifier_source": "safety_override",
        }
        logger.info("RoutingDecision | intent=%s | action=%s | conf=%.2f | source=%s", res["intent"], res["selected_action"], res["confidence"], res["classifier_source"])
        return res

    # Stage 1 (Cont): Deterministic Safety Priority 2 - Acute Medical Red-Flags
    urgent = screen_urgent_medical(clean)
    if urgent["is_urgent"]:
        res: IntentClassificationResult = {
            "intent": "URGENT_MEDICAL",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "USE_RAG_TOOL",
            "tool_name": "urgent_medical_protocol",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Safety invariant: detected acute somatic emergency red-flags.",
            "raw_intent": "URGENT_MEDICAL",
            "classifier_source": "safety_override",
        }
        logger.info("RoutingDecision | intent=%s | action=%s | conf=%.2f | source=%s", res["intent"], res["selected_action"], res["confidence"], res["classifier_source"])
        return res

    # Stage 1 (Cont): Educational or Third-Person Crisis inquiry (only when crisis/suicide terms present)
    has_crisis_term = any(k in lower_clean for k in ["suicide", "kill", "die", "death", "end life", "self harm", "harming", "hurt myself"]) or bool(re.search(r"\bend\s+(?:their|his|her|a|my)\s+life\b", lower_clean))
    if (safety["safety_status"] == "EDUCATIONAL_OR_THIRD_PERSON" or is_educational_or_third_person(clean)) and has_crisis_term:
        res: IntentClassificationResult = {
            "intent": "CRISIS",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "third_person_support_protocol",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Identified educational crisis or third-person support inquiry.",
            "raw_intent": "CRISIS",
            "classifier_source": "safety_override",
        }
        logger.info("RoutingDecision | intent=%s | action=%s | conf=%.2f | source=%s", res["intent"], res["selected_action"], res["confidence"], res["classifier_source"])
        return res

    # Stage 1 (Cont): Explicit arithmetic / math computation
    if evaluate_math_expression(clean) is not None:
        res: IntentClassificationResult = {
            "intent": "OUT_OF_SCOPE",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "scope_boundary_protocol",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Arithmetic calculation query outside mental wellness scope.",
            "raw_intent": "OUT_OF_SCOPE",
            "classifier_source": "deterministic",
        }
        logger.info("RoutingDecision | intent=%s | action=%s | conf=%.2f | source=%s", res["intent"], res["selected_action"], res["confidence"], res["classifier_source"])
        return res

    # Explicit slash commands and memory directives
    if lower_clean.startswith("/journal"):
        res: IntentClassificationResult = {
            "intent": "JOURNAL_REQUEST",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "USE_JOURNAL_TOOL",
            "tool_name": "journal_recorder",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Explicit /journal slash command.",
            "raw_intent": "JOURNAL_REQUEST",
            "classifier_source": "deterministic",
        }
        return res
    if lower_clean.startswith("/goal"):
        res: IntentClassificationResult = {
            "intent": "GOAL_REQUEST",
            "confidence": 1.0,
            "confidence_level": "HIGH",
            "selected_action": "USE_GOAL_TOOL",
            "tool_name": "goal_tracker",
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Explicit /goal slash command.",
            "raw_intent": "GOAL_REQUEST",
            "classifier_source": "deterministic",
        }
        return res
    if bool(re.search(r"^(?:remember\s+(?:that|my)|please\s+remember|don't\s+forget|dont\s+forget|keep\s+in\s+mind\s+that)\b", lower_clean)):
        spec = INTENT_SPECS["MEMORY_REQUEST"]
        res: IntentClassificationResult = {
            "intent": "MEMORY_REQUEST",
            "confidence": 0.95,
            "confidence_level": "HIGH",
            "selected_action": spec["action"],
            "tool_name": spec["tool"],
            "needs_clarification": False,
            "clarification_prompt": None,
            "routing_reason": "Explicit persistent memory storage instruction.",
            "raw_intent": "MEMORY_REQUEST",
            "classifier_source": "deterministic",
        }
        logger.info("RoutingDecision | intent=%s | action=%s | conf=%.2f | source=%s", res["intent"], res["selected_action"], res["confidence"], res["classifier_source"])
        return res

    # Stage 2: Semantic Concept Scoring
    concept_scores = compute_concept_scores(clean)
    sorted_scores = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
    top_concept_intent, top_concept_score = sorted_scores[0]

    # Stage 3: Semantic LLM Evaluation (Selectively invoked when concept scoring is low or ambiguous)
    llm_intent = None
    effective_llm = llm
    should_query_llm = (llm is not None) or (top_concept_score < 2.0 and top_concept_score > 0.0) or (top_concept_score == 0.0 and len(tokenize(clean)) > 4)

    if should_query_llm:
        if effective_llm is None:
            try:
                from config.llm_provider import get_llm
                effective_llm = get_llm(temperature=0.0, max_tokens=25)
            except Exception:
                effective_llm = None

        if effective_llm is not None:
            try:
                messages = [
                    SystemMessage(content=INTENT_PROMPT),
                    HumanMessage(content=f'User: "{clean}"\nCategory:'),
                ]
                resp = effective_llm.invoke(messages)
                raw = str(resp.content).strip().upper()
                for cat in INTENT_SPECS.keys():
                    if cat in raw:
                        llm_intent = cat
                        break
            except Exception:
                llm_intent = None

    # Stage 4: Ambiguity & Vague Input Detection (Clarification Protocol)
    is_vague_ref = bool(re.search(r"\b(?:that thing|this thing|with that|with this|about that|about it|help me with it|do that|do it)\b", lower_clean))
    clean_vague = re.sub(r"[?!.,]", "", lower_clean).strip()
    is_underspecified = clean_vague in [
        "help", "help me", "help me please", "please help me", "can you help",
        "can you help me", "i need help", "what should i do", "what to do",
        "what can i do", "tell me what to do", "i need something to help me right now",
        "tell me about that", "can you do it", "do it", "help please"
    ]

    if top_concept_score == 0.0 and (is_vague_ref or is_underspecified or llm_intent == "CLARIFICATION_NEEDED"):
        res: IntentClassificationResult = {
            "intent": "CLARIFICATION_NEEDED",
            "confidence": 0.40,
            "confidence_level": "LOW",
            "selected_action": "DIRECT_RESPONSE",
            "tool_name": "clarification_protocol",
            "needs_clarification": True,
            "clarification_prompt": "I want to make sure I support you in the best way possible. Could you clarify what you would like to focus on right now?",
            "routing_reason": "Vague or underspecified input with zero domain concept overlap.",
            "raw_intent": llm_intent or "CLARIFICATION_NEEDED",
            "classifier_source": "clarification",
        }
        logger.info("RoutingDecision | intent=%s | action=%s | conf=%.2f | source=%s", res["intent"], res["selected_action"], res["confidence"], res["classifier_source"])
        return res

    # Stage 5: Domain Boundary Resolution (Out-of-Scope)
    if top_concept_intent == "OUT_OF_SCOPE" or is_out_of_scope(clean) or (llm_intent == "OUT_OF_SCOPE" and is_out_of_scope(clean)):
        wellness_sum = sum(concept_scores[k] for k in ["EMOTIONAL_SUPPORT", "HEALTH_INFORMATION", "GROUNDING_REQUEST", "COPING_REQUEST", "GENERAL_CONVERSATION"])
        if wellness_sum == 0.0 and not is_physical_health_query(clean):
            res: IntentClassificationResult = {
                "intent": "OUT_OF_SCOPE",
                "confidence": 0.95,
                "confidence_level": "HIGH",
                "selected_action": "DIRECT_RESPONSE",
                "tool_name": "scope_boundary_protocol",
                "needs_clarification": False,
                "clarification_prompt": None,
                "routing_reason": "Domain boundary guardrail: topic is outside mental wellness and health information.",
                "raw_intent": llm_intent or "OUT_OF_SCOPE",
                "classifier_source": "concept_arbiter",
            }
            logger.info("RoutingDecision | intent=%s | action=%s | conf=%.2f | source=%s", res["intent"], res["selected_action"], res["confidence"], res["classifier_source"])
            return res

    # Stage 6: Multi-Signal Decision Arbiter
    # Determine resolved intent between Concept Profile and LLM prediction
    if top_concept_score == 0.0:
        resolved_intent = llm_intent if (llm_intent and llm_intent in INTENT_SPECS and llm_intent != "OUT_OF_SCOPE") else "GENERAL_CONVERSATION"
        source = "semantic_llm" if (llm_intent and llm_intent in INTENT_SPECS and llm_intent != "OUT_OF_SCOPE") else "general_fallback"
        calibrated_conf = 0.70
    else:
        resolved_intent = top_concept_intent
        source = "concept_arbiter"
        calibrated_conf = 0.85

    if llm_intent and llm_intent in INTENT_SPECS:
        if llm_intent == top_concept_intent:
            resolved_intent = llm_intent
            calibrated_conf = 0.95
            source = "semantic_llm"
        elif top_concept_score >= 2.5:
            resolved_intent = top_concept_intent
            calibrated_conf = 0.88
            source = "concept_arbiter"
        elif top_concept_score == 0.0 and llm_intent != "OUT_OF_SCOPE":
            resolved_intent = llm_intent
            calibrated_conf = 0.75
            source = "semantic_llm"

    # Enforce safety and domain invariants
    if is_physical_health_query(clean):
        resolved_intent = "HEALTH_INFORMATION"
        calibrated_conf = 0.95

    spec = INTENT_SPECS.get(resolved_intent, INTENT_SPECS["GENERAL_CONVERSATION"])
    conf_level = "HIGH" if calibrated_conf >= 0.80 else "MEDIUM" if calibrated_conf >= 0.60 else "LOW"

    res: IntentClassificationResult = {
        "intent": resolved_intent,
        "confidence": round(calibrated_conf, 2),
        "confidence_level": conf_level,
        "selected_action": spec["action"],
        "tool_name": spec["tool"],
        "needs_clarification": False,
        "clarification_prompt": None,
        "routing_reason": f"Semantic router matched {resolved_intent} via {source} (score={top_concept_score}).",
        "raw_intent": llm_intent or resolved_intent,
        "classifier_source": source,
    }
    logger.info("RoutingDecision | intent=%s | action=%s | conf=%.2f | source=%s", res["intent"], res["selected_action"], res["confidence"], res["classifier_source"])
    return res


def classify_intent(user_text: str, llm=None) -> str:
    """Convenience classification wrapper returning canonical intent string with backwards-compatible aliases."""
    result = classify_intent_structured(user_text, llm=llm)
    intent = result["intent"]
    lower = (user_text or "").lower()
    # Map canonical intents to legacy compatibility aliases if needed
    if intent == "CRISIS":
        return "CRISIS_RELATED"
    if intent == "URGENT_MEDICAL":
        return "URGENT_HEALTH"
    if intent in ["GOAL_REQUEST", "COPING_REQUEST"] and "plan" in lower and "goal" not in lower and "habit" not in lower:
        return "PLANNING_REQUEST"
    return intent



