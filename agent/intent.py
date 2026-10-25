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
        r"\bdiabetes\b", r"\bdiabetic\b", r"\bblood\s+sugar\b", r"\bglucose\b", r"\binsulin\b",
        r"\basthma\b", r"\bbronchitis\b", r"\bpneumonia\b", r"\bwheezing\b", r"\brespiratory\b", r"\blungs?\b",
        r"\bstomach\s+pain\b", r"\babdominal\s+pain\b", r"\bpain\s+in\s+(?:my\s+)?stomach\b", r"\bstomach\s+hurts?\b",
        r"\bstomach\s+ache\b", r"\bstomachache\b", r"\bbelly\s+ache\b", r"\btummy\s+hurt\b",
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

    # Explicit tool commands
    if lower.startswith("/journal"):
        return "JOURNAL_REQUEST"
    if lower.startswith("/goal"):
        return "GOAL_REQUEST"

    # Minimal fallback for basic wellness
    if "breathe" in lower or "grounding" in lower:
        return "GROUNDING_REQUEST"
    if any(w in lower for w in ["cope", "stress", "anxious", "sad", "overwhelm"]):
        return "EMOTIONAL_SUPPORT"

    return "GENERAL_CONVERSATION"


INTENT_PROMPT = """Classify the user input into exactly ONE category:
HEALTH_INFORMATION: physical diseases, illnesses, medical conditions, acne, skin issues, bodily pain, shoulder pain, back pain, joint pain, headaches, fever, indigestion, hypertension, asthma, diabetes.
EMOTIONAL_SUPPORT: mental wellness, emotions, feelings, jealousy, sadness, loneliness, heartbreak, feeling down, relationship concerns.
COPING_REQUEST: practical coping strategies, stress management, study advice, burnout handling, ways to manage pressure, questions asking "what should I do", "how can I cope", "ways to handle".
GROUNDING_REQUEST: breathing exercises, calming down, sensory grounding techniques, panic relief ("need help calming down", "help me breathe").
MEMORY_REQUEST: remembering user's name, user preferences, recalling stored memories, questions like "what did I ask you to remember", "remember that...".
JOURNAL_REQUEST: personal reflection journaling, saving entries to journal, viewing journal.
GOAL_REQUEST: setting, tracking, or viewing goals and habits.
PLANNING_REQUEST: study schedules, time management plans, organizing routines.
THERAPIST_REQUEST: finding licensed human therapists, psychologists, counselors, or mental health clinics.
OUT_OF_SCOPE: computer programming, coding, math calculations, stock market, non-wellness topics.
GENERAL_CONVERSATION: greetings, casual talk, jokes, telling interesting facts, questions about your identity or abilities.

Examples:
"how to handle shoulder pain" -> HEALTH_INFORMATION
"how to handle skin acnes" -> HEALTH_INFORMATION
"what causes back pain" -> HEALTH_INFORMATION
"how to stop being jealous" -> EMOTIONAL_SUPPORT
"i feel sad and lonely" -> EMOTIONAL_SUPPORT
"what should I do when I feel emotionally exhausted?" -> COPING_REQUEST
"how can i manage stress" -> COPING_REQUEST
"ways to deal with burnout" -> COPING_REQUEST
"teach me box breathing" -> GROUNDING_REQUEST
"I feel anxious and need help calming down." -> GROUNDING_REQUEST
"help me calm down" -> GROUNDING_REQUEST
"remember that I prefer being called Alex" -> MEMORY_REQUEST
"what did I ask you to remember?" -> MEMORY_REQUEST
"what is my preferred name?" -> MEMORY_REQUEST
"write this in my journal" -> JOURNAL_REQUEST
"set a goal to exercise" -> GOAL_REQUEST
"help me plan my study schedule" -> PLANNING_REQUEST
"tell me a joke" -> GENERAL_CONVERSATION
"tell me something interesting" -> GENERAL_CONVERSATION
"find a therapist near me" -> THERAPIST_REQUEST
"write code in python" -> OUT_OF_SCOPE
"hello how are you" -> GENERAL_CONVERSATION

Respond with ONLY the category name. Nothing else."""


def classify_intent(user_text: str, llm=None) -> str:
    """Classify user intent by letting the AI dynamically decide, bounded by deterministic safety."""
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

    lower_clean = clean.lower()

    # Math expressions (2+3, 2+2, etc.) and non-wellness topics (coding, stocks, etc.) are strictly OUT_OF_SCOPE
    if evaluate_math_expression(user_text) is not None or is_out_of_scope(user_text):
        return "OUT_OF_SCOPE"

    # Explicit slash commands (/journal, /goal)
    if lower_clean.startswith("/journal"):
        return "JOURNAL_REQUEST"
    if lower_clean.startswith("/goal"):
        return "GOAL_REQUEST"

    # Conversational greetings, identity & chit-chat queries ("hello mindly", "thank you", "who are you")
    if any(q in lower_clean for q in [
        "what is your name", "what's your name", "who are you", "tell me about yourself",
        "tell me something about yourself", "tell me a joke", "tell me something interesting",
        "hello mindly", "hi mindly", "hey mindly", "thank you", "thanks", "nice to meet you",
        "good morning", "good evening", "good afternoon"
    ]) or lower_clean in ["hello", "hi", "hey", "how are you", "how are you doing", "how are you doing today?"]:
        return "GENERAL_CONVERSATION"

    # Stage 1 (Cont): Physical health symptoms must not be routed to mindfulness/emotional simply because of scared/fear words
    if is_physical_health_query(user_text):
        return "HEALTH_INFORMATION"

    # Stage 2: LET THE AI DECIDE INTENT DYNAMICALLY
    effective_llm = llm
    if effective_llm is None:
        try:
            from config.llm_provider import get_llm
            effective_llm = get_llm(temperature=0.0, max_tokens=20)
        except Exception:
            effective_llm = None

    if effective_llm is not None:
        try:
            messages = [
                SystemMessage(content=INTENT_PROMPT),
                HumanMessage(content=f'Input: "{user_text}"\nCategory:'),
            ]
            response = effective_llm.invoke(messages)
            raw_category = str(response.content).strip().upper()
            for cat in INTENT_CATEGORIES:
                if cat in raw_category:
                    # Safety invariant: LLM can never downgrade high risk
                    if cat != "CRISIS_RELATED" and safety["risk_level"] == "HIGH":
                        return "CRISIS_RELATED"
                    return cat
        except Exception:
            pass

    # Stage 3: Fallback heuristic only if AI is unreachable
    return classify_intent_heuristic(user_text)


