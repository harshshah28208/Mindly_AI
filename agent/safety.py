"""Deterministic safety guard and crisis response generator for Mindly.

Provides a robust, non-LLM safety pre-check that scans for high-risk
indicators (self-harm, suicidal ideation, immediate danger) and routes
directly to a safe, supportive crisis protocol.

Hierarchical priority:
- PRIORITY 1: High-Risk Safety (first-person suicidal/self-harm intent)
- PRIORITY 2: Urgent Medical / Health Symptoms
- Educational & third-person discussions are explicitly separated to prevent false positives.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


# Regex patterns for HIGH-risk first-person intent (suicide, imminent self-harm)
HIGH_RISK_PATTERNS: List[re.Pattern] = [
    # 1. Kill myself / killing myself / murder myself
    re.compile(r"\b(?:kill|killing|murder|murdering)\s+(?:my\s*self|myself)\b", re.IGNORECASE),
    # 2. Suicide as verb, intent, or action
    re.compile(r"\b(?:want\s+to|wanna|going\s+to|gonna|will|might|plan\s+to|planning\s+to|thinking\s+(?:about|of)|decided\s+to|feel\s+like)\s+suicid(?:e|ing)\b", re.IGNORECASE),
    re.compile(r"\b(?:commit|committing|attempt|attempting)\s+suicide\b", re.IGNORECASE),
    # 3. Want/gonna die
    re.compile(r"\b(?:want\s+to|wanna|going\s+to|gonna|plan\s+to|wish\s+i\s+(?:was|were|could))\s+(?:die|be\s+dead)\b", re.IGNORECASE),
    re.compile(r"\b(?:better\s+off\s+dead|everyone\s+(?:would\s+be\s+)?better\s+off\s+without\s+me)\b", re.IGNORECASE),
    # 4. End my life / take my own life
    re.compile(r"\b(?:end\s+my\s+life|ending\s+my\s+life|end\s+it\s+all|ending\s+it\s+all)\b", re.IGNORECASE),
    re.compile(r"\b(?:take\s+my\s+own\s+life|taking\s+my\s+own\s+life|decided\s+to\s+end\s+my\s+life)\b", re.IGNORECASE),
    # 5. Don't want to live
    re.compile(r"\b(?:don't|dont)\s+(?:want\s+to|wanna)\s+(?:live|exist|wake\s+up)\s*(?:anymore|any\s*more)?\b", re.IGNORECASE),
    re.compile(r"\b(?:can't|cant)\s+go\s+on\s+living\b", re.IGNORECASE),
    re.compile(r"\b(?:no\s+reason|nothing)\s+to\s+live\s+for\b", re.IGNORECASE),
    # 6. Self-harm / cutting / hurting oneself
    re.compile(r"\b(?:hurt|hurting|harm|harming|cut|cutting|injure|injuring)\s+(?:my\s*self|myself)\b", re.IGNORECASE),
    re.compile(r"\b(?:might|gonna|going\s+to|want\s+to|wanna|thinking\s+(?:about|of))\s+(?:hurt|hurting|harm|harming|cut|cutting)\s+myself\b", re.IGNORECASE),
    # 7. Lethal methods
    re.compile(r"\b(?:hang|hanging)\s+myself\b", re.IGNORECASE),
    re.compile(r"\b(?:overdose\s+on\s+pills|swallow\s+pills\s+to\s+die|jump\s+off\s+a\s+bridge)\b", re.IGNORECASE),
    # 8. Active suicidal state
    re.compile(r"\b(?:i\s+am|i'm|feeling|feel)\s+(?:suicidal|thinking\s+of\s+suicide)\b", re.IGNORECASE),
    re.compile(r"\bsuicidal(?:\s+thoughts|\s+ideation|\s+feelings|\s+urges)\b", re.IGNORECASE),
]

# Patterns for MEDIUM-risk distress (acute distress/hopelessness without explicit self-harm statement)
MEDIUM_RISK_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:can't\s+take\s+this|cant\s+take\s+this)\s+anymore\b", re.IGNORECASE),
    re.compile(r"\b(?:feel|feeling)\s+(?:completely\s+)?hopeless\b", re.IGNORECASE),
    re.compile(r"\b(?:no\s+point\s+in\s+anything|nothing\s+matters\s+anymore)\b", re.IGNORECASE),
    re.compile(r"\b(?:severe\s+panic|having\s+a\s+panic\s+attack|unbearable\s+emotional\s+pain)\b", re.IGNORECASE),
    re.compile(r"\b(?:breaking\s+down|falling\s+apart\s+completely)\b", re.IGNORECASE),
    re.compile(r"\b(?:nobody\s+cares\s+about\s+me|nobody\s+would\s+miss\s+me)\b", re.IGNORECASE),
]


def normalize_input(user_text: str) -> str:
    """Normalize input string by stripping whitespace and control characters."""
    if not user_text:
        return ""
    # Strip whitespace and normalize multiple spaces
    cleaned = re.sub(r"\s+", " ", user_text.strip())
    return cleaned


def is_educational_or_third_person(user_text: str) -> bool:
    """Detect whether mentions of suicide/harm are purely educational, informational, or third-person.

    Prevents false-positive HIGH risk triggers on queries like:
    - "What is suicide prevention?"
    - "I'm studying suicide prevention."
    - "My friend attempted suicide."
    - "What does suicide mean?"
    - "How does suicide prevention work?"
    - "I watched a movie about suicide."
    """
    lower = user_text.lower().strip()

    # 1. First verify there is NO direct first-person self-harm intent
    first_person_triggers = [
        "i want to die", "i wanna die", "i will kill myself", "i want to kill myself",
        "kill myself", "end my life", "hurt myself", "i want to suicide", "i wanna suicide",
        "dont wanna live", "don't wanna live", "don't want to live", "dont want to live",
        "better off dead"
    ]
    if any(fp in lower for fp in first_person_triggers):
        return False

    # 2. Educational, definitional, research, or prevention inquiries
    if any(p in lower for p in [
        "suicide prevention",
        "what is suicide",
        "what does suicide mean",
        "definition of suicide",
        "studying suicide",
        "researching suicide",
        "movie about suicide",
        "book about suicide",
        "essay on suicide",
        "statistics on suicide",
        "history of suicide",
        "signs of suicide",
        "why do people commit suicide",
        "understand suicide",
        "awareness of suicide",
    ]):
        return True

    # 3. Third-person mentions (concerning another person, friend, family member)
    third_person_subjects = [
        "my friend", "a friend", "my cousin", "my brother", "my sister",
        "my mother", "my father", "my dad", "my mom", "my classmate",
        "someone i know", "a person i know", "my neighbor", "celebrity",
        "someone else", "another person"
    ]
    if any(subj in lower for subj in third_person_subjects):
        return True

    return False


def assess_safety(user_text: str) -> Dict[str, Any]:
    """Deterministically classify safety risk into LOW, MEDIUM, or HIGH.

    Enforces Priority 1: High-Risk Safety must always override normal routing.
    Protects against false positives for educational and third-person discussions.

    Returns:
        Dict with:
            - risk_level: 'LOW' | 'MEDIUM' | 'HIGH'
            - safety_status: 'PASSED' | 'HIGH_RISK_TRIGGERED' | 'MEDIUM_RISK_DISTRESS' | 'EDUCATIONAL_OR_THIRD_PERSON'
            - is_safe: bool
            - matched_patterns: List[str]
    """
    clean_text = normalize_input(user_text)
    if not clean_text:
        return {
            "risk_level": "LOW",
            "safety_status": "EMPTY_INPUT",
            "is_safe": True,
            "matched_patterns": [],
        }

    # 1. Check for Educational or Third-Person discussions first
    if is_educational_or_third_person(clean_text):
        return {
            "risk_level": "LOW",
            "safety_status": "EDUCATIONAL_OR_THIRD_PERSON",
            "is_safe": True,
            "matched_patterns": [],
        }

    # 2. Check for HIGH-risk matches (Self-harm / Suicidal intent)
    for pattern in HIGH_RISK_PATTERNS:
        match = pattern.search(clean_text)
        if match:
            return {
                "risk_level": "HIGH",
                "safety_status": "HIGH_RISK_TRIGGERED",
                "is_safe": False,
                "matched_patterns": [match.group(0)],
            }

    # 3. Check for MEDIUM-risk matches (Acute distress without self-harm intent)
    for pattern in MEDIUM_RISK_PATTERNS:
        match = pattern.search(clean_text)
        if match:
            return {
                "risk_level": "MEDIUM",
                "safety_status": "MEDIUM_RISK_DISTRESS",
                "is_safe": True,
                "matched_patterns": [match.group(0)],
            }

    return {
        "risk_level": "LOW",
        "safety_status": "PASSED",
        "is_safe": True,
        "matched_patterns": [],
    }


def screen_urgent_medical(user_text: str) -> Dict[str, Any]:
    """Screen for acute or potentially urgent medical symptoms (Priority 2).

    Detects combinations involving:
    - severe or rapidly worsening pain
    - difficulty breathing / shortness of breath
    - chest pain or tightness
    - fainting / loss of consciousness / syncope
    - vomiting blood
    - blood in stool or black tarry stool
    - rigid or swollen abdomen
    - acute confusion or severe weakness
    - severe allergic reaction / anaphylaxis

    Returns:
        Dict with:
            - is_urgent: bool
            - flagged_symptoms: List[str]
            - advisory: Optional[str]
    """
    lower = user_text.lower()
    flagged = []

    if any(k in lower for k in [
        "severe or worsening", "severe worsening", "rapidly worsening",
        "unbearable pain", "excruciating pain", "excruciating", "severe abdominal pain",
        "severe chest pain", "worsening abdominal"
    ]):
        flagged.append("severe or worsening pain")

    if any(k in lower for k in [
        "difficulty breathing", "trouble breathing", "shortness of breath",
        "can't breathe", "cant breathe", "hard to breathe", "struggling to breathe"
    ]):
        flagged.append("difficulty breathing or shortness of breath")

    if any(k in lower for k in [
        "chest pain", "pain in chest", "chest pressure", "tightness in chest", "crushing chest"
    ]):
        flagged.append("chest pain or tightness")

    if any(k in lower for k in [
        "fainting", "fainted", "pass out", "passed out", "syncope", "loss of consciousness", "blacking out"
    ]):
        flagged.append("fainting or loss of consciousness")

    if any(k in lower for k in [
        "vomiting blood", "vomit blood", "blood in vomit", "throwing up blood", "hematemesis"
    ]):
        flagged.append("vomiting blood")

    if any(k in lower for k in [
        "blood in stool", "bloody stool", "rectal bleeding", "bleeding from rectum"
    ]):
        flagged.append("blood in the stool")

    if any(k in lower for k in [
        "black stool", "tarry stool", "black/tarry stool", "black tarry"
    ]):
        flagged.append("black or tarry stool")

    if any(k in lower for k in [
        "rigid abdomen", "swollen abdomen", "abdomen is rigid", "rigid/swollen",
        "hard abdomen", "stomach is hard as a rock", "abdomen feels rigid"
    ]):
        flagged.append("a rigid or severely swollen abdomen")

    if any(k in lower for k in [
        "sudden confusion", "acute confusion", "severe weakness", "unable to stand",
        "unable to walk", "slurred speech"
    ]):
        flagged.append("acute confusion or severe weakness")

    if any(k in lower for k in [
        "anaphylaxis", "throat closing", "swollen tongue", "severe allergic reaction"
    ]):
        flagged.append("symptoms of a severe allergic reaction")

    if flagged:
        flagged_str = ", ".join(flagged)
        advisory = (
            f"🚨 **Urgent Medical Assessment Recommended**:\n"
            f"You noted acute symptoms associated with **{flagged_str}**.\n\n"
            f"Severe, rapidly worsening, or acute symptoms like these require immediate professional medical evaluation. "
            f"Please contact an emergency department, urgent care clinic, or call local emergency medical services right away.\n\n"
            f"*(Mindly is an educational wellness companion and does not provide clinical triage, emergency care, or medical diagnosis.)*"
        )
        return {"is_urgent": True, "flagged_symptoms": flagged, "advisory": advisory}

    return {"is_urgent": False, "flagged_symptoms": [], "advisory": None}


def generate_crisis_response() -> str:
    """Generate a supportive, grounded response with verified crisis resources.

    Strict boundary: Does NOT claim to contact emergency services or diagnose.
    Provides direct, warm human support hotlines.
    """
    return (
        "I hear how much pain you are carrying right now, and I want you to know that "
        "your life and safety truly matter. Please do not go through this alone.\n\n"
        "Because I am an AI companion and cannot provide emergency medical care, "
        "I strongly encourage you to connect right now with someone trained to help you through this moment:\n\n"
        "• **In the US & Canada**: Call or text **988** (Suicide & Crisis Lifeline, free & confidential 24/7)\n"
        "• **Crisis Text Line**: Text **HOME** to **741741**\n"
        "• **In the UK**: Call **111** (NHS Mental Health Services) or **116 123** (Samaritans)\n"
        "• **In India**: Call **14416** (Tele-MANAS) or **+91 9999 666 555** (Vandrevala Foundation)\n"
        "• **International**: Find resources in your country at [findahelpline.com](https://findahelpline.com)\n\n"
        "If you are in immediate physical danger, please reach out to your local emergency services, "
        "go to the nearest hospital emergency room, or call a trusted family member, counselor, or friend.\n\n"
        "Take a slow, deep breath. There are people who want to listen and support you through this."
    )


def generate_third_person_support_response(clean_text: str) -> str:
    """Generate compassionate guidance when a user is discussing a friend's crisis or educational suicide prevention."""
    return (
        "Supporting someone who is experiencing a mental health crisis can be deeply stressful, "
        "and seeking information or help is an important step.\n\n"
        "If your friend or someone you care about is in immediate crisis or having thoughts of self-harm, "
        "please encourage them to reach out to free, confidential crisis support, or connect them directly:\n\n"
        "• **In the US & Canada**: Call or text **988** (available 24/7)\n"
        "• **Crisis Text Line**: Text **HOME** to **741741**\n"
        "• **In the UK**: Call **111** or **116 123**\n"
        "• **In India**: Call **14416** (Tele-MANAS)\n\n"
        "Remember to also take care of yourself. Being there for a friend is meaningful, but you do not have to carry it alone."
    )
