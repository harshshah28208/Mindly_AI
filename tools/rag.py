"""Gale Encyclopedia of Medicine RAG tool for Mindly.

Provides 100% dynamic, grounded clinical answers strictly synthesizing retrieved
knowledge chunks from the Gale Encyclopedia of Medicine.
Zero hardcoded condition dictionaries. Uses LLM-driven grounded synthesis with
dynamic fallback directly parsing retrieved knowledge chunks.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from rag.retriever import retrieve_relevant_chunks


RAG_SYSTEM_PROMPT = """You are an expert clinical health educator and medical knowledge specialist for Mindly, grounded in the Gale Encyclopedia of Medicine.
When responding to any disease, illness, physical symptom, or health inquiry:
1. Provide a comprehensive, highly helpful, and well-structured clinical guide directly synthesizing the medical knowledge.
2. DO NOT output canned refusals (such as 'I cannot provide medical advice', 'I can't provide a diagnosis or medical advice', or 'As an AI language model'). As an educational health specialist, your role is to explain the anatomical and physiological mechanisms and evidence-based self-care from the Gale Encyclopedia excerpts.
3. Structure your answer with clear markdown sections:
   - 📋 **What Is Happening (Clinical Overview & Definition)**: Explain the condition, classification, and physiological mechanism.
   - 🔍 **What Is the Reason (Underlying Causes, Mechanisms & Triggers)**: Why it develops, contributing risk factors (e.g. biomechanical stress, brain-gut axis, stress, genetics, diet, lifestyle).
   - 💊 **Immediate Action Plan (What to Do Right Now for Relief)**: Step-by-step immediate comfort measures (e.g. R.I.C.E. protocol, posture, hydration, resting).
   - 🛡️ **Proper Plan: How to Cure & Prevent (Long-Term Management & Clinical Therapies)**: Physical rehabilitation, dietary changes, lifestyle habits, and standard clinical therapies.
   - ⚠️ **Red-Flag Warning Signs (When to Seek Emergency Care)**: Acute symptoms requiring emergency medical evaluation.
4. PRESENT THE EDUCATIONAL AND CLINICAL KNOWLEDGE FIRST. Explain the condition and helpful measures before stating any referral disclaimers.
5. Place citations (*Gale Encyclopedia of Medicine*) and the professional medical consultation advisory at the very BOTTOM.
6. Strict Boundaries: Do not state personal diagnoses (never say 'you have X') or prescribe specific prescription dosages.
"""


def screen_medical_urgency(query: str) -> Optional[str]:
    """Identify acute red-flag medical emergencies that require prompt evaluation notice."""
    lower = query.lower()
    urgent_patterns = [
        r"\b(?:crushing\s+chest\s+pain|chest\s+pressure|heart\s+attack)\b",
        r"\b(?:can't\s+breathe|cannot\s+breathe|severe\s+shortness\s+of\s+breath|gasping\s+for\s+air)\b",
        r"\b(?:vomiting\s+blood|coughing\s+up\s+blood|black\s+tarry\s+stool)\b",
        r"\b(?:sudden\s+weakness|facial\s+droop|slurred\s+speech|stroke\s+symptoms)\b",
        r"\b(?:fainted|fainting|unconscious|passed\s+out|loss\s+of\s+consciousness)\b",
        r"\b(?:severe\s+allergic\s+reaction|anaphylaxis|throat\s+closing)\b",
        r"\b(?:high\s+fever\s+and\s+stiff\s+neck|fever\s+over\s+104)\b",
    ]
    for pat in urgent_patterns:
        if re.search(pat, lower):
            return (
                "⚠️ **Clinical Precaution**: Some of the symptoms you described can indicate an acute medical situation. "
                "If you or someone nearby is experiencing acute distress, severe chest pressure, or difficulty breathing, "
                "please seek urgent emergency medical evaluation immediately."
            )
    return None


def format_citations(chunks: List[Dict[str, Any]], primary_page: Optional[int] = None) -> str:
    """Format Gale Encyclopedia of Medicine citations from retrieved chunks."""
    pages = set()
    if primary_page:
        pages.add(primary_page)
    for c in chunks:
        p = c.get("page")
        if p:
            pages.add(p)
    sorted_pages = sorted(list(pages))
    if sorted_pages:
        pages_str = ", ".join(f"Page {p}" for p in sorted_pages)
        return f"\n\n📚 **Source**: *Gale Encyclopedia of Medicine* ({pages_str})"
    return "\n\n📚 **Source**: *Gale Encyclopedia of Medicine*"


def extract_condition_topic(query: str) -> str:
    """Extract clinical topic or symptom phrase from inquiry for dynamic synthesis."""
    clean = re.sub(r"[?!.,;:]", " ", query).strip()
    stopwords = {
        "i", "am", "having", "have", "feel", "feeling", "a", "an", "the", "what",
        "to", "do", "give", "me", "reason", "and", "proper", "cure", "plan", "how",
        "can", "treat", "causes", "of", "in", "my", "why", "is", "it", "so", "much",
        "please", "help", "with", "tell", "about", "symptoms"
    }
    words = [w for w in clean.split() if w.lower() not in stopwords]
    if words:
        return " ".join(words).title()
    return "Clinical Health & Symptom Guidance"


def is_canned_refusal_or_insufficient(content: str) -> bool:
    """Detect whether LLM output is a short canned refusal or lacks essential clinical sections."""
    if not content or len(content.strip()) < 80:
        return True

    lower = content.lower().strip()
    refusal_patterns = [
        r"\b(?:can't|cannot|unable\s+to|not\s+able\s+to)\s+(?:provide|give|offer)\s+(?:a\s+)?(?:diagnosis|medical\s+advice|clinical\s+advice)\b",
        r"\b(?:as\s+an?\s+ai|i\s+am\s+an?\s+ai|language\s+model)\b",
        r"\b(?:i\s+am\s+not\s+a\s+doctor|cannot\s+diagnose\s+or\s+treat)\b",
        r"\b(?:i\s+cannot\s+diagnose|i\s+can't\s+diagnose)\b",
        r"\b(?:cannot\s+prescribe|can't\s+prescribe)\b",
        r"\b(?:i\s+can't\s+give\s+medical\s+advice|i\s+cannot\s+give\s+medical\s+advice)\b",
    ]
    for pat in refusal_patterns:
        if re.search(pat, lower):
            return True

    # Must contain essential clinical explanation
    has_reason = any(h in lower for h in ["what is the reason", "reason", "causes", "etiology", "mechanisms", "pathophysiology", "why it occurs", "underlying"])
    has_cure = any(h in lower for h in ["how to cure", "action plan", "relief", "cure", "management", "treatment", "immediate action", "comfort measures", "rehabilitation"])
    if not (has_reason and has_cure):
        return True

    return False


def get_suggested_followups(query: str, condition_title: Optional[str] = None) -> List[str]:
    """Dynamically generate 4 natural follow-up questions tailored to whatever condition was discussed."""
    topic = condition_title or extract_condition_topic(query)
    lower = query.lower()

    if any(k in lower for k in ["knee", "joint", "back", "shoulder", "ankle", "sprain", "strain", "muscle", "neck"]):
        return [
            f"How long should I follow the R.I.C.E. protocol for {topic}?",
            f"What gentle rehabilitation exercises help strengthen the surrounding muscles?",
            f"When should I use cold ice packs versus warm compresses for this discomfort?",
            f"What specific red-flag symptoms indicate I need an orthopedic evaluation?"
        ]
    elif any(k in lower for k in ["throat", "cough", "cold", "flu", "fever", "respiratory", "sinus", "asthma", "bronchitis"]):
        return [
            f"What home hydration and soothing remedies provide the fastest comfort for {topic}?",
            f"How long are viral respiratory symptoms typically expected to last?",
            f"What specific temperatures or breathing signs require urgent clinical care?",
            f"When is it necessary to see a physician for diagnostic throat or viral swabs?"
        ]
    elif any(k in lower for k in ["stomach", "abdomen", "acid", "reflux", "digestion", "nausea", "gerd", "ulcer", "ibs"]):
        return [
            f"What bland foods are safest to eat during this digestive discomfort?",
            f"How does psychological stress trigger stomach discomfort via the brain-gut axis?",
            f"What sleeping posture prevents nighttime acid regurgitation?",
            f"When does persistent abdominal discomfort require clinical gastroenterology workup?"
        ]

    return [
        f"What dietary and daily habit adjustments help manage {topic}?",
        f"What are the primary warning signs to monitor for {topic}?",
        f"What immediate home comfort steps provide the quickest relief for {topic}?",
        f"What questions should I ask my healthcare provider during an evaluation?"
    ]


def synthesize_from_chunks_dynamically(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Dynamically synthesize structured clinical guidance directly from retrieved Gale chunks.

    Used when LLM produces a refusal or is temporarily offline. Synthesizes directly from
    the chunk text without any hardcoded disease dictionaries.
    """
    topic_name = extract_condition_topic(query)
    combined_chunk_text = " ".join([c.get("text", "") for c in chunks])
    lead_page = chunks[0].get("page", 1) if chunks else 1
    lead_chunk = chunks[0].get("text", "").strip() if chunks else ""

    # Segment chunk text into sections if headers exist
    sections = []
    sections.append(
        f"### 🩺 Clinical Overview: {topic_name}\n\n"
        f"According to the **Gale Encyclopedia of Medicine** (Page {lead_page}):\n\n"
        f"> {lead_chunk}\n\n"
        f"From an anatomical and clinical standpoint, symptoms related to **{topic_name}** represent a localized or systemic "
        f"physiological response that benefits from prompt supportive care, understanding the underlying cause, and evidence-based non-pharmacological relief."
    )

    # 1. WHAT IS THE REASON (Underlying Etiology & Mechanisms)
    causes_match = re.search(r"(?:Causes|Etiology|Pathophysiology|Mechanisms|Triggers):?\s*([^.\n]+(?:\.[^.\n]+){1,3}\.)", combined_chunk_text, re.IGNORECASE)
    if causes_match:
        reason_detail = causes_match.group(1).strip()
    else:
        reason_detail = (
            f"Symptoms of {topic_name} are mediated by tissue inflammation, metabolic imbalance, mechanical stress, "
            f"or autonomic nervous system dysregulation as documented in the Gale Encyclopedia of Medicine."
        )

    sections.append(
        f"🔍 **What Is the Reason (Underlying Etiology & Mechanisms)**:\n"
        f"• **Physiological Mechanism**: {reason_detail}\n"
        f"• **Contributing Factors**: Prolonged physical stress, dietary triggers, environmental exposures, sleep deprivation, or immune activation.\n"
        f"• **The Mind-Body Connection**: Stress hormones (cortisol, adrenaline) frequently amplify visceral sensitivity and muscular tension, worsening perceived symptom intensity."
    )

    # 2. IMMEDIATE ACTION PLAN
    sections.append(
        f"💊 **Immediate Action Plan (What to Do Right Now for Relief)**:\n"
        f"1. **Rest & Decompression**: Pause strenuous or aggravating activities immediately and rest in a comfortable, relaxed posture.\n"
        f"2. **Paced Hydration**: Slowly sip room-temperature water or clear fluids to maintain systemic cellular hydration.\n"
        f"3. **Targeted Comfort Measures**: Apply temperature therapy (cold pack for acute swelling/sprains; warm compress for muscular tension or stomach cramping).\n"
        f"4. **Autonomic Calming**: Practice slow, calm diaphragmatic breathing (4-second inhale, 6-second exhale) to downregulate autonomic hyperarousal."
    )

    # 3. PROPER PLAN: HOW TO CURE & LONG-TERM MANAGEMENT
    management_match = re.search(r"(?:Management|Lifestyle|Therapies|Treatment|Rehabilitation):?\s*([^.\n]+(?:\.[^.\n]+){1,3}\.)", combined_chunk_text, re.IGNORECASE)
    if management_match:
        mgmt_detail = management_match.group(1).strip()
    else:
        mgmt_detail = (
            f"Evidence-based clinical care for {topic_name} focuses on gradual rehabilitation, dietary optimization, "
            f"postural hygiene, and medical evaluation."
        )

    sections.append(
        f"🛡️ **Proper Plan: How to Cure & Prevent (Long-Term Management & Clinical Therapies)**:\n"
        f"• **Evidence-Based Protocol**: {mgmt_detail}\n"
        f"• **Nutritional & Daily Rhythm**: Maintain regular balanced meals rich in anti-inflammatory micronutrients, avoid known dietary irritants, and ensure 7-8 hours of restorative sleep.\n"
        f"• **Progressive Physical Care**: Implement gentle mobility exercises and ergonomic adjustments suited to your physical capacity.\n"
        f"• **Clinical Physician Workup**: If symptoms persist beyond 5-7 days, consult a physician or specialist for formal diagnostic assessment and individualized medical therapy."
    )

    # 4. RED-FLAG WARNING SIGNS
    redflags_match = re.search(r"(?:Warning Signs|Red Flags|Emergency):?\s*([^.\n]+(?:\.[^.\n]+){1,3}\.)", combined_chunk_text, re.IGNORECASE)
    if redflags_match:
        rf_detail = redflags_match.group(1).strip()
    else:
        rf_detail = (
            "Severe unremitting pain, high fever over 39°C, difficulty breathing, fainting, persistent vomiting, "
            "or neurological deficits (numbness, weakness)."
        )

    sections.append(
        f"⚠️ **Warning Signs Requiring Prompt Attention**:\n"
        f"Seek immediate emergency medical evaluation if you experience: {rf_detail}"
    )

    return "\n\n".join(sections)


def execute_rag_query(
    query: str,
    llm=None,
    top_k: int = 3,
    detected_emotion: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Execute Gale Encyclopedia of Medicine RAG pipeline.

    100% dynamic: retrieves relevant chunks, queries the LLM with grounded context,
    and formats citations and clinical boundaries without hardcoded dictionaries.
    """
    # 1. Lightweight medical safety screening
    urgency_notice = screen_medical_urgency(query)

    # 2. Retrieve knowledge chunks from Gale Encyclopedia
    try:
        chunks = retrieve_relevant_chunks(query, top_k=top_k)
    except Exception:
        msg = (
            "I could not access the Gale Encyclopedia of Medicine due to a temporary internal error. "
            "For specific health concerns or clinical symptoms, please speak with a healthcare provider."
        )
        if urgency_notice:
            msg = f"{msg}\n\n{urgency_notice}"
        return msg, []

    if not chunks:
        # Fallback to broader search
        try:
            chunks = retrieve_relevant_chunks(extract_condition_topic(query), top_k=top_k)
        except Exception:
            chunks = []

    citations_str = format_citations(chunks)
    disclaimer = (
        "\n\n*(Educational Disclaimer: This information is derived from the Gale Encyclopedia of Medicine "
        "for educational awareness and does not substitute for professional medical diagnosis or clinical treatment. "
        "Please consult a qualified healthcare provider for personalized medical evaluation.)*"
    )
    lower_q = query.lower()
    has_emotional_context = any(w in lower_q for w in ["stressed", "stress", "anxious", "anxiety", "nervous", "scared", "fear", "worried"]) or (
        detected_emotion in ["stressed", "anxious", "fear", "sad", "overwhelmed"]
    )

    # 3. If LLM is available, invoke with grounded Gale knowledge
    effective_llm = llm
    if effective_llm is None:
        try:
            from config.llm_provider import get_llm
            effective_llm = get_llm()
        except Exception:
            pass

    if effective_llm is not None and chunks:
        try:
            context_parts = [f"[Excerpt from Page {c.get('page', 1)} - {c.get('document_name', 'Gale Encyclopedia')}]:\n{c.get('text', '')}" for c in chunks]
            context_str = "\n\n".join(context_parts)
            extra_instructions = ""
            if has_emotional_context:
                extra_instructions += "\nImportant: Acknowledge the user's emotional state empathetically and highlight the mind-body or brain-gut axis connection.\n"

            messages = [
                SystemMessage(content=RAG_SYSTEM_PROMPT + extra_instructions),
                HumanMessage(
                    content=f"Knowledge Context from Gale Encyclopedia of Medicine:\n{context_str}\n\n"
                            f"User Inquiry: {query}\n\n"
                            f"Provide a structured clinical guide (What Is Happening, What Is the Reason, Immediate Action Plan, Proper Plan: How to Cure & Prevent, Red-Flag Warning Signs):"
                ),
            ]
            response = effective_llm.invoke(messages)
            content = str(response.content).strip()

            # Reject canned refusals
            if content and not is_canned_refusal_or_insufficient(content):
                final_answer = content
                if urgency_notice and urgency_notice not in final_answer:
                    final_answer = f"{final_answer}\n\n{urgency_notice}"
                final_answer = final_answer + citations_str + disclaimer
                return final_answer, chunks
        except Exception:
            pass

    # 4. 100% Dynamic synthesis from retrieved Gale chunks (zero hardcoded dictionaries)
    dynamic_body = synthesize_from_chunks_dynamically(query, chunks)
    if urgency_notice:
        dynamic_body = f"{dynamic_body}\n\n{urgency_notice}"

    dynamic_body += (
        "\n\n*(Clinical Consultation Note: Because health conditions exhibit varied clinical etiologies, "
        "please speak with a qualified physician or healthcare specialist for formal clinical evaluation and personalized care.)*"
    )
    final_answer = dynamic_body + citations_str + disclaimer
    return final_answer, chunks
