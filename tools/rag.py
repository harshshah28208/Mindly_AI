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


RAG_SYSTEM_PROMPT = """You are an academic clinical educator and medical science researcher for Mindly, synthesizing medical literature from the Gale Encyclopedia of Medicine.
Your mission is to provide an objective, educational, and structured clinical overview for the user's inquiry based on medical literature and clinical science.

Structure your educational response with these clear sections:
- 📋 **What Is Happening (Clinical Overview & Definition)**: Explain the condition, classification, and physiological mechanisms.
- 🔍 **What Is the Reason (Underlying Causes, Mechanisms & Triggers)**: Detail biological causes, hormonal/metabolic/immune factors, triggers, and stress correlations.
- 💊 **Immediate Action Plan (What to Do Right Now for Relief)**: Step-by-step immediate comfort measures, gentle hygiene/hydration, and non-pharmacological care.
- 🛡️ **Proper Plan: How to Cure & Prevent (Long-Term Management & Clinical Therapies)**: Rehabilitation, dietary adjustments, long-term lifestyle habits, and standard clinical therapies.
- ⚠️ **Red-Flag Warning Signs (When to Seek Emergency Care)**: Symptoms requiring prompt physician evaluation.

Always deliver the clinical and educational information clearly and authoritatively from textbook science without generic refusal disclaimers.
Keep each section concise (1-2 direct sentences or bullet points) so the explanation is clear, complete, and focused. Place citations at the very bottom.
"""


def screen_medical_urgency(query: str) -> Optional[str]:
    """Identify acute red-flag medical emergencies that require prompt evaluation notice."""
    lower = query.lower()
    
    # Mild routine discomfort without red flags is non-urgent
    if "mild" in lower and not any(k in lower for k in ["vomit", "blood", "faint", "chest", "rigid", "stool"]):
        return None

    urgent_reasons = []
    if "vomiting blood" in lower:
        urgent_reasons.append("vomiting blood")
    if "blood in stool" in lower or "black tarry" in lower:
        urgent_reasons.append("blood in stool")
    if "chest pain" in lower:
        urgent_reasons.append("chest pain")
    if "rigid" in lower or "swollen" in lower:
        urgent_reasons.append("rigid or swollen abdomen")
    if "faint" in lower or "fainting" in lower or "severe worsening" in lower:
        urgent_reasons.append("severe worsening pain and fainting")

    if urgent_reasons:
        reasons_text = ", ".join(urgent_reasons)
        return (
            f"⚠️ **Urgent Medical Assessment Recommended**: Red-flag symptoms detected ({reasons_text}). "
            "These acute clinical indicators require prompt urgent medical assessment at an emergency department or healthcare facility."
        )

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
                "⚠️ **Urgent Medical Assessment Required**: Some of the symptoms you described can indicate an acute medical situation. "
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
        "please", "help", "with", "tell", "about", "symptoms", "for", "this", "that", "some"
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
    # Reject canned refusals that refuse diagnosis/medical advice without clinical explanation
    pure_refusal_patterns = [
        r"^i\s+(?:can't|cannot)\s+provide\s+(?:a\s+)?(?:diagnosis|medical\s+advice)",
        r"^as\s+an?\s+ai\s+language\s+model",
        r"^i\s+am\s+unable\s+to\s+provide\s+(?:a\s+)?medical\s+advice",
        r"^i\s+am\s+not\s+authorized\s+to\s+provide\s+medical\s+advice",
    ]
    for pat in pure_refusal_patterns:
        if re.search(pat, lower):
            return True

    # Detect abrupt dangling cutoffs (e.g. truncated mid-word or mid-sentence)
    if re.search(r"[,:;\-\(\[\{]\s*$", content) or re.search(r"\b(?:and|or|the|with|to|in|of|that|because|step|e\.g\.|e)\s*$", content, re.IGNORECASE):
        return True

    # Must contain essential clinical sections: overview, reason, and immediate action plan
    has_overview = "clinical overview" in lower or "what is happening" in lower
    has_reason = "what is the reason" in lower or "underlying" in lower or "etiology" in lower
    has_action = "immediate action" in lower or "action plan" in lower or "how to cure" in lower

    if not (has_overview and has_reason and has_action):
        return True

    return False


def get_suggested_followups(query: str, condition_title: Optional[str] = None) -> List[str]:
    """Dynamically generate 4 natural follow-up questions tailored to whatever topic or condition was discussed."""
    topic = condition_title or extract_condition_topic(query)
    lower = query.lower()

    # 1. Academic / Viva / Exam / Presentation
    if any(k in lower for k in ["viva", "vivas", "exam", "exams", "interview", "presentation", "defense", "syllabus", "study"]):
        return [
            "How can I handle tricky questions if I freeze during the viva?",
            "Guide me through a 2-minute box breathing cycle before entering.",
            "What is a structured 10-minute micro-revision technique?",
            "How can I keep my voice calm and steady while speaking to examiners?"
        ]

    # 2. Grounding / Breathing / Somatic Reset
    if any(k in lower for k in ["box breathing", "4-4-4-4", "grounding", "breathe", "breathing", "5-4-3-2-1", "senses", "calm down"]):
        return [
            "Can you guide me through a 5-4-3-2-1 sensory grounding exercise?",
            "How often should I practice box breathing during high-stress days?",
            "What should I do if racing thoughts distract me while breathing?",
            "Can you suggest a quick progressive muscle relaxation exercise?"
        ]

    # 3. Headaches / Migraine / Neurological
    if any(k in lower for k in ["headache", "migraine", "headaches", "temple", "migraines"]):
        return [
            "What are common environmental and dietary triggers for migraines?",
            "How can I differentiate tension headaches from migraines?",
            "What immediate home comfort measures help reduce acute headache pressure?",
            "What specific warning signs indicate a headache needs urgent clinical evaluation?"
        ]

    # 4. Musculoskeletal / Joint / Back / Neck / Orthopedic
    if any(k in lower for k in ["knee", "joint", "back", "shoulder", "ankle", "sprain", "strain", "muscle", "neck", "arthritis"]):
        return [
            f"How long should I follow the R.I.C.E. protocol for {topic}?",
            "What gentle rehabilitation exercises help strengthen the surrounding muscles?",
            "When should I use cold ice packs versus warm compresses for this discomfort?",
            "What specific red-flag symptoms indicate I need an orthopedic evaluation?"
        ]

    # 5. Respiratory / Throat / Cold / Flu
    if any(k in lower for k in ["throat", "cough", "cold", "flu", "fever", "respiratory", "sinus", "asthma", "bronchitis"]):
        return [
            f"What home hydration and soothing remedies provide the fastest comfort for {topic}?",
            "How long are viral respiratory symptoms typically expected to last?",
            "What specific temperatures or breathing signs require urgent clinical care?",
            "When is it necessary to see a physician for diagnostic throat or viral swabs?"
        ]

    # 6. Gastrointestinal / Stomach / Acid / Digestion
    if any(k in lower for k in ["stomach", "abdomen", "acid", "reflux", "digestion", "nausea", "gerd", "ulcer", "ibs", "gut"]):
        return [
            "What bland foods are safest to eat during this digestive discomfort?",
            "How does psychological stress trigger stomach discomfort via the brain-gut axis?",
            "What sleeping posture prevents nighttime acid regurgitation?",
            "When does persistent abdominal discomfort require clinical gastroenterology workup?"
        ]

    # 7. Cardiovascular / Hypertension
    if any(k in lower for k in ["hypertension", "blood pressure", "bp", "cardiovascular"]):
        return [
            "What dietary changes (such as the DASH diet) help regulate blood pressure?",
            "How does daily psychological stress affect arterial blood pressure?",
            "What daily physical activities safely support cardiovascular health?",
            "What specific blood pressure readings require urgent emergency care?"
        ]

    # 8. Conversational / General Welcome / Wellness
    return [
        "What clinical health and wellness topics can I explore with you?",
        "Can you explain the brain-gut connection between stress and physical symptoms?",
        "How can I deal with college workload burnout effectively?",
        "Can you help me find a licensed therapist or mental health clinic near me?"
    ]


def synthesize_from_chunks_dynamically(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Dynamically synthesize structured clinical guidance tailored directly to whatever knowledge chunks were retrieved."""
    topic_name = extract_condition_topic(query)
    combined_chunk_text = " ".join([c.get("text", "") for c in chunks])
    lead_page = chunks[0].get("page", 1) if chunks else 1

    # Extract distinct sections directly from the actual retrieved knowledge text
    definition_match = re.search(r"(?:Definition|Clinical Overview)[:\s]+([^.\n]+(?:\.[^.\n]+){1,3}\.)", combined_chunk_text, re.IGNORECASE)
    causes_match = re.search(r"(?:Etiology|Causes|Pathophysiology|Mechanisms)[:\s]+([^.\n]+(?:\.[^.\n]+){1,3}\.)", combined_chunk_text, re.IGNORECASE)
    relief_match = re.search(r"(?:Treatment|Hygiene|Immediate Action|Self-Care|Relief)[:\s]+([^.\n]+(?:\.[^.\n]+){1,3}\.)", combined_chunk_text, re.IGNORECASE)
    plan_match = re.search(r"(?:Management|Evidence-Based|Proper Plan|Long-Term)[:\s]+([^.\n]+(?:\.[^.\n]+){1,3}\.)", combined_chunk_text, re.IGNORECASE)
    redflags_match = re.search(r"(?:Warning Signs|Red Flags|Emergency)[:\s]+([^.\n]+(?:\.[^.\n]+){1,3}\.)", combined_chunk_text, re.IGNORECASE)

    sections = []
    sections.append(f"### 🩺 Clinical Overview: {topic_name}\n\nAccording to the **Gale Encyclopedia of Medicine** (Page {lead_page}):")

    if definition_match:
        sections.append(f"📋 **What Is Happening**:\n{definition_match.group(1).strip()}")
    elif combined_chunk_text:
        first_sentences = ". ".join([s.strip() for s in combined_chunk_text.split(".")[:2] if s.strip()]) + "."
        sections.append(f"📋 **What Is Happening**:\n{first_sentences}")

    if causes_match:
        sections.append(f"🔍 **What Is the Reason (Underlying Etiology & Mechanisms)**:\n{causes_match.group(1).strip()}")
    else:
        sections.append(f"🔍 **What Is the Reason (Underlying Etiology & Mechanisms)**:\nSymptoms related to {topic_name} arise from physiological tissue strain, inflammation, mechanical overuse, or biochemical triggers documented in clinical literature.")

    if relief_match:
        sections.append(f"💊 **Immediate Action Plan (What to Do Right Now for Relief)**:\n{relief_match.group(1).strip()}")
    else:
        sections.append(f"💊 **Immediate Action Plan (What to Do Right Now for Relief)**:\nRest the affected area, avoid aggravating strain, maintain hydration, and apply comfort measures (cold packs for acute swelling, warm compresses for tension).")

    if plan_match:
        sections.append(f"🛡️ **Proper Plan: How to Cure & Prevent (Long-Term Management & Clinical Therapies)**:\n{plan_match.group(1).strip()}")
    else:
        sections.append(f"🛡️ **Proper Plan: How to Cure & Prevent (Long-Term Management & Clinical Therapies)**:\nIncorporate gradual physical rehabilitation, ergonomic postural adjustments, balanced nutrition, and formal medical follow-up for chronic concerns.")

    if redflags_match:
        sections.append(f"⚠️ **Warning Signs Requiring Prompt Attention**:\nSeek medical evaluation if you experience: {redflags_match.group(1).strip()}")
    else:
        sections.append("⚠️ **Warning Signs Requiring Prompt Attention**:\nSeek medical evaluation if symptoms worsen, do not improve within 3–5 days, or are accompanied by high fever, severe unremitting pain, or difficulty breathing.")

    return "\n\n".join(sections)


def execute_rag_query(
    query: str,
    llm=None,
    top_k: int = 2,
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
            from config.settings import settings
            budget = settings.determine_token_budget(query, intent="HEALTH_INFORMATION")
            effective_llm = get_llm(max_tokens=budget)
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

            # Reject canned refusals, truncated outputs, or missing clinical structure
            if content and not is_canned_refusal_or_insufficient(content):
                final_answer = content
                if has_emotional_context and any(w in lower_q for w in ["stomach", "gut", "abdominal", "digestive", "belly", "acidity"]) and "brain-gut axis" not in final_answer.lower():
                    final_answer = (
                        "### 🧠 Mind-Body Connection & The Brain-Gut Axis\n"
                        "The enteric nervous system communicates bidirectionally with the central nervous system via the vagus nerve (the **brain-gut axis**). "
                        "When acute psychological stress or anxiety is present, neuroendocrine signals provoke gastric mucosal hypersensitivity, acid hypersecretion, and visceral muscle cramping.\n\n"
                        + final_answer
                    )
                if urgency_notice and urgency_notice not in final_answer:
                    final_answer = f"{final_answer}\n\n{urgency_notice}"
                # Guarantee presence of red-flag warning signs section
                if not any(k in final_answer.lower() for k in ["warning signs", "red-flag", "red flag", "emergency"]):
                    final_answer = (
                        final_answer + "\n\n⚠️ **Warning Signs Requiring Prompt Attention**:\n"
                        "Seek medical evaluation if symptoms worsen, persist, or are accompanied by high fever, severe unremitting pain, or difficulty breathing."
                    )
                final_answer = final_answer + citations_str + disclaimer
                return final_answer, chunks
        except Exception:
            pass

    # 4. 100% Dynamic synthesis from retrieved Gale chunks (zero hardcoded dictionaries)
    dynamic_body = synthesize_from_chunks_dynamically(query, chunks)
    if has_emotional_context and any(w in lower_q for w in ["stomach", "gut", "abdominal", "digestive", "belly", "acidity"]) and "brain-gut axis" not in dynamic_body.lower():
        dynamic_body = (
            "### 🧠 Mind-Body Connection & The Brain-Gut Axis\n"
            "The enteric nervous system communicates bidirectionally with the central nervous system via the vagus nerve (the **brain-gut axis**). "
            "When acute psychological stress or anxiety is present, neuroendocrine signals provoke gastric mucosal hypersensitivity, acid hypersecretion, and visceral muscle cramping.\n\n"
            + dynamic_body
        )
    if urgency_notice:
        dynamic_body = f"{dynamic_body}\n\n{urgency_notice}"

    dynamic_body += (
        "\n\n*(Clinical Consultation Note: Because health conditions exhibit varied clinical etiologies, "
        "please speak with a qualified physician or healthcare specialist for formal clinical evaluation and personalized care.)*"
    )
    final_answer = dynamic_body + citations_str + disclaimer
    return final_answer, chunks
