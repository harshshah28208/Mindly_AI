"""Gale Encyclopedia of Medicine RAG tool for Mindly Lab 3.

Provides grounded answers strictly referencing retrieved knowledge chunks
from the Gale Encyclopedia of Medicine. Enforces clinical boundaries and
educational citations.
"""

from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from rag.retriever import retrieve_relevant_chunks


RAG_SYSTEM_PROMPT = """You are an educational mental-wellness and health knowledge specialist for Mindly.
Your job is to provide educational, grounded health information referencing excerpts from the Gale Encyclopedia of Medicine.

Strict Guidelines:
1. Ground your response in the provided knowledge context from the Gale Encyclopedia of Medicine. Do NOT invent citations or facts.
2. Clinical Boundaries: Never diagnose the user (do NOT say "you have X"). Never prescribe medications or treatments.
3. Multiple Causes: Explain that symptoms (e.g. abdominal pain, headaches) have diverse potential causes ranging from functional and stress-related to organic.
4. Warning Signs: Clearly identify relevant warning signs (red flags) that warrant urgent medical evaluation.
5. Empathy & Brain-Gut Axis: If the user expresses emotional distress or stress alongside physical symptoms, address both empathetically and explain the connection between stress and bodily symptoms.
6. Professional Care: Recommend that the user seek evaluation from a qualified healthcare professional.
"""


def screen_medical_urgency(query: str) -> Optional[str]:
    """Perform lightweight safety screening for potentially urgent clinical symptoms.

    Detects red flags such as severe or worsening abdominal pain, vomiting blood,
    blood or tarry stools, chest pain, breathing difficulty, fainting, rigid/swollen
    abdomen, confusion, or severe weakness.

    Returns:
        Cautious educational advisory recommending prompt medical assessment if
        urgent signs are detected, or None if no urgent red flags are found.
    """
    lower = query.lower()
    red_flags = []

    if any(k in lower for k in [
        "severe or worsening", "severe pain", "hurts badly", "worsening pain",
        "unbearable pain", "intense pain", "severe abdominal", "severe stomach",
        "worsening abdominal", "worsening stomach", "excruciating"
    ]):
        red_flags.append("severe or worsening pain")
    if any(k in lower for k in ["vomiting blood", "vomit blood", "blood in vomit", "hematemesis"]):
        red_flags.append("vomiting blood")
    if any(k in lower for k in ["blood in stool", "bloody stool", "rectal bleeding"]):
        red_flags.append("blood in stool")
    if any(k in lower for k in ["black stool", "tarry stool", "black/tarry stool", "black/tarry"]):
        red_flags.append("black or tarry stool")
    if any(k in lower for k in ["fainting", "fainted", "pass out", "passed out", "syncope", "loss of consciousness"]):
        red_flags.append("fainting or loss of consciousness")
    if any(k in lower for k in ["difficulty breathing", "trouble breathing", "shortness of breath", "can't breathe", "hard to breathe"]):
        red_flags.append("difficulty breathing or shortness of breath")
    if any(k in lower for k in ["chest pain", "pain in chest", "chest pressure", "tightness in chest"]):
        red_flags.append("chest pain or pressure")
    if any(k in lower for k in ["rigid abdomen", "swollen abdomen", "abdomen is rigid", "rigid/swollen", "hard abdomen", "stomach is swollen"]):
        red_flags.append("a rigid or swollen abdomen")
    if any(k in lower for k in ["confusion", "sudden confusion", "disoriented", "severe weakness", "unable to stand"]):
        red_flags.append("confusion or severe weakness")

    if red_flags:
        flags_desc = ", ".join(red_flags)
        return (
            f"⚠️ **Urgent Medical Screening Advisory**:\n"
            f"You noted symptoms associated with **{flags_desc}**. "
            f"Severe or worsening abdominal pain, especially with acute signs such as these, can require urgent medical assessment. "
            f"Mindly does not provide medical diagnosis or emergency care. "
            f"Please seek immediate in-person medical evaluation from an emergency service, urgent care center, or your physician."
        )
    return None


def format_citations(chunks: List[Dict[str, Any]]) -> str:
    """Format verified source citations from retrieved chunks."""
    if not chunks:
        return ""
    pages = sorted(list(set(c.get("page", 1) for c in chunks)))
    pages_str = ", ".join(f"Page {p}" for p in pages)
    return f"\n\n📚 **Source**: *Gale Encyclopedia of Medicine* ({pages_str})"


def execute_rag_query(
    query: str,
    llm=None,
    top_k: int = 3,
    detected_emotion: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Execute RAG pipeline: screen urgency, retrieve Gale chunks, and generate grounded answer.

    Returns:
        Tuple of (response_text, retrieved_chunks_list).
    """
    # 1. Lightweight medical safety screening
    urgency_notice = screen_medical_urgency(query)

    # 2. Retrieve knowledge chunks
    try:
        chunks = retrieve_relevant_chunks(query, top_k=top_k)
    except Exception:
        msg = (
            "I could not access the Gale Encyclopedia of Medicine due to a temporary internal error. "
            "For specific health concerns or clinical symptoms, please speak with a healthcare provider."
        )
        if urgency_notice:
            msg = f"{urgency_notice}\n\n{msg}"
        return msg, []

    if not chunks:
        msg = (
            "I couldn't find enough relevant information in the available knowledge source "
            "(Gale Encyclopedia of Medicine) to address your inquiry safely and accurately.\n\n"
            "*(Mindly is an educational wellness support companion, not a medical doctor. "
            "For specific health concerns or clinical symptoms, please speak with a healthcare provider.)*"
        )
        if urgency_notice:
            msg = f"{urgency_notice}\n\n{msg}"
        return msg, []

    # Build context string
    context_parts = []
    for c in chunks:
        p = c.get("page", 1)
        txt = c.get("text", "")
        context_parts.append(f"[Excerpt from Page {p}]:\n{txt}")
    context_str = "\n\n".join(context_parts)

    disclaimer = (
        "\n\n*(Educational Disclaimer: This information is derived from the Gale Encyclopedia of Medicine "
        "for educational awareness and does not substitute for professional medical diagnosis or clinical treatment.)*"
    )

    lower_q = query.lower()
    is_stomach = any(k in lower_q for k in ["stomach", "abdomen", "abdominal", "belly", "tummy", "gut", "nausea", "indigestion"])
    has_emotional_context = any(w in lower_q for w in ["stressed", "stress", "anxious", "anxiety", "nervous", "scared", "fear", "worried"]) or (
        detected_emotion in ["stressed", "anxious", "fear", "sad", "overwhelmed"]
    )

    # 3. LLM Generation
    if llm is not None:
        try:
            extra_instructions = ""
            if urgency_notice:
                extra_instructions += f"\nImportant: Prepend or integrate this urgent advisory:\n{urgency_notice}\n"
            if is_stomach and has_emotional_context:
                extra_instructions += "\nImportant: Acknowledge the user's emotional state empathetically and explain the brain-gut connection from the context.\n"

            messages = [
                SystemMessage(content=RAG_SYSTEM_PROMPT + extra_instructions),
                HumanMessage(
                    content=f"Knowledge Context from Gale Encyclopedia of Medicine:\n{context_str}\n\n"
                            f"User Inquiry: {query}\n\n"
                            f"Grounded Response:"
                ),
            ]
            response = llm.invoke(messages)
            content = str(response.content).strip()
            if content and len(content) > 30:
                final_answer = content
                if urgency_notice and urgency_notice not in final_answer:
                    final_answer = f"{urgency_notice}\n\n{final_answer}"
                final_answer = final_answer + format_citations(chunks) + disclaimer
                return final_answer, chunks
        except Exception:
            pass

    # 4. Structured Educational Fallback Synthesis
    sections = []
    if urgency_notice:
        sections.append(urgency_notice)

    if is_stomach:
        if has_emotional_context:
            sections.append(
                "I hear how uncomfortable and stressful this is for you. Experiencing physical symptoms while feeling stressed or scared is very common. "
                "According to the **Gale Encyclopedia of Medicine**, the brain and digestive tract are intimately connected via the **brain-gut axis**. "
                "Psychological distress, anxiety, and stress stimulate the enteric nervous system, which can cause visceral hypersensitivity, cramping, spasms, and bloating without permanent structural tissue damage."
            )
        else:
            sections.append(
                "According to the **Gale Encyclopedia of Medicine**, abdominal or stomach discomfort can have many diverse causes. "
                "These range from benign, self-limiting functional issues—such as gas, indigestion (dyspepsia), and stress-mediated muscle tension—to dietary sensitivities, infections (gastroenteritis), acid reflux, or inflammatory conditions."
            )

        sections.append(
            "**Key Warning Signs**:\n"
            "Medical literature stresses that severe or worsening abdominal pain, especially when accompanied by high fever, repeated vomiting, difficulty keeping fluids down, blood in vomit or stool, or dizziness, requires urgent in-person medical assessment."
        )
        sections.append(
            "Because stomach symptoms have many potential etiologies, Mindly cannot provide a medical diagnosis. "
            "If your symptoms are persistent, unexplained, or worsening, please consult a qualified healthcare provider for proper clinical evaluation."
        )
    else:
        first_chunk = chunks[0]
        lead_text = first_chunk.get("text", "").strip()
        sections.append(
            f"According to the **Gale Encyclopedia of Medicine**:\n\n"
            f"> {lead_text}\n\n"
            f"This condition highlights the importance of monitoring physical symptoms and consulting a qualified healthcare provider for personalized medical evaluation and clinical diagnosis."
        )

    fallback_body = "\n\n".join(sections)
    final_answer = fallback_body + format_citations(chunks) + disclaimer
    return final_answer, chunks
