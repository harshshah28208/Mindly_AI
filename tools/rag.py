"""Gale Encyclopedia of Medicine RAG tool for Mindly.

Provides grounded answers strictly referencing retrieved knowledge chunks
from the Gale Encyclopedia of Medicine. Enforces clinical boundaries and
educational citations.
"""

from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from rag.retriever import retrieve_relevant_chunks


RAG_SYSTEM_PROMPT = """You are an expert clinical health educator and medical knowledge specialist for Mindly, grounded in the Gale Encyclopedia of Medicine.
When responding to any disease, illness, physical symptom, or health inquiry:
1. Provide a comprehensive, highly helpful, and well-structured clinical guide.
2. Structure your answer with clear markdown sections:
   - 📋 **Clinical Overview & Definition**: Explain the condition, classification, and physiological mechanism.
   - 🔍 **Causes, Risk Factors & Etiology**: Why it develops, contributing risk factors (e.g. genetics, stress, brain-gut axis, lifestyle).
   - ⚠️ **Key Symptoms & Red-Flag Signs**: Observable symptoms and acute warning signs requiring emergency medical attention.
   - 💊 **Evidence-Based Management & Helpful Therapies**: What can help (evidence-based lifestyle changes, dietary support, practical home comfort measures, and standard clinical treatments).
   - 🛡️ **Prevention & Daily Self-Care**: Practical habits to prevent recurrence or progression.
3. PRESENT THE EDUCATIONAL AND CLINICAL KNOWLEDGE FIRST. Explain the condition and helpful measures before stating any referral disclaimers.
4. Place citations (*Gale Encyclopedia of Medicine*) and the professional medical consultation advisory at the very BOTTOM.
5. Strict Boundaries: Do not state personal diagnoses (never say "you have X") or prescribe specific prescription dosages.
"""


# Comprehensive Clinical Reference Knowledge for Common & Critical Conditions
CLINICAL_KNOWLEDGE_ENTRIES = {
    "hypertension": {
        "title": "Hypertension (High Blood Pressure)",
        "page": 1,
        "definition": (
            "Hypertension is a chronic cardiovascular condition characterized by persistent elevation of arterial blood pressure "
            "(systolic pressure ≥ 130 mmHg and/or diastolic pressure ≥ 80 mmHg). Often termed the 'silent killer', it forces the heart "
            "to pump against excessive vascular resistance, increasing the workload of the cardiac muscle and straining systemic arterial walls."
        ),
        "causes": (
            "• **Essential (Primary) Hypertension**: Accounts for ~90-95% of cases, arising from a complex interplay of genetic predisposition, "
            "sympathetic nervous system overactivity, excessive dietary sodium intake, arterial stiffening, and chronic psychological stress.\n"
            "• **Secondary Hypertension**: Caused by identifiable underlying conditions such as renal artery stenosis, chronic kidney disease, "
            "endocrine disorders (e.g., Cushing syndrome, hyperaldosteronism), or obstructive sleep apnea."
        ),
        "symptoms": (
            "• **Asymptomatic in Early Stages**: Most individuals experience no noticeable symptoms until blood pressure reaches severe levels.\n"
            "• **Associated Symptoms**: Occipital morning headaches, dizziness, blurred vision, mild chest tightness, and facial flushing."
        ),
        "management": (
            "• **DASH Dietary Pattern**: Emphasize potassium-rich foods (leafy greens, bananas, legumes), reduce dietary sodium intake (< 1,500-2,300 mg/day), and minimize saturated fats.\n"
            "• **Aerobic Exercise**: Engage in 150 minutes per week of moderate-intensity cardiovascular exercise (brisk walking, swimming, cycling).\n"
            "• **Stress Reduction & Autonomic Calming**: Practice slow-paced diaphragmatic breathing and box breathing to reduce sympathetic tone.\n"
            "• **Clinical Pharmacotherapy**: When indicated by a physician, standard clinical treatments include ACE inhibitors, Angiotensin Receptor Blockers (ARBs), calcium channel blockers, and thiazide diuretics."
        ),
        "warning_signs": (
            "**Hypertensive Crisis (≥ 180/120 mmHg)**: Immediate emergency evaluation is required if high blood pressure is accompanied by "
            "severe crushing chest pain, acute shortness of breath, sudden numbness or speech difficulty, or severe thunderclap headache."
        ),
        "prevention": "Maintain a healthy body mass index (BMI), limit alcohol consumption, avoid tobacco, and perform regular blood pressure self-monitoring."
    },
    "migraine": {
        "title": "Migraine Headache Disorders",
        "page": 2,
        "definition": (
            "Migraine is a complex primary neurovascular disorder characterized by recurrent attacks of moderate to severe unilateral, "
            "throbbing headache pain, typically lasting between 4 to 72 hours. It is frequently accompanied by autonomic symptoms, photophobia, "
            "phonophobia, and nausea."
        ),
        "causes": (
            "• **Neurovascular Pathophysiology**: Waves of cortical spreading depression trigger trigeminovascular activation and the release of "
            "calcitonin gene-related peptide (CGRP) and substance P, leading to neurogenic inflammation of the cranial meninges.\n"
            "• **Common Triggers**: Disrupted sleep cycles, fluctuating stress levels, skipped meals, hormonal fluctuations (estrogen shifts), "
            "dehydration, bright lights, sensory overload, and dietary triggers (tyramine-containing cheeses, nitrates, alcohol)."
        ),
        "symptoms": (
            "• **Prodrome & Aura**: Fatigue, neck stiffness, mood shifts, followed in ~25% of individuals by visual scotomas, zigzag lines, or sensory paresthesia.\n"
            "• **Headache Phase**: Severe pulsating or throbbing pain, predominantly on one side of the head, worsened by routine physical activity.\n"
            "• **Associated Features**: Marked nausea, sensory hypersensitivity to light and sound, and postdrome 'brain fog'."
        ),
        "management": (
            "• **Acute Comfort Protocol**: Rest in a dark, quiet, temperature-controlled room with a cold compress placed over the forehead or neck.\n"
            "• **Hydration & Electrolytes**: Drink cool water slowly to correct potential dehydration.\n"
            "• **Clinical Treatments**: Medical abortive options (prescribed by a physician) include triptans (e.g. sumatriptan), NSAIDs, antiemetics, and modern CGRP antagonists.\n"
            "• **Preventive Therapies**: Beta-blockers, topiramate, CGRP monoclonal antibodies, and magnesium glycinate supplementation."
        ),
        "warning_signs": (
            "**Red-Flag Headaches (SNOOP criteria)**: Seek immediate emergency assessment if a headache is sudden and explosive ('worst headache of life'), "
            "or accompanied by high fever, stiff neck, confusion, seizure, or focal neurological deficits."
        ),
        "prevention": "Maintain consistent sleep and meal schedules, practice stress reduction, keep a trigger diary, and ensure adequate daily hydration."
    },
    "stomach": {
        "title": "Abdominal Pain & Gastrointestinal Disorders",
        "page": 5,
        "definition": (
            "Abdominal pain refers to discomfort or cramping felt in the peritoneal or gastrointestinal region. It encompasses a wide spectrum "
            "ranging from benign functional discomfort—such as stress-mediated dysmotility and indigestion—to organic inflammatory conditions "
            "like gastritis, peptic ulcer disease, GERD, and irritable bowel syndrome (IBS)."
        ),
        "causes": (
            "• **Brain-Gut Axis & Visceral Hypersensitivity**: The central nervous system and enteric nervous system communicate bidirectionally "
            "via the vagus nerve. Psychological stress and anxiety trigger the release of corticotropin-releasing factor, causing mucosal inflammation, "
            "altered peristalsis, visceral hypersensitivity, and painful cramping.\n"
            "• **Organic & Dietary Causes**: Gastric hyperacidity, Helicobacter pylori infection, non-steroidal anti-inflammatory drugs (NSAIDs), "
            "lactose intolerance, fatty foods, food allergies, or viral gastroenteritis."
        ),
        "symptoms": (
            "• **Localized or Diffuse Discomfort**: Epigastric burning, lower abdominal cramping, bloating, early satiety, and excessive flatulence.\n"
            "• **Associated Sensations**: Mild nausea, altered bowel frequency, and postprandial heaviness."
        ),
        "management": (
            "• **Gentle Digestive Rest**: Consume small, bland, easily digestible meals (e.g., BRAT diet: bananas, rice, applesauce, toast); avoid acidic, greasy, and spicy triggers.\n"
            "• **Hydration**: Sip room-temperature water, herbal chamomile, or peppermint tea slowly.\n"
            "• **Autonomic Calming**: Apply a warm heating pad to the abdomen and practice slow diaphragmatic breathing to downregulate enteric nerve spasms.\n"
            "• **Clinical Interventions**: When evaluated by a clinician, proton pump inhibitors (PPIs), H2 blockers, antacids, or antispasmodics may be indicated."
        ),
        "warning_signs": (
            "**Key Warning Signs (Red Flags)**: Severe or worsening abdominal pain, especially accompanied by high fever, persistent vomiting, "
            "inability to keep liquids down, blood in vomit (hematemesis) or stool, a rigid or distended abdomen, or fainting requires immediate emergency medical evaluation."
        ),
        "prevention": "Eat at regular intervals, avoid late-night heavy meals, manage emotional stressors, limit NSAID overuse, and stay well-hydrated."
    },
    "diabetes": {
        "title": "Diabetes Mellitus & Metabolic Regulation",
        "page": 1,
        "definition": (
            "Diabetes mellitus is a metabolic disorder characterized by persistent hyperglycemia (elevated blood glucose levels) resulting from "
            "defects in insulin secretion, insulin action, or both. Type 1 diabetes involves autoimmune destruction of pancreatic beta cells, "
            "while Type 2 diabetes involves progressive peripheral insulin resistance coupled with relative secretory deficiency."
        ),
        "causes": (
            "• **Type 2 Diabetes Etiology**: Genetic susceptibility, visceral adiposity, sedentary lifestyle, chronic low-grade inflammation, "
            "and hepatic glucose overproduction.\n"
            "• **Type 1 Diabetes Etiology**: Genetic predisposition (HLA complex) triggered by environmental factors leading to autoimmune beta-cell loss."
        ),
        "symptoms": (
            "• **Classic Osmotic Symptoms**: Polydipsia (excessive thirst), polyuria (frequent urination), and polyphagia (increased hunger).\n"
            "• **Systemic Manifestations**: Chronic fatigue, unexplained weight loss, blurred vision, slow-healing wounds, and recurrent infections."
        ),
        "management": (
            "• **Dietary Glycemic Control**: Emphasize complex low-glycemic carbohydrates, high-fiber legumes, vegetables, and lean proteins while restricting refined sugars.\n"
            "• **Physical Activity**: Regular aerobic and resistance training enhances skeletal muscle glucose uptake independently of insulin.\n"
            "• **Self-Monitoring**: Regular capillary blood glucose or continuous glucose monitoring (CGM) to evaluate glycemic stability.\n"
            "• **Clinical Pharmacotherapy**: Metformin, SGLT-2 inhibitors, GLP-1 receptor agonists, DPP-4 inhibitors, and insulin therapy under physician guidance."
        ),
        "warning_signs": (
            "**Acute Metabolic Emergencies**: Diabetic ketoacidosis (DKA) or hyperosmolar hyperglycemic state (HHS), characterized by fruity breath odor, "
            "rapid deep breathing (Kussmaul breathing), nausea, confusion, or severe lethargy, requires immediate emergency medical resuscitation."
        ),
        "prevention": "Weight management, routine physical activity, Mediterranean-style dietary habits, and early screening for prediabetes."
    },
    "asthma": {
        "title": "Asthma & Bronchial Hyperresponsiveness",
        "page": 1,
        "definition": (
            "Asthma is a chronic inflammatory disorder of the conducting airways characterized by bronchial hyperresponsiveness, mucosal edema, "
            "excessive mucus production, and variable, reversible airflow obstruction."
        ),
        "causes": (
            "• **Allergic (Extrinsic) Asthma**: IgE-mediated immunological response to airborne allergens such as pollen, dust mites, mold, and animal dander.\n"
            "• **Non-Allergic (Intrinsic) Asthma**: Triggered by viral respiratory infections, cold air inhalation, physical exertion, airborne irritants, "
            "and emotional distress/hyperventilation."
        ),
        "symptoms": (
            "• **Classic Triad**: Expiratory wheezing, shortness of breath (dyspnea), and chest tightness.\n"
            "• **Cough Variant**: Persistent dry cough, often exacerbated during the night or early morning."
        ),
        "management": (
            "• **Trigger Avoidance**: Eliminate environmental irritants (smoke, strong fragrances, dust mites) and use HEPA air filtration.\n"
            "• **Paced Respiration**: Sit upright, remain calm, and practice pursed-lip breathing during mild breathlessness.\n"
            "• **Clinical Inhaler Therapy**: Inhaled corticosteroids (ICS) for maintenance control and short-acting beta-agonists (SABA e.g. albuterol) "
            "as prescribed by a physician for rapid bronchodilation."
        ),
        "warning_signs": (
            "**Severe Respiratory Distress**: Inability to speak in full sentences, cyanosis (bluish lips or nails), sternal retractions, "
            "or lack of response to quick-relief bronchodilators necessitates urgent emergency medical care."
        ),
        "prevention": "Consistent controller medication adherence, annual influenza vaccination, and monitoring peak expiratory flow."
    },
    "arthritis": {
        "title": "Arthritis & Musculoskeletal Joint Disorders",
        "page": 1,
        "definition": (
            "Arthritis encompasses a diverse group of inflammatory and degenerative conditions affecting articular cartilage, synovial membranes, "
            "and periarticular tissues, leading to joint pain, stiffness, and diminished range of motion."
        ),
        "causes": (
            "• **Osteoarthritis**: Biomechanical wear-and-tear leading to progressive cartilage loss, subchondral bone remodeling, and osteophyte formation.\n"
            "• **Rheumatoid Arthritis**: Systemic autoimmune disease characterized by chronic synovial inflammation, joint erosion, and autoantibodies (RF, anti-CCP)."
        ),
        "symptoms": (
            "• **Joint Pain & Stiffness**: Morning stiffness lasting > 30 minutes (inflammatory) or brief inactivity stiffness (degenerative).\n"
            "• **Physical Signs**: Joint swelling, tenderness, crepitus (grating sensation), warmth, and restricted mobility."
        ),
        "management": (
            "• **Low-Impact Exercise**: Swimming, cycling, and tai chi to maintain joint flexibility and build periarticular muscle support.\n"
            "• **Thermal Therapy**: Warm baths or heating pads to relieve chronic stiffness; cold packs for acute joint flares.\n"
            "• **Anti-Inflammatory Nutrition**: Incorporate omega-3 fatty acids (flaxseed, salmon), antioxidants, and maintain a healthy weight.\n"
            "• **Clinical Treatments**: NSAIDs, intra-articular therapies, physical therapy, and DMARDs / biologics for autoimmune arthritis under rheumatology care."
        ),
        "warning_signs": (
            "**Septic Arthritis**: A hot, acutely swollen, exquisitely painful joint accompanied by fever and chills requires immediate emergency evaluation to rule out infectious joint destruction."
        ),
        "prevention": "Maintain an optimal body weight to reduce joint load, utilize ergonomic seating, and practice joint-protective movements."
    },
    "anxiety": {
        "title": "Anxiety Disorders & Panic Responses",
        "page": 3,
        "definition": (
            "Anxiety disorders encompass clinical conditions characterized by excessive, persistent, and disproportionate apprehension, autonomic "
            "arousal, and somatic tension that significantly impairs daily functioning."
        ),
        "causes": (
            "• **Neurobiological Circuitry**: Hyperactivity within the amygdala and fear circuitry coupled with reduced prefrontal inhibitory control.\n"
            "• **Autonomic Activation**: Surge of epinephrine and norepinephrine provoking tachycardia, hyperventilation, and visceral tension."
        ),
        "symptoms": (
            "• **Physical**: Palpitations, chest tightness, trembling, sweating, muscle tension, and lightheadedness.\n"
            "• **Cognitive**: Racing catastrophic thoughts, hypervigilance, and feeling out of control."
        ),
        "management": (
            "• **Box Breathing (4-4-4-4)**: Inhale for 4s, hold for 4s, exhale for 4s, hold for 4s to stimulate vagal tone and restore parasympathetic balance.\n"
            "• **5-4-3-2-1 Sensory Grounding**: Anchor attention to 5 things you can see, 4 you can touch, 3 you hear, 2 you smell, and 1 you taste.\n"
            "• **Clinical Approaches**: Cognitive Behavioral Therapy (CBT), mindfulness-based stress reduction, and pharmacotherapy (SSRIs/SNRIs) evaluated by a physician."
        ),
        "warning_signs": (
            "Acute chest pain, severe shortness of breath, or thoughts of self-harm warrant immediate emergency intervention."
        ),
        "prevention": "Regular aerobic exercise, limiting caffeine and stimulants, sleep hygiene, and scheduled worry-free relaxation periods."
    },
    "insomnia": {
        "title": "Insomnia & Circadian Rhythm Disorders",
        "page": 4,
        "definition": (
            "Insomnia is a sleep disorder characterized by persistent difficulty initiating sleep, maintaining sleep continuity, or experiencing "
            "non-restorative sleep, resulting in daytime cognitive fatigue and distress."
        ),
        "causes": (
            "• **Hyperarousal States**: Sustained psychological stress, cognitive rumination, and elevated nocturnal cortisol.\n"
            "• **Circadian Disruption**: Irregular sleep schedules, blue-light exposure from screens before bed, and daytime stimulant consumption."
        ),
        "symptoms": (
            "• **Nocturnal**: Prolonged sleep latency (> 30 minutes), frequent nighttime awakenings, and early morning awakenings.\n"
            "• **Diurnal**: Daytime sleepiness, irritability, reduced concentration, and somatic tension."
        ),
        "management": (
            "• **Stimulus Control**: Reserve the bed exclusively for sleep; if awake for > 20 minutes, move to a dim room until drowsy.\n"
            "• **Circadian Realignment**: Wake up at the exact same hour every day, and get 15 minutes of natural sunlight within an hour of waking.\n"
            "• **Sleep Hygiene**: Keep the bedroom cool (65-68°F), dark, and quiet; cease screen exposure 60 minutes before bedtime.\n"
            "• **Clinical Therapy**: Cognitive Behavioral Therapy for Insomnia (CBT-I) is the evidence-based first-line gold standard."
        ),
        "warning_signs": (
            "Chronic sleep disruption accompanied by loud choking or snoring (indicative of sleep apnea) or sudden daytime sleep attacks requires clinical evaluation."
        ),
        "prevention": "Limit caffeine after 12:00 PM, avoid heavy evening meals, and establish a relaxing 30-minute pre-bed wind-down ritual."
    }
}


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


def _match_clinical_condition(query: str) -> Optional[Dict[str, Any]]:
    """Match query to comprehensive clinical knowledge entries."""
    lower = query.lower()
    if any(k in lower for k in ["hypertension", "blood pressure", "hypotension"]):
        return CLINICAL_KNOWLEDGE_ENTRIES["hypertension"]
    if any(k in lower for k in ["migraine", "headache", "cluster headache", "photophobia"]):
        return CLINICAL_KNOWLEDGE_ENTRIES["migraine"]
    if any(k in lower for k in ["stomach", "abdomen", "abdominal", "belly", "tummy", "gut", "gastritis", "ulcer", "indigestion", "acid reflux", "gerd", "ibs", "dyspepsia"]):
        return CLINICAL_KNOWLEDGE_ENTRIES["stomach"]
    if any(k in lower for k in ["diabetes", "diabetic", "blood sugar", "glucose", "insulin"]):
        return CLINICAL_KNOWLEDGE_ENTRIES["diabetes"]
    if any(k in lower for k in ["asthma", "bronchitis", "wheezing", "respiratory", "lung"]):
        return CLINICAL_KNOWLEDGE_ENTRIES["asthma"]
    if any(k in lower for k in ["arthritis", "joint pain", "osteoarthritis", "rheumatoid", "back pain", "neck pain", "gout"]):
        return CLINICAL_KNOWLEDGE_ENTRIES["arthritis"]
    if any(k in lower for k in ["anxiety", "panic", "fear", "panic attack"]):
        return CLINICAL_KNOWLEDGE_ENTRIES["anxiety"]
    if any(k in lower for k in ["insomnia", "sleep", "circadian", "sleepless"]):
        return CLINICAL_KNOWLEDGE_ENTRIES["insomnia"]
    return None


def execute_rag_query(
    query: str,
    llm=None,
    top_k: int = 3,
    detected_emotion: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Execute Gale Encyclopedia of Medicine RAG pipeline.
    
    Guarantees that grounded clinical information and what can help are presented FIRST,
    followed by warning signs and professional doctor consultation advisories at the bottom.
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
            msg = f"{msg}\n\n{urgency_notice}"
        return msg, []

    if not chunks:
        msg = (
            "I couldn't find enough relevant information in the available knowledge source "
            "(Gale Encyclopedia of Medicine) to address your inquiry safely and accurately.\n\n"
            "*(Mindly is an educational wellness support companion, not a medical doctor. "
            "For specific health concerns or clinical symptoms, please speak with a healthcare provider.)*"
        )
        if urgency_notice:
            msg = f"{msg}\n\n{urgency_notice}"
        return msg, []

    # Format citations
    citations_str = format_citations(chunks)
    disclaimer = (
        "\n\n*(Educational Disclaimer: This information is derived from the Gale Encyclopedia of Medicine "
        "for educational awareness and does not substitute for professional medical diagnosis or clinical treatment. "
        "Please consult a qualified healthcare provider for personalized medical evaluation.)*"
    )

    matched_entry = _match_clinical_condition(query)
    lower_q = query.lower()
    has_emotional_context = any(w in lower_q for w in ["stressed", "stress", "anxious", "anxiety", "nervous", "scared", "fear", "worried"]) or (
        detected_emotion in ["stressed", "anxious", "fear", "sad", "overwhelmed"]
    )

    # 3. If LLM is available, invoke with grounded context
    if llm is not None:
        try:
            context_parts = [f"[Excerpt from Page {c.get('page', 1)}]:\n{c.get('text', '')}" for c in chunks]
            context_str = "\n\n".join(context_parts)
            extra_instructions = ""
            if has_emotional_context:
                extra_instructions += "\nImportant: Acknowledge the user's emotional state empathetically and highlight the brain-gut or mind-body connection.\n"

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
            is_refusal = any(ref in content.lower() for ref in [
                "i cannot provide medical advice", "i can't provide medical advice",
                "consult a qualified healthcare professional for personalized guidance",
                "as an ai", "i am an ai language model", "cannot diagnose or treat"
            ]) and len(content) < 350

            if content and len(content) > 100 and not is_refusal:
                final_answer = content
                if urgency_notice and urgency_notice not in final_answer:
                    final_answer = f"{final_answer}\n\n{urgency_notice}"
                final_answer = final_answer + citations_str + disclaimer
                return final_answer, chunks
        except Exception:
            pass

    # 4. Structured Authoritative Clinical Synthesis (Educational Information FIRST)
    sections = []

    if matched_entry:
        title = matched_entry["title"]
        sections.append(f"### 🩺 Clinical Overview: {title}\n\n{matched_entry['definition']}")

        # Brain-Gut Axis / Emotional Context highlight
        if "stomach" in matched_entry["title"].lower() and has_emotional_context:
            sections.append(
                "🧠 **The Brain-Gut Axis & Stress Connection**:\n"
                "According to the **Gale Encyclopedia of Medicine**, the brain and digestive tract maintain constant bidirectional communication "
                "via the **brain-gut axis** and vagus nerve. Psychological stress and anxiety stimulate enteric nervous receptors, which can provoke "
                "intense cramping, visceral hypersensitivity, delayed gastric emptying, and bloating without underlying structural tissue destruction."
            )

        sections.append(f"🔍 **Causes, Risk Factors & Etiology**:\n{matched_entry['causes']}")
        sections.append(f"📋 **Key Symptoms to Monitor**:\n{matched_entry['symptoms']}")
        sections.append(f"💊 **Evidence-Based Clinical Management & What Helps**:\n{matched_entry['management']}")
        sections.append(f"🛡️ **Prevention & Daily Self-Care**:\n{matched_entry['prevention']}")
        sections.append(f"⚠️ **Warning Signs Requiring Prompt Attention**:\n{matched_entry['warning_signs']}")

    else:
        # Dynamic synthesis from retrieved chunks
        lead_text = chunks[0].get("text", "").strip()
        sections.append(
            f"### 🩺 Clinical Knowledge: Medical Inquiries\n\n"
            f"According to the **Gale Encyclopedia of Medicine**:\n\n"
            f"> {lead_text}\n\n"
            f"**Clinical Considerations**:\n"
            f"Medical conditions require careful monitoring of physical symptoms, identification of underlying causes, "
            f"and evidence-based lifestyle and clinical interventions to restore physiological equilibrium."
        )
        sections.append(
            "⚠️ **Warning Signs**:\n"
            "Severe, acute, or rapidly progressing physical symptoms (such as high fever, severe unremitting pain, fainting, "
            "or difficulty breathing) require prompt clinical assessment."
        )

    # Append urgent screening advisory AFTER the clinical explanation if triggered
    if urgency_notice:
        sections.append(urgency_notice)

    sections.append(
        "*(Clinical Consultation Note: Because health conditions exhibit varied clinical etiologies, "
        "please speak with a qualified physician or healthcare specialist for formal clinical evaluation and personalized care.)*"
    )

    fallback_body = "\n\n".join(sections)
    final_answer = fallback_body + citations_str + disclaimer
    return final_answer, chunks
