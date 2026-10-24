"""Script to generate data/knowledge/gale_encyclopedia.pdf with 12 comprehensive medical entries:
1. Hypertension (High Blood Pressure) & Cardiovascular Health
2. Migraine Headache & Neurological Disorders
3. Anxiety Disorders, Panic Responses & Autonomic Hyperarousal
4. Insomnia & Circadian Rhythm Sleep Disorders
5. Abdominal Pain, Gastric Acidity, GERD & Gastrointestinal Disorders
6. Musculoskeletal Disorders, Knee Pain, Joint Strain & Osteoarthritis
7. Diabetes Mellitus, Glycemic Control & Metabolic Regulation
8. Respiratory Disorders, Asthma & Bronchitis
9. Infectious Illnesses, Fever & Viral Upper Respiratory Infections
10. Dermatological Conditions, Eczema & Inflammatory Skin Rashes
11. Spinal Disorders, Lumbar Back Pain & Postural Rehabilitation
12. Preventative Health, Clinical Nutrition & Immune Support
"""

from pathlib import Path
import textwrap


def format_page_lines(title: str, subtitle: str, sections: list) -> list:
    """Format structured sections into wrapped PDF line entries."""
    lines = ["GALE ENCYCLOPEDIA OF MEDICINE", title.upper(), ""]
    if subtitle:
        lines.extend([subtitle, ""])
    
    for sec_header, sec_text in sections:
        header_line = f"{sec_header}: " if sec_header else ""
        full_text = f"{header_line}{sec_text}".strip()
        wrapped = textwrap.wrap(full_text, width=82)
        lines.extend(wrapped)
        lines.append("")
    return lines


def generate_pdf():
    out_path = Path("data/knowledge/gale_encyclopedia.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pages_data = [
        # PAGE 1: Hypertension
        format_page_lines(
            "Hypertension (High Blood Pressure)",
            "Cardiovascular & Circulatory Disorders",
            [
                ("Definition", "Hypertension is a chronic medical condition characterized by persistent elevation of arterial blood pressure above 130/80 mmHg. It is widely recognized as a major risk factor for coronary artery disease, stroke, and chronic kidney disease."),
                ("Causes and Pathophysiology", "Primary or essential hypertension accounts for approximately 90 to 95 percent of cases. It develops gradually over years due to a complex interplay of genetic factors, sympathetic nervous system hyperactivity, increased dietary sodium consumption, arterial wall stiffening, obesity, and prolonged psychological stress."),
                ("Symptoms and Clinical Manifestations", "Hypertension is famously referred to as the silent killer because most individuals experience no overt symptoms until target-organ damage occurs. However, some patients experience morning occipital headaches, dizziness, palpitations, visual blurriness, or episodic fatigue."),
                ("Management and Lifestyle Approaches", "Non-pharmacological management includes sodium restriction, adherence to the DASH (Dietary Approaches to Stop Hypertension) diet rich in fruits and vegetables, regular aerobic physical activity, moderation of alcohol, and stress reduction. Antihypertensive medications may include ACE inhibitors, calcium channel blockers, and diuretics as prescribed by a medical physician."),
            ]
        ),
        # PAGE 2: Migraine Headache
        format_page_lines(
            "Migraine Headache Disorders",
            "Neurovascular & Cephalic Pain Syndromes",
            [
                ("Definition", "Migraine is a complex neurological disorder characterized by recurrent attacks of moderate to severe unilateral throbbing headache pain, typically lasting from 4 to 72 hours if untreated."),
                ("Symptoms and Clinical Presentation", "Common symptoms include severe pulsating pain usually focused on one side of the head, nausea, vomiting, photophobia (extreme sensitivity to light), phonophobia (sensitivity to sound), and occasional cutaneous allodynia. In approximately 20 to 30 percent of patients, attacks are preceded by transient focal neurological symptoms known as a visual aura (scintillating scotomas or zigzag lines)."),
                ("Etiology and Triggers", "Cortical spreading depression is considered the underlying neurophysiological substrate of migraine aura, triggering trigeminovascular activation and calcitonin gene-related peptide (CGRP) release. Common triggers include emotional stress, hormonal fluctuations, irregular sleep patterns, skipped meals, sensory overstimulation, and specific dietary items."),
                ("Evidence-Based Management", "Acute pharmacotherapy includes triptans, NSAIDs, and antiemetics. Preventive interventions include beta-blockers, CGRP antagonists, lifestyle stabilization, and progressive muscular relaxation."),
            ]
        ),
        # PAGE 3: Anxiety and Panic
        format_page_lines(
            "Anxiety Disorders and Panic Responses",
            "Psychiatric & Autonomic Hyperarousal Syndromes",
            [
                ("Definition", "Anxiety disorders encompass psychiatric conditions characterized by excessive, persistent, and uncontrollable apprehension, nervous arousal, and worry regarding everyday events, out of proportion to actual threats."),
                ("Physiological and Somatic Manifestations", "Autonomic nervous system hyperarousal produces rapid tachycardia, tachypnea, peripheral vasoconstriction, diaphoresis, gastrointestinal motility changes, muscle tension, tremors, and dizziness. In acute panic episodes, hyperventilation can induce hypocapnia leading to paresthesias and depersonalization."),
                ("Evidence-Based Coping and Stabilization", "Paced breathing techniques (such as 4-second box breathing) actively stimulate the parasympathetic vagal nerve, moderating heart rate variability and downregulating amygdaloid activation. Sensory 5-4-3-2-1 grounding exercises reorient cognitive attention to external stimuli, interrupting acute dissociative and panic cycles."),
                ("Clinical Care and Assessment", "Persistent generalized anxiety disorder (GAD) or panic disorder responds effectively to cognitive behavioral therapy (CBT) and SSRI pharmacotherapy under professional supervision."),
            ]
        ),
        # PAGE 4: Insomnia
        format_page_lines(
            "Insomnia and Circadian Rhythm Disorders",
            "Somnological & Sleep Architecture Dysregulation",
            [
                ("Definition", "Insomnia involves persistent difficulty initiating sleep, maintaining sleep continuity, or experiencing non-restorative sleep, accompanied by significant daytime impairment, fatigue, and mood disruption."),
                ("Etiological Factors and Mechanisms", "Hyperarousal of the central nervous system, elevated evening cortisol levels, autonomic imbalance, and dysfunctional cognitive beliefs about sleep perpetuate chronic insomnia. In college and young adult populations, circadian rhythm phase delays and blue light exposure from digital displays suppress physiological nocturnal melatonin secretion."),
                ("Sleep Hygiene and Cognitive Interventions", "Recommended behavioral practices include maintaining strict regular wake times seven days a week, establishing a dark, cool, and quiet sleep environment, avoiding caffeine within 8 hours of bedtime, and stimulus control therapy (leaving the bed if awake after 20 minutes to prevent conditioning the bed to wakefulness)."),
                ("Medical Evaluation", "Chronic sleep disruption exceeding three months should be evaluated by a healthcare specialist to rule out obstructive sleep apnea, restless legs syndrome, or clinical depression."),
            ]
        ),
        # PAGE 5: Abdominal and Stomach Pain
        format_page_lines(
            "Abdominal Pain and Gastrointestinal Disorders",
            "Gastroenterology, Acidity, GERD & Enteric Nervous System",
            [
                ("Definition", "Abdominal pain, commonly experienced as stomach ache, localized cramping, or visceral discomfort, is pain felt anywhere between the inferior costal margin and the groin. It is one of the most frequent clinical presentations in both ambulatory and emergency medicine."),
                ("Common Causes and the Brain-Gut Axis", "Common benign and functional etiologies include dyspepsia (indigestion), acute gastroenteritis, gastroesophageal reflux (GERD), gas accumulation, irritable bowel syndrome (IBS), gastritis, and dietary sensitivities. The bidirectional brain-gut axis connects the central nervous system and the enteric nervous system; consequently, acute psychological stress, exam anxiety, and autonomic arousal frequently cause real physiological abdominal cramping, nausea, and altered motility without structural organ damage."),
                ("Clinical Warning Signs (Red Flags)", "While many stomach aches are self-limiting, certain symptoms indicate severe underlying pathology requiring urgent medical evaluation. Red flag symptoms include: sudden, severe, or rapidly escalating pain; high fever; persistent vomiting with inability to tolerate fluids; vomiting blood (hematemesis) or coffee-ground emesis; black, tarry stools or bright red rectal bleeding; severe abdominal rigidity (board-like hardness) or marked rebound tenderness; unexplained weight loss; or pain accompanied by dizziness, fainting, or yellowing of the skin (jaundice)."),
                ("Educational Guidance and Assessment", "Mild functional stomach discomfort often improves with rest, adequate hydration with clear fluids, small bland meals (such as bananas, rice, applesauce, toast), and stress reduction. Heat packs applied to the abdomen may relieve benign muscular cramping. Self-diagnosis and self-medication with NSAIDs should be avoided, as NSAIDs can irritate gastric mucosa. Individuals experiencing persistent, recurrent, or severe abdominal pain should seek formal in-person clinical assessment from a licensed healthcare provider."),
            ]
        ),
        # PAGE 6: Musculoskeletal Disorders, Knee Pain & Joint Conditions
        format_page_lines(
            "Musculoskeletal Disorders, Knee Pain and Joint Strain",
            "Orthopedic Medicine, Patellofemoral Mechanics & Arthritis",
            [
                ("Definition", "Knee pain is a frequent musculoskeletal complaint arising from mechanical stress, traumatic injury, inflammatory processes, or degenerate cartilage in the knee joint. The knee is a complex weight-bearing hinge joint composed of the femur, tibia, patella, stabilizing ligaments (ACL, PCL, MCL, LCL), fibrocartilage shock absorbers (menisci), and protective articular cartilage."),
                ("Underlying Etiology and Biomechanical Reasons", "Knee discomfort develops from multiple clinical factors: 1) Patellofemoral Pain Syndrome (runner's knee), caused by abnormal tracking of the kneecap over the femoral groove due to quadriceps and vastus medialis imbalances; 2) Meniscus Tears or Ligamentous Sprains resulting from sudden twisting, impact, or repetitive deceleration; 3) Osteoarthritis, involving progressive wear and loss of articular cartilage cushioning, triggering joint friction and bone-on-bone stiffness; 4) Patellar Tendinitis or Bursitis (inflammation of fluid-filled sacs surrounding the joint); 5) Excess Mechanical Load, where every pound of excess body weight multiplies compressive joint force by four to six times during weight-bearing movement."),
                ("Immediate Relief Action Plan (R.I.C.E. Protocol)", "For acute knee or joint pain, immediate conservative management follows the evidence-based R.I.C.E. clinical protocol: 1) Rest: Cease strenuous weight-bearing activities immediately to prevent tissue micro-trauma; 2) Ice: Apply cold packs wrapped in a thin towel for 15 to 20 minutes every 2 to 3 hours to constrict local blood vessels and reduce inflammatory effusion; 3) Compression: Apply an elastic tubular bandage or knee sleeve with gentle, even compression to limit joint swelling (avoid wrapping so tightly as to compromise arterial or venous circulation); 4) Elevation: Prop the leg up on pillows above the level of the heart while resting to promote passive lymphatic and venous fluid drainage."),
                ("Long-Term Cure and Rehabilitation Plan", "Lasting recovery and joint preservation require structured non-pharmacological care: 1) Targeted Physical Therapy focusing on quadriceps strengthening (especially straight leg raises and isometric quad sets), hamstring flexibility, and hip abductor stabilization; 2) Low-Impact Cardiovascular Conditioning such as swimming or stationary cycling with the saddle adjusted to avoid hyperflexion; 3) Weight Optimization to diminish cumulative compressive cartilage loading; 4) Proper Supportive Footwear and ergonomic orthotics if overpronation is present; 5) Clinical Evaluation by an orthopedic physician or physical therapist for guided therapeutic exercises or knee aspiration if effusion persists."),
                ("Warning Signs and Urgent Orthopedic Red Flags", "Seek prompt medical or emergency orthopedic attention if any of the following occur: complete inability to bear weight or take four consecutive steps; visible joint deformity or abnormal knee angle; a loud 'pop' followed by rapid, tense joint swelling within 2 hours (suggesting hemarthrosis or cruciate rupture); sensation of the knee catching, locking, or giving way entirely; severe redness, hot skin, high fever, and extreme joint tenderness (raising suspicion of septic bacterial arthritis); or sudden numbness, tingling, or cold pallor in the lower leg and foot."),
            ]
        ),
        # PAGE 7: Diabetes Mellitus & Metabolic Regulation
        format_page_lines(
            "Diabetes Mellitus and Glycemic Regulation",
            "Endocrine & Metabolic Health Disorders",
            [
                ("Definition", "Diabetes mellitus is a chronic metabolic disorder characterized by persistent hyperglycemia resulting from defects in insulin secretion, insulin action, or both. It is broadly classified into Type 1 (autoimmune destruction of pancreatic beta cells) and Type 2 (progressive insulin resistance accompanied by secretory defect)."),
                ("Causes and Pathophysiological Mechanisms", "In Type 2 diabetes, peripheral tissues (muscle, adipose, liver) exhibit blunted sensitivity to circulating insulin, requiring compensatory hyperinsulinemia until beta cell exhaustion occurs. Contributing risk factors include genetic predisposition, abdominal adiposity, sedentary lifestyle, chronic systemic inflammation, and prolonged intake of refined carbohydrates."),
                ("Symptoms and Clinical Indicators", "Classic symptoms include polyuria (excessive urination), polydipsia (excessive thirst), polyphagia (constant hunger), unintended weight loss, chronic fatigue, blurred vision, and delayed wound healing."),
                ("Evidence-Based Management and Prevention", "Primary management combines medical nutrition therapy (low glycemic index carbohydrates, dietary fiber, lean proteins), at least 150 minutes of weekly moderate aerobic and resistance training, continuous glucose monitoring, and physician-prescribed oral hypoglycemics (such as metformin) or insulin regimens."),
            ]
        ),
        # PAGE 8: Respiratory Disorders, Asthma & Bronchitis
        format_page_lines(
            "Respiratory Disorders, Asthma and Bronchitis",
            "Pulmonary Medicine, Airway Hyperreactivity & Bronchospasm",
            [
                ("Definition", "Asthma and bronchitis are common inflammatory airway conditions causing recurrent episodes of wheezing, breathlessness, chest tightness, and coughing, linked to reversible bronchial hyperreactivity and mucus hypersecretion."),
                ("Etiology and Environmental Triggers", "Bronchial inflammation triggers smooth muscle constriction around the bronchioles, mucosal edema, and tenacious mucus plug formation. Key triggers include viral respiratory pathogens, airborne aeroallergens (dust mites, pollen, animal dander), cold dry air, physical exertion, airborne irritants (tobacco smoke, chemical fumes), and emotional hyperventilation."),
                ("Immediate Action and Relief Protocol", "At the first sign of respiratory tightness: 1) Sit upright immediately; never recline; 2) Eliminate exposure to active triggers; 3) Utilize prescribed fast-acting bronchodilator rescue inhalers (such as albuterol) with a spacer as directed by a physician; 4) Practice pursed-lip breathing to prevent dynamic airway collapse during expiration."),
                ("Warning Signs Requiring Emergency Medical Care", "Call emergency medical services immediately if breathing difficulty causes inability to complete full sentences without pausing, severe retractions (chest wall sinking in), blue or gray discoloration around the lips or fingernails (cyanosis), or when rescue bronchodilators fail to provide prompt relief."),
            ]
        ),
        # PAGE 9: Infectious Illnesses, Fever & Viral Infections
        format_page_lines(
            "Infectious Illnesses, Fever and Viral Infections",
            "Infectious Disease, Thermoregulation & Upper Respiratory Illness",
            [
                ("Definition", "Fever (pyrexia) is a regulated elevation of the internal body core temperature above the normal physiological baseline (typically > 38.0 C or 100.4 F), orchestrated by the preoptic anterior hypothalamus in response to infectious or inflammatory pyrogens."),
                ("Mechanisms and Immune Defense Role", "Exogenous microbial toxins stimulate host leukocytes and macrophages to release endogenous pyrogenic cytokines (interleukin-1, interleukin-6, TNF-alpha). These induce hypothalamic prostaglandin E2 (PGE2) synthesis, shifting the thermal set-point upward to enhance leukocyte phagocytosis, inhibit viral replication, and mobilize antimicrobial defenses."),
                ("Evidence-Based Relief and Hydration Care", "Supportive care focuses on comfort and systemic hydration: 1) Drink plentiful fluids (oral rehydration salts, broths, dilute juices) to offset insensible fluid losses from sweating and tachypnea; 2) Rest in a well-ventilated, temperate environment with light clothing; 3) Apply cool or lukewarm damp compresses to the forehead and neck; 4) Over-the-counter antipyretics (acetaminophen or ibuprofen) may be used strictly according to package instructions to relieve discomfort."),
                ("Warning Signs Requiring Urgent Medical Evaluation", "Seek emergency medical evaluation if fever exceeds 39.5 C (103.1 F), persists beyond 72 hours, or is accompanied by stiff neck and photophobia (concerning for meningitis), new petechial rash, persistent vomiting, extreme lethargy, confusion, or difficulty breathing."),
            ]
        ),
        # PAGE 10: Dermatological Conditions, Eczema & Skin Rashes
        format_page_lines(
            "Dermatological Disorders, Eczema and Allergic Rashes",
            "Dermatology, Epidermal Barrier Dysfunction & Pruritus",
            [
                ("Definition", "Eczema (atopic dermatitis) and allergic contact dermatitis are chronic inflammatory skin disorders characterized by pruritus (intense itching), erythema, xerosis (dryness), and excoriated eczematous lesions."),
                ("Pathophysiological Etiology", "Genetic defects in epidermal barrier proteins (such as filaggrin) increase transepidermal water loss and facilitate environmental antigen penetration, triggering hyperreactive cutaneous immune responses and local mast cell degranulation."),
                ("Immediate Relief and Soothing Care", "1) Apply thick, fragrance-free ceramide-rich emollient ointments immediately after lukewarm bathing (the 'soak and seal' method); 2) Apply cold, damp cloth compresses to pruritic patches for 10-15 minutes to soothe nerve endings; 3) Keep fingernails trimmed short to minimize mechanical skin excoriation and secondary bacterial infection."),
                ("Warning Signs Requiring Clinical Attention", "Consult a physician if rash spreads rapidly across large body surface areas, develops yellow honey-colored crusting or oozing pus (impetigo), exhibits severe pain or swelling, or if accompanied by high systemic fever or facial swelling."),
            ]
        ),
        # PAGE 11: Spinal Disorders & Lumbar Back Pain
        format_page_lines(
            "Spinal Disorders, Lumbar Back Pain and Ergonomics",
            "Spinal Biomechanics, Paraspinal Spasm & Ergonomic Care",
            [
                ("Definition", "Low back pain (lumbago) refers to discomfort localized between the lower costal margin and the inferior gluteal folds, representing one of the leading global causes of activity limitation and occupational disability."),
                ("Common Causes and Biomechanical Stress", "Over 85 percent of back pain episodes are non-specific musculoskeletal strains involving paraspinal muscles, ligaments, and facet joints. Contributing factors include prolonged static sitting, poor workstation ergonomics, heavy unergonomic lifting, sudden spinal rotational torque, core abdominal muscular weakness, and intervertebral disc degeneration."),
                ("Immediate Self-Care and Relief Measures", "1) Relative Rest: Avoid strict prolonged bed rest (which stiffens spinal joints and delays functional recovery); maintain gentle daily ambulation; 2) Postural Positioning: Rest in a comfortable supine position with knees bent and supported by pillows, or in a reclined semi-Fowler's posture; 3) Thermal Therapy: Apply ice packs for the first 48 hours for acute spasm, followed by soothing moist heat packs to relax hypertonic muscular bands."),
                ("Urgent Neurological Warning Signs (Red Flags)", "Seek immediate emergency neurosurgical evaluation if back pain is accompanied by loss of bowel or bladder control (cauda equina syndrome), progressive sensory numbness in the groin/perineal region (saddle anesthesia), or progressive motor weakness in the legs (such as foot drop)."),
            ]
        ),
        # PAGE 12: Preventative Health & Immune Support
        format_page_lines(
            "Preventative Health, Clinical Nutrition and Immune Health",
            "Health Promotion, Preventative Medicine & Lifestyle Science",
            [
                ("Definition", "Preventative medicine and lifestyle health encompass evidence-based strategies designed to protect physiological homeostasis, bolster innate and adaptive immune resilience, and mitigate chronic disease risks before clinical onset occurs."),
                ("Core Pillars of Cellular and Systemic Health", "1) Balanced Mediterranean-Style Nutrition emphasizing diverse polyphenol-rich plant foods, lean proteins, and omega-3 fatty acids; 2) Regular Aerobic and Resistance Exercise (150 minutes of moderate activity weekly) to enhance cardiovascular fitness, metabolic insulin sensitivity, and immune cell recirculation; 3) Restorative Sleep (7 to 9 hours nightly) essential for endocrine balance and tissue repair; 4) Stress Modulation via mindfulness and social connection to suppress chronic pro-inflammatory cortisol elevation."),
                ("Clinical Preventative Screening Guidelines", "Routine periodic health checkups with a licensed primary care physician should include blood pressure assessment, fasting lipid profiles, hemoglobin A1c monitoring, body mass index tracking, and age-appropriate cancer screenings."),
            ]
        ),
    ]

    # Build pure PDF
    objects = []
    
    # 1: Catalog
    objects.append("1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj")
    
    # 2: Pages
    kid_refs = " ".join(f"{3 + i*2} 0 R" for i in range(len(pages_data)))
    objects.append(f"2 0 obj\n<</Type /Pages /Kids [{kid_refs}] /Count {len(pages_data)}>>\nendobj")

    font_obj_idx = 3 + len(pages_data) * 2

    # Pages and Contents
    for i, page_lines in enumerate(pages_data):
        page_idx = 3 + i * 2
        content_idx = 4 + i * 2
        
        # Build content stream
        stream_lines = ["BT", "/F1 10 Tf", "45 745 Td", "13.5 TL"]
        for line in page_lines:
            if not line:
                stream_lines.append("() '")
            else:
                escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                stream_lines.append(f"({escaped}) '")
        stream_lines.append("ET")
        stream_text = "\n".join(stream_lines)
        stream_bytes = stream_text.encode("latin1")
        
        # Page object
        objects.append(
            f"{page_idx} 0 obj\n"
            f"<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources <</Font <</F1 {font_obj_idx} 0 R>>>> /Contents {content_idx} 0 R>>\n"
            f"endobj"
        )
        # Content stream object
        objects.append(
            f"{content_idx} 0 obj\n"
            f"<</Length {len(stream_bytes)}>>\n"
            f"stream\n{stream_text}\nendstream\nendobj"
        )

    # Font object
    objects.append(
        f"{font_obj_idx} 0 obj\n"
        f"<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>\n"
        f"endobj"
    )

    # Assemble PDF with cross reference table
    pdf_parts = ["%PDF-1.4\n"]
    offsets = [0]
    
    current_pos = len(pdf_parts[0].encode("latin1"))
    for obj in objects:
        offsets.append(current_pos)
        part = obj + "\n"
        pdf_parts.append(part)
        current_pos += len(part.encode("latin1"))

    xref_pos = current_pos
    xref = [f"xref\n0 {len(offsets)}\n0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(f"{off:010d} 00000 n \n")
    
    trailer = (
        f"trailer\n<</Size {len(offsets)} /Root 1 0 R>>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    )

    pdf_content = "".join(pdf_parts) + "".join(xref) + trailer
    out_path.write_bytes(pdf_content.encode("latin1"))
    print(f"Successfully wrote {len(pages_data)}-page PDF to {out_path}")


if __name__ == "__main__":
    generate_pdf()

