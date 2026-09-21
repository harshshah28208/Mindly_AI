"""Script to generate data/knowledge/gale_encyclopedia.pdf with 5 comprehensive medical entries:
1. Hypertension (High Blood Pressure)
2. Migraine Headache Disorders
3. Anxiety Disorders and Panic Responses
4. Insomnia and Circadian Rhythm Disorders
5. Abdominal and Stomach Pain (Gastrointestinal Disorders)
"""

from pathlib import Path


def generate_pdf():
    out_path = Path("data/knowledge/gale_encyclopedia.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pages_data = [
        # PAGE 1: Hypertension
        [
            "GALE ENCYCLOPEDIA OF MEDICINE",
            "HYPERTENSION (HIGH BLOOD PRESSURE)",
            "",
            "Definition: Hypertension is a chronic medical condition characterized by persistent elevation of arterial blood pressure above 130/80 mmHg. It is widely recognized as a major risk factor for coronary artery disease, stroke, and chronic kidney disease.",
            "",
            "Causes and Pathophysiology: Primary or essential hypertension accounts for approximately 90 to 95 percent of cases. It develops gradually over years due to a complex interplay of genetic factors, sympathetic nervous system hyperactivity, increased dietary sodium consumption, arterial wall stiffening, obesity, and prolonged psychological stress.",
            "",
            "Symptoms and Clinical Manifestations: Hypertension is famously referred to as the silent killer because most individuals experience no overt symptoms until target-organ damage occurs. However, some patients experience morning occipital headaches, dizziness, palpitations, visual blurriness, or episodic fatigue.",
            "",
            "Management and Lifestyle Approaches: Non-pharmacological management includes sodium restriction, adherence to the DASH (Dietary Approaches to Stop Hypertension) diet rich in fruits and vegetables, regular aerobic physical activity, moderation of alcohol, and stress reduction. Antihypertensive medications may include ACE inhibitors, calcium channel blockers, and diuretics as prescribed by a medical physician.",
        ],
        # PAGE 2: Migraine Headache
        [
            "GALE ENCYCLOPEDIA OF MEDICINE",
            "MIGRAINE HEADACHE DISORDERS",
            "",
            "Definition: Migraine is a complex neurological disorder characterized by recurrent attacks of moderate to severe unilateral throbbing headache pain, typically lasting from 4 to 72 hours if untreated.",
            "",
            "Symptoms and Clinical Presentation: Common symptoms include severe pulsating pain usually focused on one side of the head, nausea, vomiting, photophobia (extreme sensitivity to light), phonophobia (sensitivity to sound), and occasional cutaneous allodynia. In approximately 20 to 30 percent of patients, attacks are preceded by transient focal neurological symptoms known as a visual aura (scintillating scotomas or zigzag lines).",
            "",
            "Etiology and Triggers: Cortical spreading depression is considered the underlying neurophysiological substrate of migraine aura, triggering trigeminovascular activation and calcitonin gene-related peptide (CGRP) release. Common triggers include emotional stress, hormonal fluctuations, irregular sleep patterns, skipped meals, sensory overstimulation, and specific dietary items.",
            "",
            "Evidence-Based Management: Acute pharmacotherapy includes triptans, NSAIDs, and antiemetics. Preventive interventions include beta-blockers, CGRP antagonists, lifestyle stabilization, and progressive muscular relaxation.",
        ],
        # PAGE 3: Anxiety and Panic
        [
            "GALE ENCYCLOPEDIA OF MEDICINE",
            "ANXIETY DISORDERS AND PANIC RESPONSES",
            "",
            "Definition: Anxiety disorders encompass psychiatric conditions characterized by excessive, persistent, and uncontrollable apprehension, nervous arousal, and worry regarding everyday events, out of proportion to actual threats.",
            "",
            "Physiological and Somatic Manifestations: Autonomic nervous system hyperarousal produces rapid tachycardia, tachypnea, peripheral vasoconstriction, diaphoresis, gastrointestinal motility changes, muscle tension, tremors, and dizziness. In acute panic episodes, hyperventilation can induce hypocapnia leading to paresthesias and depersonalization.",
            "",
            "Evidence-Based Coping and Stabilization: Paced breathing techniques (such as 4-second box breathing) actively stimulate the parasympathetic vagal nerve, moderating heart rate variability and downregulating amygdaloid activation. Sensory 5-4-3-2-1 grounding exercises reorient cognitive attention to external stimuli, interrupting acute dissociative and panic cycles.",
            "",
            "Clinical Care and Assessment: Persistent generalized anxiety disorder (GAD) or panic disorder responds effectively to cognitive behavioral therapy (CBT) and SSRI pharmacotherapy under professional supervision.",
        ],
        # PAGE 4: Insomnia
        [
            "GALE ENCYCLOPEDIA OF MEDICINE",
            "INSOMNIA AND CIRCADIAN RHYTHM DISORDERS",
            "",
            "Definition: Insomnia involves persistent difficulty initiating sleep, maintaining sleep continuity, or experiencing non-restorative sleep, accompanied by significant daytime impairment, fatigue, and mood disruption.",
            "",
            "Etiological Factors and Mechanisms: Hyperarousal of the central nervous system, elevated evening cortisol levels, autonomic imbalance, and dysfunctional cognitive beliefs about sleep perpetuate chronic insomnia. In college and young adult populations, circadian rhythm phase delays and blue light exposure from digital displays suppress physiological nocturnal melatonin secretion.",
            "",
            "Sleep Hygiene and Cognitive Interventions: Recommended behavioral practices include maintaining strict regular wake times seven days a week, establishing a dark, cool, and quiet sleep environment, avoiding caffeine within 8 hours of bedtime, and stimulus control therapy (leaving the bed if awake after 20 minutes to prevent conditioning the bed to wakefulness).",
            "",
            "Medical Evaluation: Chronic sleep disruption exceeding three months should be evaluated by a healthcare specialist to rule out obstructive sleep apnea, restless legs syndrome, or clinical depression.",
        ],
        # PAGE 5: Abdominal and Stomach Pain
        [
            "GALE ENCYCLOPEDIA OF MEDICINE",
            "ABDOMINAL PAIN AND GASTROINTESTINAL DISORDERS",
            "",
            "Definition: Abdominal pain, commonly experienced as stomach ache, localized cramping, or visceral discomfort, is pain felt anywhere between the inferior costal margin and the groin. It is one of the most frequent clinical presentations in both ambulatory and emergency medicine.",
            "",
            "Common Causes and the Brain-Gut Axis: Common benign and functional etiologies include dyspepsia (indigestion), acute gastroenteritis, gastroesophageal reflux (GERD), gas accumulation, irritable bowel syndrome (IBS), gastritis, and dietary sensitivities. The bidirectional brain-gut axis connects the central nervous system and the enteric nervous system; consequently, acute psychological stress, exam anxiety, and autonomic arousal frequently cause real physiological abdominal cramping, nausea, and altered motility without structural organ damage.",
            "",
            "Clinical Warning Signs (Red Flags): While many stomach aches are self-limiting, certain symptoms indicate severe underlying pathology requiring urgent medical evaluation. Red flag symptoms include: sudden, severe, or rapidly escalating pain; high fever; persistent vomiting with inability to tolerate fluids; vomiting blood (hematemesis) or coffee-ground emesis; black, tarry stools or bright red rectal bleeding; severe abdominal rigidity (board-like hardness) or marked rebound tenderness; unexplained weight loss; or pain accompanied by dizziness, fainting, or yellowing of the skin (jaundice).",
            "",
            "Educational Guidance and Assessment: Mild functional stomach discomfort often improves with rest, adequate hydration with clear fluids, small bland meals (such as bananas, rice, applesauce, toast), and stress reduction. Heat packs applied to the abdomen may relieve benign muscular cramping. Self-diagnosis and self-medication with NSAIDs should be avoided, as NSAIDs can irritate gastric mucosa. Individuals experiencing persistent, recurrent, or severe abdominal pain should seek formal in-person clinical assessment from a licensed healthcare provider.",
        ],
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
        stream_lines = ["BT", "/F1 11 Tf", "50 740 Td", "15 TL"]
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
