"""Comprehensive generator for data/knowledge/gale_encyclopedia.pdf.

Builds a complete, authoritative 30-chapter clinical medical encyclopedia:
1. Hypertension (High Blood Pressure) & Cardiovascular Health
2. Migraine Headache Disorders & Neurovascular Cephalalgia
3. Anxiety Disorders, Panic Responses & Autonomic Hyperarousal
4. Insomnia & Circadian Rhythm Sleep Architecture
5. Abdominal Pain, Acidity, GERD & Enteric Brain-Gut Axis
6. Musculoskeletal Disorders, Knee Pain, Patellofemoral Mechanics & Osteoarthritis
7. Diabetes Mellitus, Glycemic Control & Metabolic Regulation
8. Respiratory Disorders, Asthma & Acute Bronchitis
9. Infectious Illnesses, Viral Fever & Upper Respiratory Infections
10. Dermatological Conditions, Eczema & Inflammatory Skin Rashes
11. Spinal Disorders, Lumbar Back Pain & Sciatica
12. Preventative Health, Dehydration, Clinical Nutrition & Immune Support
13. Peptic Ulcer Disease, Gastric Erosions & Helicobacter Pylori
14. Irritable Bowel Syndrome (IBS) & Visceral Hypersensitivity
15. Major Depressive Disorder, Neurochemistry & Affective Regulation
16. Thyroid Disorders: Hypothyroidism, Hyperthyroidism & Metabolic Balance
17. Allergic Rhinitis, Environmental Allergies & Histaminic Pathways
18. Urinary Tract Infections (UTI), Cystitis & Renal Health
19. Cervical Spondylosis, Neck Pain & Postural Ergonomics
20. Shoulder Impingement, Rotator Cuff Tendinopathy & Frozen Shoulder
21. Ankle Sprains, Ligamentous Strain & R.I.C.E. Rehabilitation
22. Tension-Type Headache & Craniofacial Myofascial Trigger Points
23. Vertigo, Benign Paroxysmal Positional Vertigo (BPPV) & Vestibular Health
24. Iron Deficiency Anemia & Hematological Oxygenation
25. Vitamin D Deficiency, Bone Mineralization & Musculoskeletal Weakness
26. Dengue, Malaria & Tropical Vector-Borne Febrile Syndromes
27. Typhoid Fever & Salmonella Enteric Systemic Illness
28. Polycystic Ovary Syndrome (PCOS) & Endocrine Metabolism
29. Acute Gastroenteritis, Food Poisoning & Fluid-Electrolyte Loss
30. Chronic Fatigue, Burnout & Neuroendocrine Stress Adaptation
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
            "Abdominal Pain, Acidity and Gastrointestinal Disorders",
            "Gastroenterology, Acidity, GERD & Enteric Nervous System",
            [
                ("Definition", "Abdominal pain and gastric hyperacidity (acid reflux, GERD) involve chemical mucosal irritation, retrosternal heartburn, epigastric burning, and visceral smooth muscle cramping between the lower ribcage and pelvis."),
                ("Common Causes and the Brain-Gut Axis", "Lower esophageal sphincter (LES) laxity allows gastric acid regurgitation. The bidirectional brain-gut axis connects the central and enteric nervous systems; psychological stress, exam anxiety, and autonomic arousal stimulate enteric receptors, provoking hypersecretion of stomach acid, delayed gastric emptying, and visceral hypersensitivity."),
                ("Immediate Relief and Management", "Maintain an upright posture (do not lie down for 3 hours after eating), sip lukewarm water, loosen tight clothing, and elevate the head of the bed 6 inches. Avoid fried foods, citrus, sodas, caffeine, and NSAID pain relievers. Over-the-counter antacids, H2 blockers, and proton pump inhibitors (PPIs) suppress stomach acidity under clinical guidance."),
                ("Clinical Warning Signs (Red Flags)", "Seek immediate emergency medical attention for severe unremitting pain, vomiting blood or coffee-ground material, black tarry stools, board-like abdominal rigidity, high fever, or difficulty swallowing."),
            ]
        ),
        # PAGE 6: Musculoskeletal Disorders, Knee Pain & Joint Conditions
        format_page_lines(
            "Musculoskeletal Disorders, Knee Pain and Joint Strain",
            "Orthopedic Medicine, Patellofemoral Mechanics & Osteoarthritis",
            [
                ("Definition", "Knee pain is a common orthopedic presentation arising from acute ligamentous sprain, meniscal tears, patellofemoral tracking syndrome, tendonitis, or degenerative articular cartilage wear (osteoarthritis)."),
                ("Etiology and Biomechanics", "Causes include mechanical tissue overload, sports pivoting injuries, repetitive kneeling or squatting, quadriceps and hamstring muscular weakness, obesity, and systemic inflammatory arthritis. Micro-trauma releases inflammatory mediators triggering swelling, effusion, and localized joint stiffness."),
                ("Immediate Action Plan (R.I.C.E. Protocol)", "Follow the R.I.C.E. protocol: Rest (avoid weight-bearing stress), Ice (apply cold pack wrapped in a cloth for 15-20 minutes every 3-4 hours), Compression (use an elastic bandage for mild uniform support), and Elevation (prop the knee above heart level). Avoid deep squats and running until acute symptoms subside."),
                ("Rehabilitation and Clinical Red Flags", "Long-term recovery relies on gentle range-of-motion exercises, isometric quadriceps sets, straight leg raises, and low-impact cross-training (swimming, cycling). Seek urgent orthopedic evaluation for inability to bear weight, visible joint deformity, joint locking, sudden tense joint effusion after a pop, or high fever with severe redness (septic arthritis)."),
            ]
        ),
        # PAGE 7: Diabetes Mellitus & Metabolic Regulation
        format_page_lines(
            "Diabetes Mellitus and Glycemic Regulation",
            "Endocrinology & Metabolic Disorders",
            [
                ("Definition", "Diabetes mellitus is a chronic metabolic disorder defined by chronic hyperglycemia resulting from defects in insulin secretion, insulin action, or both."),
                ("Classification and Etiology", "Type 1 diabetes is an autoimmune disease characterized by destruction of pancreatic beta cells leading to absolute insulin deficiency. Type 2 diabetes accounts for 90-95% of cases, stemming from peripheral insulin resistance coupled with progressive secretory defect. Key risk factors include visceral adiposity, physical inactivity, genetic predisposition, and chronic metabolic inflammation."),
                ("Clinical Symptoms and Complications", "Classic manifestations include the triad of polydipsia (excessive thirst), polyuria (frequent urination), and polyphagia (increased hunger), alongside blurred vision, unexplained weight loss, and chronic fatigue. Chronic uncontrolled hyperglycemia damages microvascular and macrovascular beds, leading to retinopathy, nephropathy, peripheral neuropathy, and cardiovascular disease."),
                ("Evidence-Based Management", "Comprehensive glycemic management requires balanced nutritional planning with complex carbohydrates, regular aerobic and resistance exercise (at least 150 minutes weekly), self-monitoring of blood glucose, and prescribed pharmacotherapy (e.g., metformin, SGLT2 inhibitors, GLP-1 receptor agonists, or exogenous insulin under medical direction)."),
            ]
        ),
        # PAGE 8: Respiratory Disorders, Asthma & Bronchitis
        format_page_lines(
            "Respiratory Disorders, Asthma and Acute Bronchitis",
            "Pulmonology & Bronchial Hyperreactivity",
            [
                ("Definition", "Asthma is a chronic inflammatory disorder of the airways characterized by bronchial hyperresponsiveness, reversible airflow obstruction, and recurrent episodes of wheezing, breathlessness, chest tightness, and coughing."),
                ("Etiology and Common Triggers", "Airway inflammation is orchestrated by mast cells, eosinophils, and T-lymphocytes. Bronchospasm is precipitated by environmental allergens (pollen, dust mites, pet dander), viral respiratory infections, cold air, physical exertion, airborne irritants (tobacco smoke, pollution), and strong emotional stress."),
                ("Clinical Management", "Pharmacotherapy is stratified into rescue medications (short-acting beta-agonists such as albuterol for rapid bronchodilation) and long-term controller therapies (inhaled corticosteroids to suppress underlying mucosal inflammation). Non-pharmacological measures include allergen avoidance, breathing pacing, and monitoring peak expiratory flow rates."),
                ("Emergency Warning Signs", "Immediate emergency care is mandatory if a patient exhibits severe breathlessness, inability to speak full sentences in one breath, cyanosis of lips or nails, retractions of intercostal muscles, or lack of response to rescue inhalers."),
            ]
        ),
        # PAGE 9: Infectious Illnesses, Viral Fever & Upper Respiratory Infections
        format_page_lines(
            "Infectious Illnesses, Viral Fever and Upper Respiratory Infections",
            "Infectious Diseases & Clinical Immunology",
            [
                ("Definition", "Viral upper respiratory tract infections (URI) and viral fevers are acute communicable illnesses caused by rhinovirus, influenza, coronavirus, adenovirus, or respiratory syncytial virus, manifesting as pyrexia, sore throat, and constitutional malaise."),
                ("Pathophysiology of Fever", "Pyrogens stimulate hypothalamic prostaglandin E2 (PGE2) synthesis, shifting the thermoregulatory set-point upward to facilitate host immune defense mechanisms, lymphocyte activation, and viral suppression."),
                ("Symptom Management and Self-Care", "Self-care focuses on adequate hydration (clear broths, herbal teas, oral rehydration salts), physical rest, warm saline gargles for pharyngeal comfort, cool-mist humidifiers, and antipyretics (acetaminophen or ibuprofen) used prudently as indicated by clinical guidance. Antibiotics are strictly ineffective against viral etiologies."),
                ("Red Flags Requiring Prompt Evaluation", "Consult a physician promptly if fever persists beyond 3-5 days, exceeds 39.5°C (103°F), fails to respond to antipyretics, or is accompanied by stiff neck, confusion, petechial rash, persistent vomiting, or difficulty breathing."),
            ]
        ),
        # PAGE 10: Dermatological Conditions, Eczema & Skin Rashes
        format_page_lines(
            "Dermatological Conditions, Eczema and Inflammatory Rashes",
            "Dermatology & Cutaneous Barrier Function",
            [
                ("Definition", "Eczema (atopic dermatitis) is a chronic, pruritic inflammatory skin condition characterized by epidermal barrier dysfunction, dry skin (xerosis), erythematous patches, and intense itching."),
                ("Etiology and Triggers", "Pathogenesis involves filaggrin gene mutations compromising the stratum corneum, aberrant immune Th2 activation, and cutaneous microbiome dysbiosis. Triggers include dry climate, harsh soaps, synthetic fabrics, stress, and secondary staphylococcal colonization."),
                ("Management and Barrier Repair", "Treatment centers on liberal application of ceramide-rich emollients within 3 minutes after lukewarm bathing, mild fragrance-free cleansers, topical corticosteroids or calcineurin inhibitors for acute flares, and maintaining cool indoor humidity."),
                ("Warning Signs", "Seek dermatological evaluation for widespread blistering, honey-colored crusting (impetiginization indicating secondary bacterial infection), or fever with painful skin lesions."),
            ]
        ),
        # PAGE 11: Spinal Disorders, Lumbar Back Pain & Sciatica
        format_page_lines(
            "Spinal Disorders, Lumbar Back Pain and Postural Health",
            "Physical Medicine, Spine Biomechanics & Rheumatology",
            [
                ("Definition", "Lumbar back pain is musculoskeletal discomfort originating from the lumbosacral spine, surrounding paraspinal muscles, intervertebral discs, or facet joints, affecting up to 80% of adults during their lifetime."),
                ("Etiology and Mechanisms", "Acute mechanical back strain results from heavy lifting, poor ergonomic postures during prolonged desk work, sudden torsional stress, or core muscle deconditioning. Lumbar disc herniation or spinal stenosis may cause nerve root compression (radiculopathy/sciatica) characterized by radiating pain, tingling, or numbness down the leg."),
                ("Evidence-Based Management", "Current clinical guidelines advise against prolonged strict bed rest; early resumption of light walking and gentle mobility is superior for tissue healing. Ergonomic workstation setup, alternating cold packs during initial 48 hours with warm moist heat afterward, core stabilization exercises (planks, bird-dog, pelvic tilts), and lumbar support promote recovery."),
                ("Red-Flag Emergency Symptoms (Cauda Equina)", "Immediate emergency neurosurgical evaluation is required if back pain is accompanied by loss of bowel or bladder control, saddle anesthesia (numbness in the groin/buttocks), progressive lower extremity weakness ('foot drop'), or severe pain with unexplained fever."),
            ]
        ),
        # PAGE 12: Preventative Health, Clinical Nutrition & Immune Support
        format_page_lines(
            "Preventative Health, Clinical Nutrition and Immune Support",
            "Lifestyle Medicine, Hydration & Preventative Care",
            [
                ("Definition", "Preventative medicine and clinical nutrition encompass evidence-based lifestyle behaviors designed to optimize immune function, prevent chronic disease onset, and maintain physiological homeostasis."),
                ("Hydration and Electrolyte Homeostasis", "Adequate fluid intake (approximately 2.5 to 3.5 liters daily for adults) supports renal perfusion, mucociliary clearance, cognitive sharpness, and cellular metabolism. Mild dehydration (1-2% loss of body weight) triggers fatigue, headaches, decreased concentration, and elevated perceived exertion."),
                ("Nutritional Immunity", "The human immune system requires adequate micronutrients, including Vitamin C (cellular antioxidant defense), Vitamin D (macrophage and dendritic cell regulation), Zinc (lymphocyte maturation and antiviral integrity), and dietary fiber (nurturing gut microbiota producing immunomodulatory short-chain fatty acids)."),
                ("Daily Preventative Habits", "Prioritize 7-9 hours of consolidated sleep, engage in at least 150 minutes of moderate aerobic exercise weekly, incorporate colorful phytonutrient-dense whole foods, practice daily stress decompression, and complete annual preventative clinical health screenings."),
            ]
        ),
        # PAGE 13: Peptic Ulcer Disease & H. Pylori
        format_page_lines(
            "Peptic Ulcer Disease and Gastric Mucosal Health",
            "Gastroenterology, Mucosal Defense & Helicobacter Pylori",
            [
                ("Definition", "Peptic ulcer disease (PUD) refers to mucosal defects in the stomach (gastric ulcer) or proximal duodenum (duodenal ulcer) that penetrate through the muscularis mucosae."),
                ("Pathophysiology and Etiology", "The two primary causes are chronic infection with Helicobacter pylori bacteria (which produces urease and cytotoxins that degrade protective mucus) and widespread use of NSAID pain relievers (which inhibit cyclooxygenase-1 and block mucosal prostaglandin synthesis). Smoking and intense psychological stress worsen mucosal ischemia."),
                ("Clinical Signs and Symptoms", "Patients typically present with burning epigastric discomfort occurring 2 to 3 hours post-meal or at night, temporarily relieved by antacids or food in duodenal ulcers, accompanied by bloating, belching, and nausea."),
                ("Management and Red Flags", "Treatment involves triple or quadruple eradication therapy for H. pylori, proton pump inhibitors (PPIs) for 4 to 8 weeks, avoidance of NSAIDs, alcohol, and spicy foods. Red flags include vomiting blood, passing black tarry stools, or sudden board-like abdominal rigidity indicating ulcer perforation."),
            ]
        ),
        # PAGE 14: Irritable Bowel Syndrome (IBS)
        format_page_lines(
            "Irritable Bowel Syndrome (IBS) and Gut-Brain Disorders",
            "Neurogastroenterology & Enteric Motility Syndromes",
            [
                ("Definition", "Irritable Bowel Syndrome (IBS) is a chronic disorder of gut-brain interaction characterized by recurrent abdominal pain associated with defecation or change in bowel frequency or form (diarrhea, constipation, or mixed)."),
                ("Etiology and the Brain-Gut Axis", "IBS results from altered visceral sensory processing, autonomic dysregulation, low-grade mucosal immune activation, and microbiome dysbiosis. Psychological stress and anxiety amplify pain perception through the vagal-enteric pathway without causing endoscopic tissue ulceration."),
                ("Management and Dietary Approaches", "First-line interventions include a trial of a low-FODMAP diet, regular soluble fiber intake (psyllium), peppermint oil capsules for antispasmodic effect, stress-reduction techniques (gut-directed hypnotherapy, CBT, mindfulness), and targeted probiotics."),
                ("Clinical Workup and Red Flags", "Diagnostic criteria require symptoms for at least 6 months. Warning signs requiring formal colonoscopy include onset after age 50, nocturnal diarrhea, unintentional weight loss, anemia, or family history of inflammatory bowel disease or colorectal cancer."),
            ]
        ),
        # PAGE 15: Major Depressive Disorder & Affective Regulation
        format_page_lines(
            "Major Depressive Disorder and Affective Regulation",
            "Psychiatry, Neurobiology & Clinical Mental Health",
            [
                ("Definition", "Major Depressive Disorder (MDD) is a prevalent, treatable clinical mood disorder marked by persistent sadness, loss of interest or pleasure (anhedonia), and functional impairment lasting at least two consecutive weeks."),
                ("Neurobiology and Somatic Symptoms", "Involves dysregulation of serotonin, norepinephrine, and dopamine neurotransmission, hypothalamic-pituitary-adrenal (HPA) axis hyperactivity, and reduced neuroplasticity. Somatic symptoms include profound fatigue, changes in appetite or weight, insomnia or hypersomnia, psychomotor slowing, and cognitive concentration difficulties."),
                ("Evidence-Based Therapies", "Combines evidence-based psychotherapy (Cognitive Behavioral Therapy, Interpersonal Therapy, Acceptance & Commitment Therapy) with modern pharmacotherapy (SSRIs, SNRIs). Behavioral activation, daily morning sunlight exposure, and regular aerobic exercise offer powerful neurogenic benefits."),
                ("Crisis Protocol and Safety", "Any manifestation of active suicidal intent, self-harm impulses, or feelings of hopelessness requires immediate crisis intervention: dial 988 (US/Canada), 14416 (India Tele-MANAS), or text 741741 for 24/7 confidential clinical support."),
            ]
        ),
        # PAGE 16: Thyroid Disorders (Hypo & Hyperthyroidism)
        format_page_lines(
            "Thyroid Disorders: Hypothyroidism and Hyperthyroidism",
            "Clinical Endocrinology & Metabolic Homeostasis",
            [
                ("Definition", "Thyroid disorders stem from dysfunction of the thyroid gland, leading to inadequate production of thyroid hormones (hypothyroidism) or excessive production (hyperthyroidism/thyrotoxicosis)."),
                ("Pathophysiology and Symptoms", "Hypothyroidism (commonly Hashimoto's thyroiditis) slows systemic metabolism, causing fatigue, cold intolerance, weight gain, dry coarse skin, constipation, bradycardia, and depression. Hyperthyroidism (commonly Graves' disease) accelerates metabolism, presenting with palpitations, heat intolerance, weight loss despite increased appetite, tremor, anxiety, and insomnia."),
                ("Diagnostic Assessment and Therapy", "Diagnosis is confirmed via serum thyroid-stimulating hormone (TSH) and free T4 assays. Hypothyroidism is managed with daily levothyroxine hormone replacement. Hyperthyroidism is treated with antithyroid medications (methimazole), beta-blockers for autonomic symptom relief, or radioiodine therapy."),
            ]
        ),
        # PAGE 17: Allergic Rhinitis & Seasonal Allergies
        format_page_lines(
            "Allergic Rhinitis and Environmental Allergies",
            "Immunology, Allergic Sensitization & Upper Airway Health",
            [
                ("Definition", "Allergic rhinitis is an IgE-mediated inflammatory disease of the nasal mucosa triggered by inhalation of airborne environmental allergens, affecting 10 to 30% of global populations."),
                ("Triggers and Mechanisms", "Exposure to tree, grass, or weed pollen (seasonal), or dust mites, mold spores, and pet dander (perennial) cross-links IgE on mucosal mast cells, causing immediate degranulation and release of histamine, leukotrienes, and prostaglandins."),
                ("Symptoms and Clinical Care", "Presents with paroxysmal sneezing, clear rhinorrhea, nasal congestion, itching of the nose, eyes, and palate, and allergic shiners (periorbital venous dark circles). Management includes nasal saline irrigations, second-generation oral antihistamines (cetirizine, fexofenadine), and daily intranasal corticosteroid sprays."),
            ]
        ),
        # PAGE 18: Urinary Tract Infections (UTI) & Renal Health
        format_page_lines(
            "Urinary Tract Infections (UTI) and Cystitis",
            "Urology, Nephrology & Microbial Infection",
            [
                ("Definition", "Urinary tract infections (UTIs) are microbial infections of the urinary bladder (cystitis) or kidneys (pyelonephritis), primarily caused by uropathogenic Escherichia coli bacteria ascending through the urethra."),
                ("Symptoms and Clinical Manifestations", "Lower UTI presents with dysuria (painful, burning urination), urinary urgency, increased urinary frequency, suprapubic cramping, and cloudy or foul-smelling urine."),
                ("Management and Hydration", "Mild symptoms require abundant oral hydration to flush the urinary tract. Clinicians confirm diagnosis with urinalysis and urine culture, prescribing targeted short-course oral antibiotics. Cranberry proanthocyanidins can help inhibit bacterial adherence to uroepithelial cells."),
                ("Red Flags (Pyelonephritis)", "Seek immediate emergency evaluation for high fever, rigors/chills, flank/kidney back pain, nausea, and vomiting, which indicate ascending kidney infection (pyelonephritis)."),
            ]
        ),
        # PAGE 19: Cervical Spondylosis, Neck Pain & Ergonomics
        format_page_lines(
            "Cervical Spondylosis, Neck Pain and Ergonomics",
            "Orthopedic Medicine, Spine Ergonomics & Radiculopathy",
            [
                ("Definition", "Cervical neck pain and cervical spondylosis involve age-related or postural wear-and-tear of cervical intervertebral discs and facet joints, frequently exacerbated by modern 'tech-neck' slouching."),
                ("Etiology and Postural Mechanics", "Holding the head forward at an angle increases the effective mechanical load on the cervical spine from 10-12 pounds to over 40-50 pounds, straining trapezius, levator scapulae, and sternocleidomastoid muscles."),
                ("Management and Ergonomics", "Position screens at eye level, maintain supportive sitting posture, apply warm moist compresses for muscular tightness, perform gentle chin tucks and neck range-of-motion exercises, and use an orthopedic contour pillow."),
                ("Warning Signs", "Urgent orthopedic assessment is needed for pain radiating into the arms or hands, numbness/weakness in fingers, loss of fine motor dexterity (difficulty buttoning shirts), or balance unsteadiness."),
            ]
        ),
        # PAGE 20: Shoulder Impingement & Rotator Cuff Tendinopathy
        format_page_lines(
            "Shoulder Impingement, Rotator Cuff Tendinopathy & Frozen Shoulder",
            "Orthopedic Medicine, Glenohumeral Biomechanics & Tendonitis",
            [
                ("Definition", "Shoulder pain commonly stems from subacromial impingement, rotator cuff tendinitis, subacromial bursitis, or adhesive capsulitis ('frozen shoulder'), impairing shoulder elevation and rotation."),
                ("Etiology and Biomechanics", "Repetitive overhead motions, poor scapular stabilization, or acute trauma compress the supraspinatus tendon beneath the acromion process, creating local ischemia, micro-tears, and inflammation."),
                ("Treatment and Physical Rehabilitation", "Initial treatment involves relative rest, avoiding provocative overhead lifting, applying cold packs for 15 minutes post-activity, and non-narcotic anti-inflammatory analgesics. Progressive physical therapy focuses on rotator cuff strengthening and scapular retraining."),
                ("Red Flags", "Sudden left shoulder pain accompanied by chest tightness, shortness of breath, and diaphoresis demands immediate emergency cardiac screening."),
            ]
        ),
        # PAGE 21: Ankle Sprains & Ligamentous Rehabilitation
        format_page_lines(
            "Ankle Sprains, Ligamentous Tears and R.I.C.E. Rehabilitation",
            "Sports Medicine, Orthopedic Trauma & Ligament Healing",
            [
                ("Definition", "Ankle sprains are common acute soft-tissue injuries occurring when the foot rolls inward (inversion), stretching or tearing the lateral stabilizing ligaments (primarily the anterior talofibular ligament - ATFL)."),
                ("Immediate Care Protocol (P.O.L.I.C.E. / R.I.C.E.)", "Immediately initiate the R.I.C.E. protocol: Rest (avoid bearing weight on the injured foot), Ice (cold therapy for 15-20 minutes every 2-3 hours), Compression (wrap with a snug elastic bandage starting from toes upward), and Elevation (prop ankle above heart level)."),
                ("Recovery and Functional Proprioception", "Progressive rehabilitation includes early pain-free weight-bearing with an ankle brace, alphabet ankle circles, calf stretching, and balance wobble-board training to restore neuromuscular proprioception and prevent chronic ankle instability."),
                ("Ottawa Ankle Rules (Imaging Triggers)", "Consult an urgent care physician or orthopedic specialist for X-rays if unable to take 4 steps immediately and in the clinic, or if there is point tenderness over the posterior edge of the medial or lateral malleolus."),
            ]
        ),
        # PAGE 22: Tension-Type Headache & Myofascial Trigger Points
        format_page_lines(
            "Tension-Type Headache and Myofascial Pain",
            "Neurology, Craniofacial Pain & Stress-Induced Cephalalgia",
            [
                ("Definition", "Tension-type headache (TTH) is the most prevalent primary headache disorder worldwide, described as a dull, non-throbbing, band-like tightness or pressure around both sides of the head."),
                ("Pathophysiology and Triggers", "Arises from sustained involuntary contraction of pericranial and neck muscles, peripheral myofascial sensitization, and altered central pain processing. Primary triggers include mental stress, eye strain from digital screens, poor posture, fatigue, and skipped meals."),
                ("Therapeutic Relief and Prevention", "Immediate comfort includes taking a break from screens, gentle neck and shoulder massage, applying warm compresses to tight neck muscles, drinking a large glass of water, and over-the-counter analgesics. Long-term prevention emphasizes stress management, regular sleep, and ergonomic desk adjustments."),
            ]
        ),
        # PAGE 23: Vertigo, BPPV & Vestibular Disorders
        format_page_lines(
            "Vertigo, Benign Paroxysmal Positional Vertigo (BPPV) & Vestibular Health",
            "Otolaryngology, Vestibular System & Neuro-Otology",
            [
                ("Definition", "Vertigo is the false illusion of environmental motion, spinning, or tilting caused by dysfunction within the peripheral vestibular apparatus of the inner ear or central vestibular pathways."),
                ("Common Etiologies (BPPV vs Labyrinthitis)", "The most common form is Benign Paroxysmal Positional Vertigo (BPPV), caused by displaced calcium carbonate otoconia crystals dislodged into the posterior semicircular canal. Other causes include vestibular neuritis (viral inflammation), Meniere's disease, and labyrinthitis."),
                ("Repositioning Maneuvers and Safety", "BPPV is clinically cured via particle repositioning procedures such as the Epley maneuver, which uses gravity to guide crystals back into the vestibule. Patients should move slowly, avoid sudden head turns, and ensure home fall prevention."),
                ("Emergency Warning Signs (Stroke Screening)", "Urgent emergency medical evaluation is critical if vertigo is accompanied by sudden weakness, difficulty speaking (dysarthria), double vision (diplopia), loss of limb coordination, or severe headache."),
            ]
        ),
        # PAGE 24: Iron Deficiency Anemia & Hematological Fatigue
        format_page_lines(
            "Iron Deficiency Anemia and Hematological Health",
            "Hematology, Oxygen Transport & Erythropoiesis",
            [
                ("Definition", "Iron deficiency anemia is a condition in which total body iron stores become depleted, compromising hemoglobin synthesis and diminishing the oxygen-carrying capacity of circulating red blood cells."),
                ("Causes and Clinical Features", "Causes include chronic blood loss (menstrual or gastrointestinal), inadequate dietary iron intake, impaired absorption (celiac disease, post-gastric bypass), or increased physiological demand (pregnancy, rapid growth). Symptoms include insidious fatigue, pallor, cold hands, brittle spoon-shaped nails (koilonychia), dizziness, and cravings for non-nutritive substances (pica/ice chewing)."),
                ("Diagnostic Confirmation and Dietary Therapy", "Confirmed by low serum ferritin, low hemoglobin, and microcytic hypochromic red blood cells. Treatment includes oral ferrous iron supplementation taken with vitamin C for enhanced absorption, paired with iron-rich foods (lentils, dark leafy greens, beans, lean meats)."),
            ]
        ),
        # PAGE 25: Vitamin D Deficiency & Bone Mineralization
        format_page_lines(
            "Vitamin D Deficiency and Bone Mineral Density",
            "Metabolic Bone Disease, Calcium Homeostasis & Immunology",
            [
                ("Definition", "Vitamin D deficiency is a widespread nutritional condition marked by circulating serum 25-hydroxyvitamin D concentrations below 20 ng/mL, essential for intestinal calcium absorption and skeletal integrity."),
                ("Pathophysiology and Symptoms", "Insufficient sun exposure, skin pigmentation, obesity, and limited dietary sources lead to secondary hyperparathyroidism, accelerated bone resorption, osteomalacia (softening of bones), bone pain, proximal muscle weakness, and impaired immune defense."),
                ("Clinical Management and Supplementation", "Management requires measured oral cholecalciferol (Vitamin D3) supplementation under physician supervision, sensible sun exposure, and dietary inclusion of fortified milk, egg yolks, and fatty fish."),
            ]
        ),
        # PAGE 26: Dengue, Malaria & Tropical Vector-Borne Fevers
        format_page_lines(
            "Dengue, Malaria and Tropical Vector-Borne Febrile Syndromes",
            "Tropical Medicine, Arboviral Infections & Parasitology",
            [
                ("Definition", "Dengue and malaria are life-threatening vector-borne acute febrile illnesses transmitted via mosquitoes (Aedes aegypti for Dengue; Anopheles for Plasmodium malaria)."),
                ("Clinical Manifestations", "Dengue causes acute high fever, retro-orbital eye pain, severe joint and muscle aches ('breakbone fever'), and petechial skin rashes. Malaria manifests as cyclical paroxysms of severe chills/rigors followed by spiking fevers and drenching sweats."),
                ("Management and Monitoring", "Dengue requires strict oral and intravenous hydration, avoidance of NSAIDs/aspirin (which worsen bleeding risks), and monitoring of platelet counts and hematocrit. Malaria requires diagnostic peripheral blood smear microscopy or rapid antigen testing, followed by artemisinin-based combination therapies (ACTs)."),
                ("Severe Red Flags", "Emergency hospitalization is required for severe abdominal pain, persistent vomiting, mucosal bleeding from gums or nose, lethargy, or rapid drop in blood pressure (dengue shock syndrome)."),
            ]
        ),
        # PAGE 27: Typhoid Fever & Salmonella Enteric Illness
        format_page_lines(
            "Typhoid Fever and Salmonella Enteric Illness",
            "Infectious Diseases, Enteric Bacteriology & Systemic Infection",
            [
                ("Definition", "Typhoid (enteric) fever is a systemic bacterial infection caused by Salmonella enterica serotype Typhi, transmitted through food or water contaminated with feces."),
                ("Clinical Course and Symptoms", "Features a gradual stepladder fever climbing daily, constitutional headache, dry cough, malaise, relative bradycardia (Faget's sign), abdominal discomfort, splenomegaly, and characteristic 'rose spots' on the trunk."),
                ("Diagnosis and Clinical Care", "Confirmed via blood cultures, bone marrow cultures, or rapid serological assays. Treatment necessitates culture-guided antibiotic therapy (such as azithromycin, ceftriaxone, or fluoroquinolones), safe drinking water practices, and adequate nutritional support."),
                ("Complications", "Life-threatening complications such as intestinal hemorrhage or ileal perforation typically manifest during the third week of untreated illness."),
            ]
        ),
        # PAGE 28: Polycystic Ovary Syndrome (PCOS)
        format_page_lines(
            "Polycystic Ovary Syndrome (PCOS) and Endocrine Health",
            "Reproductive Endocrinology, Hyperandrogenism & Insulin Resistance",
            [
                ("Definition", "PCOS is a common heterogeneous endocrine disorder affecting 6 to 15% of reproductive-aged women, defined by the Rotterdam criteria (at least two of: ovulatory dysfunction/irregular cycles, biochemical or clinical hyperandrogenism, and polycystic ovarian morphology on ultrasound)."),
                ("Pathophysiology and Symptoms", "Underlying pathophysiology centers on insulin resistance, hyperinsulinemia, and ovarian theca-cell androgen hypersecretion. Manifestations include oligomenorrhea, hirsutism (excessive facial/body hair), severe cystic acne, androgenic alopecia, and central weight gain."),
                ("Multimodal Clinical Management", "First-line therapy focuses on lifestyle modification (low-glycemic nutritional diet, regular aerobic and resistance exercise) to enhance insulin sensitivity. Pharmacotherapy includes combined oral contraceptive pills for cycle regulation and androgen suppression, and metformin for metabolic stabilization."),
            ]
        ),
        # PAGE 29: Acute Gastroenteritis & Food Poisoning
        format_page_lines(
            "Acute Gastroenteritis, Food Poisoning and Rehydration",
            "Gastroenterology, Enteric Pathogens & Electrolyte Homeostasis",
            [
                ("Definition", "Acute gastroenteritis is sudden mucosal inflammation of the stomach and intestines caused by foodborne bacterial toxins (Staphylococcus aureus, Bacillus cereus), bacterial pathogens (Campylobacter, Salmonella), or enteric viruses (Norovirus, Rotavirus)."),
                ("Pathophysiology and Symptoms", "Presents with acute nausea, projectile vomiting, watery diarrhea, diffuse cramping abdominal pain, low-grade fever, and dehydration due to rapid intestinal fluid and electrolyte secretion."),
                ("Rehydration Therapy and Care", "The cornerstone of management is continuous oral rehydration therapy using WHO-formulated oral rehydration salts (ORS), sipped slowly in small quantities. Avoid anti-motility drugs (like loperamide) during acute infectious diarrhea, and avoid dairy, fatty, or caffeinated items until normal stool consistency returns."),
                ("Emergency Warning Signs", "Seek immediate emergency intravenous care for severe dehydration (sunken eyes, dizziness on standing, absence of urination for over 8 hours), high fever, inability to tolerate oral sips, or bloody stools (dysentery)."),
            ]
        ),
        # PAGE 30: Chronic Fatigue, Burnout & Neuroendocrine Stress
        format_page_lines(
            "Chronic Fatigue, Academic Burnout & Neuroendocrine Stress Adaptation",
            "Behavioral Medicine, Neuroendocrinology & Cortisol Homeostasis",
            [
                ("Definition", "Burnout and chronic stress-related exhaustion represent a state of profound emotional, physical, and cognitive depletion resulting from chronic, unmitigated occupational or academic stressors."),
                ("Pathophysiology and Systems Impact", "Prolonged stress hyperactivates the hypothalamic-pituitary-adrenal (HPA) axis, leading to chronically dysregulated cortisol secretion, autonomic nervous exhaustion, impaired prefrontal cognitive processing, and down-regulated immune responsiveness."),
                ("Holistic Recovery and Boundary Setting", "Sustainable recovery requires structural boundary setting (limiting screen hours, dedicated study disconnect periods), physiological restorative sleep (8 hours consolidated sleep), progressive somatic relaxation, restorative nutritional meals, and cognitive reframing."),
                ("Clinical Referral", "Persistent fatigue exceeding six months unresponsive to rest requires clinical workup to rule out chronic fatigue syndrome (ME/CFS), occult anemia, hypothyroidism, or major depressive disorder."),
            ]
        ),
    ]

    # Generate PDF Objects
    objects = []
    
    # Catalog
    objects.append("1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj")
    
    # Pages root
    page_obj_ids = [3 + i * 2 for i in range(len(pages_data))]
    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects.append(f"2 0 obj\n<</Type /Pages /Kids [{kids_str}] /Count {len(pages_data)}>>\nendobj")

    font_obj_idx = 3 + len(pages_data) * 2

    # Build each page
    for idx, page_lines in enumerate(pages_data):
        page_idx = 3 + idx * 2
        content_idx = page_idx + 1

        stream_lines = ["BT", "/F1 9 Tf", "40 750 Td", "14 TL"]
        for line in page_lines:
            safe_text = (
                line.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
                .encode("latin1", "replace")
                .decode("latin1")
            )
            stream_lines.append(f"({safe_text}) Tj")
            stream_lines.append("T*")
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
    print(f"Successfully generated full {len(pages_data)}-page Gale Encyclopedia PDF at {out_path} ({len(pdf_content)} bytes)")


if __name__ == "__main__":
    generate_pdf()
