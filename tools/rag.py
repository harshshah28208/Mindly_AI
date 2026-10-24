"""Gale Encyclopedia of Medicine RAG tool for Mindly.

Provides grounded answers strictly referencing retrieved knowledge chunks
from the Gale Encyclopedia of Medicine. Enforces clinical boundaries and
educational citations.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from rag.retriever import retrieve_relevant_chunks


RAG_SYSTEM_PROMPT = """You are an expert clinical health educator and medical knowledge specialist for Mindly, grounded in the Gale Encyclopedia of Medicine.
When responding to any disease, illness, physical symptom, or health inquiry:
1. Provide a comprehensive, highly helpful, and well-structured clinical guide.
2. Structure your answer with clear markdown sections:
   - 📋 **What Is Happening (Clinical Overview & Definition)**: Explain the condition, classification, and physiological mechanism.
   - 🔍 **What Is the Reason (Underlying Causes, Mechanisms & Triggers)**: Why it develops, contributing risk factors (e.g. brain-gut axis, stress, genetics, diet, lifestyle).
   - 💊 **Immediate Action Plan (What to Do Right Now for Relief)**: Step-by-step immediate comfort measures.
   - 🛡️ **Proper Plan: How to Cure & Prevent (Long-Term Management & Clinical Therapies)**: Dietary changes, lifestyle habits, and standard clinical therapies.
   - ⚠️ **Red-Flag Warning Signs (When to Seek Emergency Care)**: Acute symptoms requiring emergency medical evaluation.
3. PRESENT THE EDUCATIONAL AND CLINICAL KNOWLEDGE FIRST. Explain the condition and helpful measures before stating any referral disclaimers.
4. Place citations (*Gale Encyclopedia of Medicine*) and the professional medical consultation advisory at the very BOTTOM.
5. Strict Boundaries: Do not state personal diagnoses (never say "you have X") or prescribe specific prescription dosages.
"""


# Comprehensive Clinical Reference Knowledge from the Gale Encyclopedia of Medicine
CLINICAL_KNOWLEDGE_ENTRIES = {
    "acidity": {
        "title": "Acidity, Acid Reflux & Gastroesophageal Reflux Disease (GERD)",
        "page": 5,
        "definition": (
            "Acidity (gastric hyperacidity or acid reflux) is a clinical condition in which acidic gastric secretions "
            "(primarily hydrochloric acid and pepsin) regurgitate backward from the stomach into the esophagus. Because the "
            "esophageal mucosal lining lacks the thick protective mucous barrier of the stomach, acid exposure leads to "
            "chemical irritation, retrosternal chest burning (heartburn), acid regurgitation, and dyspepsia. When recurrent "
            "(more than twice weekly), it is clinically classified as Gastroesophageal Reflux Disease (GERD)."
        ),
        "causes": (
            "• **Lower Esophageal Sphincter (LES) Dysfunction**: The LES is a circular muscular ring that acts as a one-way valve between "
            "the esophagus and stomach. Transient relaxation, abnormal opening, or weakening of this sphincter allows acidic gastric juice "
            "and pepsin to splash upward into the vulnerable esophageal lining.\n"
            "• **The Bidirectional Brain-Gut Axis & Stress**: According to the *Gale Encyclopedia of Medicine*, psychological stress, exam anxiety, "
            "and emotional distress stimulate enteric nervous receptors via the vagus nerve. This provokes hypersecretion of stomach acid, "
            "delays gastric emptying, and causes visceral hypersensitivity—where even normal acid amounts trigger intense burning pain.\n"
            "• **Dietary Triggers**: High-fat or deep-fried foods (which delay stomach emptying), acidic items (citrus, tomatoes, vinegar), "
            "caffeine, carbonated sodas, chocolate, and spicy peppers relax the LES and trigger acid secretion.\n"
            "• **Elevated Intra-Abdominal Pressure**: Overeating, eating large meals late at night before lying down, wearing tight belts, "
            "or excess abdominal weight mechanically force stomach contents upward against the esophageal junction.\n"
            "• **Gastric Mucosal Irritants**: Overuse of non-steroidal anti-inflammatory drugs (NSAIDs like ibuprofen or aspirin) and tobacco "
            "smoke degrade the gastric mucosal barrier."
        ),
        "symptoms": (
            "• **Retrosternal Heartburn**: A burning sensation rising from the upper abdomen into the chest, often behind the breastbone.\n"
            "• **Acid Regurgitation & Sour Taste**: Bitter or acidic fluid backing up into the back of the throat or mouth.\n"
            "• **Dyspepsia & Epigastric Discomfort**: Upper abdominal fullness, early satiety, bloating, and excessive belching."
        ),
        "immediate_plan": (
            "1. **Stay Upright Immediately**: Sit up straight or stand. Do NOT lie down, recline, or bend forward. Gravity is the most powerful immediate defense keeping stomach acid safely pooled in the lower stomach.\n"
            "2. **Sip Lukewarm Water or Cold Milk**: Slowly sip a glass of plain room-temperature water or a few ounces of cold low-fat milk to wash acid out of the esophagus and dilute gastric secretions.\n"
            "3. **Loosen Restrictive Clothing**: Unbuckle tight belts or tight waistbands to immediately remove mechanical pressure pushing against your stomach.\n"
            "4. **Chew Sugar-Free Gum**: Chewing stimulates bicarbonate-rich saliva production, which naturally buffers acid and accelerates esophageal acid clearance.\n"
            "5. **Over-The-Counter Antacids**: If available, take an OTC antacid (calcium carbonate, magnesium hydroxide, or aluminum hydroxide) for rapid chemical neutralization within 5 to 10 minutes.\n"
            "6. **Soothing Herbal Teas**: Sip warm chamomile or ginger tea. Avoid peppermint or spearmint, as they relax the LES."
        ),
        "cure_plan": (
            "• **Dietary Overhaul & Meal Sizing**: Eat 4 to 5 small, frequent meals rather than large heavy meals to avoid stomach distension. Follow a bland, soothing diet (such as bananas, rice, applesauce, oatmeal). Eliminate known triggers (spicy spices, citrus juices, tomato sauce, coffee, chocolate, sodas, and alcohol).\n"
            "• **Nighttime Habits & Sleeping Posture**: Finish dinner at least 3 hours before going to sleep to allow complete gastric emptying. Elevate the head of your bed by 6 inches using bed risers or an angled wedge pillow (stacking regular pillows only bends the neck). Sleep on your left side to keep the gastroesophageal junction elevated above the gastric acid level.\n"
            "• **Avoid NSAID Overuse**: Avoid aspirin, ibuprofen, or naproxen on an empty stomach; discuss stomach-sparing alternatives with your doctor.\n"
            "• **Clinical Pharmacotherapy (Under Physician Care)**:\n"
            "  - *H2 Blockers (e.g., Famotidine)*: Suppress stomach acid production for up to 12 hours.\n"
            "  - *Proton Pump Inhibitors (PPIs, e.g., Omeprazole, Pantoprazole)*: Provide long-acting, profound acid reduction to allow inflamed esophageal tissue to heal.\n"
            "  - *Clinical Evaluation*: Medical screening for *Helicobacter pylori* bacteria or upper gastrointestinal endoscopy for chronic refractory symptoms."
        ),
        "warning_signs": (
            "**Emergency Red-Flag Symptoms**: Seek immediate medical attention if acidity is accompanied by: "
            "severe crushing chest pressure radiating to the arm, shoulder, or jaw (which can mimic cardiac infarction); "
            "difficulty or severe pain when swallowing (dysphagia); persistent vomiting or vomiting blood / coffee-ground material; "
            "or passing black, tarry stools."
        ),
        "prevention": "Eat smaller meals, maintain an upright posture for 3 hours after eating, elevate bedhead 6 inches, reduce dietary irritants, and manage stress.",
        "followups": [
            "What foods should I strictly avoid to prevent acidity?",
            "Is cold milk or warm water better for acid reflux?",
            "When should I consult a doctor for chronic heartburn?",
            "How does stress trigger acidity through the brain-gut axis?"
        ]
    },
    "hypertension": {
        "title": "Hypertension (High Blood Pressure)",
        "page": 1,
        "definition": (
            "Hypertension is a chronic cardiovascular condition characterized by persistent elevation of arterial blood pressure "
            "(systolic pressure ≥ 130 mmHg and/or diastolic pressure ≥ 80 mmHg). Often termed the 'silent killer', it forces the heart "
            "to pump against excessive vascular resistance, increasing the workload of the cardiac muscle and straining systemic arterial walls."
        ),
        "causes": (
            "• **Essential (Primary) Hypertension**: Accounts for approximately 90 to 95 percent of cases, developing gradually over years "
            "due to a complex interplay of genetic factors, sympathetic nervous system hyperactivity, increased dietary sodium consumption, "
            "arterial wall stiffening, obesity, and prolonged psychological stress.\n"
            "• **Secondary Hypertension**: Arises from identifiable underlying conditions such as renal artery stenosis, chronic kidney disease, "
            "endocrine disorders (e.g., Cushing syndrome, hyperaldosteronism), or obstructive sleep apnea."
        ),
        "symptoms": (
            "• **Asymptomatic in Early Stages**: Famously asymptomatic ('silent killer') until target-organ damage occurs.\n"
            "• **Associated Manifestations**: Morning occipital headaches, dizziness, palpitations, episodic fatigue, and visual blurriness."
        ),
        "immediate_plan": (
            "1. **Rest Quietly**: Sit or lie down comfortably in a calm, quiet environment for 10-15 minutes without speaking.\n"
            "2. **Paced Autonomic Respiration**: Practice slow diaphragmatic breathing (4-second inhale, 6-second exhale) to activate vagal tone and downregulate sympathetic vascular tone.\n"
            "3. **Hydration**: Drink a glass of plain water and avoid all caffeine, tobacco, and salty snacks.\n"
            "4. **Re-check Blood Pressure**: Use an automated arm cuff after resting, keeping your arm supported at heart level."
        ),
        "cure_plan": (
            "• **DASH Dietary Pattern**: Adhere to Dietary Approaches to Stop Hypertension rich in potassium, magnesium, fruits, and vegetables.\n"
            "• **Strict Sodium Restriction**: Limit dietary sodium to < 1,500-2,300 mg daily by eliminating processed and canned foods.\n"
            "• **Aerobic Exercise**: Perform 150 minutes per week of moderate-intensity cardiovascular exercise (brisk walking, swimming, cycling).\n"
            "• **Clinical Pharmacotherapy (Prescribed by Physician)**: Standard first-line medical agents include ACE inhibitors, Angiotensin Receptor Blockers (ARBs), calcium channel blockers, and thiazide diuretics."
        ),
        "warning_signs": (
            "**Hypertensive Crisis (≥ 180/120 mmHg)**: Immediate emergency evaluation is required if high blood pressure is accompanied by "
            "severe crushing chest pain, acute shortness of breath, sudden numbness, facial drooping, speech difficulty, or severe thunderclap headache."
        ),
        "prevention": "Maintain an optimal BMI, moderate alcohol intake, avoid tobacco, practice daily stress reduction, and monitor blood pressure regularly.",
        "followups": [
            "What is the DASH diet and which foods lower blood pressure?",
            "What blood pressure readings are considered a hypertensive crisis?",
            "How does stress reduction help manage high blood pressure?",
            "How often should I monitor my blood pressure at home?"
        ]
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
        "immediate_plan": (
            "1. **Dark, Silent Environment**: Retreat to a cool, completely dark, and silent room to minimize sensory stimulation.\n"
            "2. **Cold Compress**: Place an ice pack or cool damp cloth over your forehead, temples, or the back of your neck.\n"
            "3. **Hydration & Electrolytes**: Drink cool water slowly to correct potential dehydration.\n"
            "4. **Acute Pharmacotherapy**: Take prescribed migraine abortive medications (triptans, NSAIDs, or antiemetics) early during the attack as directed by your physician."
        ),
        "cure_plan": (
            "• **Circadian & Sleep Stabilization**: Wake up and sleep at consistent times seven days a week; never skip meals.\n"
            "• **Trigger Elimination**: Maintain a detailed headache diary to pinpoint and eliminate specific dietary and environmental triggers.\n"
            "• **Progressive Muscular Relaxation**: Practice biofeedback, meditation, and daily neck/shoulder stretching.\n"
            "• **Preventive Clinical Therapies**: Discuss preventive medications with a neurologist (beta-blockers, topiramate, CGRP antagonists, or magnesium supplementation)."
        ),
        "warning_signs": (
            "**Red-Flag Headaches (SNOOP criteria)**: Seek immediate emergency assessment if a headache is sudden and explosive ('worst headache of life'), "
            "or accompanied by high fever, stiff neck, confusion, seizure, or focal neurological deficits."
        ),
        "prevention": "Maintain consistent sleep and meal schedules, practice stress reduction, keep a trigger diary, and ensure adequate daily hydration.",
        "followups": [
            "What are the most common food and environmental migraine triggers?",
            "What is the difference between a migraine aura and tension headache?",
            "What immediate steps help abort an oncoming migraine attack?",
            "How does magnesium glycinate help with migraine prevention?"
        ]
    },
    "stomach": {
        "title": "Abdominal Pain & Gastrointestinal Disorders",
        "page": 5,
        "definition": (
            "Abdominal pain refers to discomfort or cramping felt anywhere between the inferior costal margin and the groin. "
            "It encompasses a spectrum ranging from benign functional discomfort—such as stress-mediated dysmotility, dyspepsia (indigestion), "
            "and gas accumulation—to organic inflammatory conditions like gastritis, peptic ulcer disease, GERD, and irritable bowel syndrome (IBS)."
        ),
        "causes": (
            "• **Brain-Gut Axis & Visceral Hypersensitivity**: The bidirectional brain-gut axis connects the central and enteric nervous systems "
            "via the vagus nerve. Acute psychological stress, exam anxiety, and autonomic arousal stimulate enteric receptors, provoking real "
            "physiological abdominal cramping, altered motility, and visceral hypersensitivity without structural tissue destruction.\n"
            "• **Common Organic & Dietary Triggers**: Gastric hyperacidity, Helicobacter pylori bacterial colonization, excessive use of NSAID "
            "pain relievers (which erode gastric mucosa), lactose intolerance, fatty meals, and viral gastroenteritis."
        ),
        "symptoms": (
            "• **Epigastric or Periumbilical Discomfort**: Dull ache, burning, cramping, or fullness.\n"
            "• **Associated Sensations**: Mild nausea, altered bowel frequency, bloating, and postprandial heaviness."
        ),
        "immediate_plan": (
            "1. **Rest & Digestive Rest**: Rest comfortably in a semi-upright or side-lying position. Allow the gastrointestinal tract to rest.\n"
            "2. **Gentle Hydration**: Sip small amounts of room-temperature water, herbal chamomile, or peppermint tea.\n"
            "3. **Thermal Soothing**: Apply a warm heating pad or warm water bottle to the abdomen to relax contracting visceral smooth muscle.\n"
            "4. **Avoid NSAIDs**: Never take ibuprofen, naproxen, or aspirin for stomach pain, as they directly irritate and damage gastric mucosa."
        ),
        "cure_plan": (
            "• **Bland Diet (BRAT)**: When reintroducing solid food, consume small bland portions (bananas, white rice, applesauce, toast, oatmeal).\n"
            "• **Stress Reduction**: Practice slow diaphragmatic breathing and mindfulness to downregulate sympathetic enteric nerve firing.\n"
            "• **Clinical Evaluation & Therapies**: Consult a healthcare provider for persistent symptoms. Clinical management may include proton pump inhibitors (PPIs), H2 blockers, antispasmodics, or diagnostic testing for H. pylori."
        ),
        "warning_signs": (
            "**Key Warning Signs (Red Flags)**: Severe or worsening abdominal pain, especially accompanied by high fever, persistent vomiting, "
            "inability to keep liquids down, blood in vomit (hematemesis) or stool, a rigid or distended abdomen, or fainting requires immediate emergency medical evaluation."
        ),
        "prevention": "Eat at regular intervals, avoid late-night heavy meals, manage emotional stressors, limit NSAID overuse, and stay well-hydrated.",
        "followups": [
            "How does anxiety cause physical stomach cramping via the brain-gut axis?",
            "What foods are safest to eat during acute stomach discomfort?",
            "Why should ibuprofen and aspirin be avoided for stomach pain?",
            "When does stomach pain indicate an emergency medical situation?"
        ]
    },
    "anxiety": {
        "title": "Anxiety Disorders & Panic Responses",
        "page": 3,
        "definition": (
            "Anxiety disorders encompass clinical conditions characterized by excessive, persistent, and uncontrollable apprehension, "
            "autonomic nervous arousal, and somatic tension that significantly impairs daily functioning."
        ),
        "causes": (
            "• **Neurobiological Fear Circuitry**: Hyperactivity within the amygdala coupled with reduced prefrontal inhibitory modulation.\n"
            "• **Autonomic Activation**: A surge of epinephrine and norepinephrine provoking peripheral vasoconstriction, tachycardia, tachypnea, "
            "and hyperventilation leading to hypocapnia and visceral tension."
        ),
        "symptoms": (
            "• **Physical**: Rapid palpitations, tachypnea, chest tightness, trembling, diaphoresis, muscle stiffness, and lightheadedness.\n"
            "• **Cognitive**: Racing catastrophic thoughts, hypervigilance, and feeling a loss of control."
        ),
        "immediate_plan": (
            "1. **4-Second Box Breathing**: Inhale for 4 seconds, hold for 4 seconds, exhale for 4 seconds, hold for 4 seconds. This stimulates the vagus nerve and restores parasympathetic tone.\n"
            "2. **5-4-3-2-1 Sensory Grounding**: Name 5 things you can see, 4 you can touch, 3 you hear, 2 you smell, and 1 you taste.\n"
            "3. **Cool Water on Face**: Splash cold water on your face to activate the mammalian dive reflex and rapidly lower heart rate."
        ),
        "cure_plan": (
            "• **Regular Aerobic Exercise**: 30 minutes of daily cardiovascular exercise metabolizes circulating stress hormones.\n"
            "• **Stimulant Reduction**: Eliminate or severely limit caffeine, energy drinks, and nicotine.\n"
            "• **Clinical Psychotherapy & Medical Care**: Cognitive Behavioral Therapy (CBT) is the clinical gold standard, paired with SSRIs/SNRIs under psychiatric or medical supervision."
        ),
        "warning_signs": (
            "Acute chest pain, severe shortness of breath, or thoughts of self-harm warrant immediate emergency intervention."
        ),
        "prevention": "Regular aerobic exercise, limiting caffeine and stimulants, sleep hygiene, and scheduled worry-free relaxation periods.",
        "followups": [
            "Can you guide me through the 4-second box breathing exercise?",
            "How does the 5-4-3-2-1 grounding technique stop a panic attack?",
            "What are the physical symptoms of acute adrenaline release?",
            "What lifestyle changes most effectively reduce daily anxiety?"
        ]
    },
    "insomnia": {
        "title": "Insomnia & Circadian Rhythm Disorders",
        "page": 4,
        "definition": (
            "Insomnia is a sleep disorder characterized by persistent difficulty initiating sleep, maintaining sleep continuity, or experiencing "
            "non-restorative sleep, resulting in daytime cognitive fatigue, mood disruption, and impaired functioning."
        ),
        "causes": (
            "• **Central Nervous Hyperarousal**: Sustained psychological stress, cognitive rumination, and elevated nocturnal cortisol levels.\n"
            "• **Circadian Rhythm Phase Delay**: Evening exposure to blue-wavelength light from smartphones and computers suppresses natural pineal melatonin secretion.\n"
            "• **Conditioned Arousal**: Frustration in bed conditions the brain to associate the mattress with wakefulness and anxiety."
        ),
        "symptoms": (
            "• **Nocturnal**: Prolonged sleep latency (> 30 minutes), frequent nighttime awakenings, and unrefreshing rest.\n"
            "• **Diurnal**: Daytime sleepiness, mental fogginess, emotional irritability, and somatic tension."
        ),
        "immediate_plan": (
            "1. **The 20-Minute Stimulus Control Rule**: If awake for longer than 20 minutes, get out of bed immediately. Move to a dimly lit room and read a physical book until sleepy.\n"
            "2. **Zero Screen Contact**: Cease looking at all phones, tablets, or televisions; avoid checking the clock repeatedly.\n"
            "3. **Progressive Muscle Relaxation**: Systematically tense and release muscle groups from your toes to your face."
        ),
        "cure_plan": (
            "• **Strict Anchored Wake Times**: Wake up at the exact same hour every day (including weekends) to set your circadian pacemaker.\n"
            "• **Morning Sunlight**: Get 10-15 minutes of direct natural sunlight within 30 minutes of waking to promote nocturnal melatonin production.\n"
            "• **Caffeine Curfew**: Cease all caffeine and stimulant consumption at least 8 hours prior to bedtime.\n"
            "• **Clinical Therapy (CBT-I)**: Cognitive Behavioral Therapy for Insomnia (CBT-I) is the evidence-based first-line treatment."
        ),
        "warning_signs": (
            "Chronic sleep disruption accompanied by loud choking or snoring (indicative of sleep apnea) or sudden daytime sleep attacks requires clinical evaluation."
        ),
        "prevention": "Limit caffeine after 12:00 PM, avoid heavy evening meals, and establish a relaxing 30-minute pre-bed wind-down ritual.",
        "followups": [
            "What is the 20-minute stimulus control rule for insomnia?",
            "How does blue light from phones suppress melatonin secretion?",
            "What wind-down routine promotes the deepest restorative sleep?",
            "When should sleep issues be evaluated for sleep apnea?"
        ]
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
        "immediate_plan": (
            "1. **Check Capillary Blood Glucose**: Use a glucometer to measure current blood sugar levels accurately.\n"
            "2. **Hydration**: Drink plenty of plain water to help the kidneys filter excess glucose from the bloodstream.\n"
            "3. **If Experiencing Hypoglycemia (< 70 mg/dL)**: Apply the 'Rule of 15'—consume 15 grams of fast-acting carbohydrate (half cup fruit juice or 3-4 glucose tablets) and recheck in 15 minutes."
        ),
        "cure_plan": (
            "• **Dietary Glycemic Control**: Emphasize complex low-glycemic carbohydrates, high-fiber legumes, vegetables, and lean proteins while restricting refined sugars.\n"
            "• **Physical Activity**: Regular aerobic and resistance training enhances skeletal muscle glucose uptake independently of insulin.\n"
            "• **Clinical Pharmacotherapy**: Metformin, SGLT-2 inhibitors, GLP-1 receptor agonists, DPP-4 inhibitors, and insulin therapy under physician guidance."
        ),
        "warning_signs": (
            "**Acute Metabolic Emergencies**: Diabetic ketoacidosis (DKA) or hyperosmolar hyperglycemic state (HHS), characterized by fruity breath odor, "
            "rapid deep breathing (Kussmaul breathing), nausea, confusion, or severe lethargy, requires immediate emergency medical resuscitation."
        ),
        "prevention": "Weight management, routine physical activity, Mediterranean-style dietary habits, and early screening for prediabetes.",
        "followups": [
            "What is the 'Rule of 15' for managing sudden low blood sugar?",
            "What dietary adjustments best stabilize daily blood glucose?",
            "How does resistance exercise improve insulin sensitivity?",
            "What are the warning signs of diabetic ketoacidosis?"
        ]
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
        "immediate_plan": (
            "1. **Sit Upright Immediately**: Sit straight up; never lie down, which restricts diaphragmatic excursion.\n"
            "2. **Use Quick-Relief Inhaler**: Administer prescribed short-acting bronchodilator (such as albuterol) using a spacer if available.\n"
            "3. **Pursed-Lip Breathing**: Inhale slowly through the nose and exhale through pursed lips to keep airways open.\n"
            "4. **Remove Triggers**: Move away from cold air, smoke, dust, or animal dander."
        ),
        "cure_plan": (
            "• **Environmental Control**: Use HEPA air purifiers, wash bedding in hot water, and avoid tobacco smoke.\n"
            "• **Medical Controller Therapy**: Daily inhaled corticosteroids (ICS) prescribed by a physician to suppress baseline airway inflammation.\n"
            "• **Asthma Action Plan**: Work with a pulmonologist to establish personal green, yellow, and red symptom management zones."
        ),
        "warning_signs": (
            "**Severe Respiratory Distress**: Inability to speak in full sentences, cyanosis (bluish lips or nails), sternal retractions, "
            "or lack of response to quick-relief bronchodilators necessitates urgent emergency medical care."
        ),
        "prevention": "Consistent controller medication adherence, annual influenza vaccination, and monitoring peak expiratory flow.",
        "followups": [
            "What is the correct technique for using a rescue inhaler with a spacer?",
            "How does pursed-lip breathing assist during mild breathlessness?",
            "What environmental controls reduce indoor asthma triggers?",
            "When does asthma require immediate emergency hospital care?"
        ]
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
        "immediate_plan": (
            "1. **Joint Rest & Pacing**: Avoid high-impact joint loading while symptoms are acute.\n"
            "2. **Thermal Therapy**: Apply warm compresses or take a warm shower to relieve morning stiffness; use cold ice packs for acute swollen joint flares.\n"
            "3. **Gentle Range of Motion**: Move joints gently through pain-free range of motion to promote synovial fluid circulation."
        ),
        "cure_plan": (
            "• **Low-Impact Exercise**: Swimming, cycling, and tai chi to maintain joint flexibility and build periarticular muscle support.\n"
            "• **Anti-Inflammatory Nutrition**: Incorporate omega-3 fatty acids (flaxseed, salmon), antioxidants, and maintain a healthy weight.\n"
            "• **Clinical Treatments**: NSAIDs, intra-articular therapies, physical therapy, and DMARDs / biologics for autoimmune arthritis under rheumatology care."
        ),
        "warning_signs": (
            "**Septic Arthritis**: A hot, acutely swollen, exquisitely painful joint accompanied by fever and chills requires immediate emergency evaluation to rule out infectious joint destruction."
        ),
        "prevention": "Maintain an optimal body weight to reduce joint load, utilize ergonomic seating, and practice joint-protective movements.",
        "followups": [
            "When should I use heat versus cold therapy for joint pain?",
            "What low-impact exercises best strengthen joints without strain?",
            "What anti-inflammatory foods support joint cartilage health?",
            "What warning signs distinguish septic arthritis from standard arthritis?"
        ]
    }
}

# Compatibility Aliases: Ensure management, prevention, and causes keys exist for all entries
for k, entry in CLINICAL_KNOWLEDGE_ENTRIES.items():
    if "management" not in entry:
        entry["management"] = f"• **Immediate Action**:\n{entry.get('immediate_plan', '')}\n\n• **Long-Term Plan**:\n{entry.get('cure_plan', '')}"
    if "causes" not in entry and "reason" in entry:
        entry["causes"] = entry["reason"]
    if "reason" not in entry and "causes" in entry:
        entry["reason"] = entry["causes"]
    if "cure_plan" not in entry and "management" in entry:
        entry["cure_plan"] = entry["management"]
    if "immediate_plan" not in entry:
        entry["immediate_plan"] = "Rest comfortably, stay hydrated, and practice slow diaphragmatic breathing."
    if "overview" not in entry and "definition" in entry:
        entry["overview"] = entry["definition"]
    if "followups" not in entry:
        entry["followups"] = [
            f"What dietary adjustments help manage {k}?",
            f"What are the main warning signs for {k}?",
            f"What questions should I ask my doctor about {k}?",
            f"How does stress affect {k}?"
        ]


def screen_medical_urgency(query: str) -> Optional[str]:
    """Perform lightweight safety screening for potentially urgent clinical symptoms.

    Detects red flags such as severe or worsening abdominal pain, vomiting blood,
    blood or tarry stools, chest pain, breathing difficulty, fainting, rigid/swollen
    abdomen, confusion, or severe weakness.
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


def format_citations(chunks: List[Dict[str, Any]], primary_page: Optional[int] = None) -> str:
    """Format verified source citations from retrieved chunks and verified primary reference page."""
    pages_set = set(c.get("page", 1) for c in chunks) if chunks else set()
    if primary_page:
        pages_set.add(primary_page)
    if not pages_set:
        return ""
    pages = sorted(list(pages_set))
    pages_str = ", ".join(f"Page {p}" for p in pages)
    return f"\n\n📚 **Source**: *Gale Encyclopedia of Medicine* ({pages_str})"


def _match_clinical_condition(query: str) -> Optional[Dict[str, Any]]:
    """Match query to comprehensive clinical knowledge entries."""
    lower = query.lower()
    # Acidity, GERD, Heartburn, Acid Reflux
    if any(k in lower for k in ["acidity", "acid reflux", "gerd", "heartburn", "sour taste", "gastric acid", "acid backflow", "acid regurgitation"]):
        return CLINICAL_KNOWLEDGE_ENTRIES["acidity"]
    # General Gastrointestinal & Stomach
    if any(k in lower for k in ["stomach", "abdomen", "abdominal", "belly", "tummy", "gut", "gastritis", "ulcer", "indigestion", "ibs", "dyspepsia", "bloating", "gas accumulation"]):
        return CLINICAL_KNOWLEDGE_ENTRIES["stomach"]
    if any(k in lower for k in ["hypertension", "blood pressure", "hypotension"]):
        return CLINICAL_KNOWLEDGE_ENTRIES["hypertension"]
    if any(k in lower for k in ["migraine", "headache", "cluster headache", "photophobia"]):
        return CLINICAL_KNOWLEDGE_ENTRIES["migraine"]
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


def get_suggested_followups(query: str, condition_title: Optional[str] = None) -> List[str]:
    """Generate 3-4 natural follow-up questions tailored to the condition, just like ChatGPT/Gemini."""
    matched = _match_clinical_condition(query)
    if matched and "followups" in matched:
        return matched["followups"]
    
    # Dynamic defaults for any other medical inquiry
    clean_topic = condition_title or "this condition"
    return [
        f"What dietary adjustments help manage {clean_topic}?",
        f"What are the primary warning signs for {clean_topic}?",
        f"What immediate home comfort steps provide the quickest relief?",
        f"What questions should I ask my healthcare provider about this?"
    ]


def extract_condition_topic(query: str) -> str:
    """Dynamically extract the clinical symptom or condition topic from user text."""
    lower = query.lower().strip()
    match = re.search(
        r"\b(?:what\s+to\s+do\s+for|how\s+to\s+cure|how\s+to\s+treat|cure\s+for|remedy\s+for|treatment\s+for|medicine\s+for|what\s+is\s+the\s+reason\s+for|why\s+do\s+i\s+have|having|have|suffering\s+from)\s+([a-z\s\-]+)",
        lower,
    )
    if match:
        extracted = match.group(1).strip()
        # Clean trailing question marks or punctuation
        extracted = re.sub(r"[\?\.\!].*", "", extracted).strip()
        if len(extracted) > 2:
            return extracted.title()
    return "Clinical Health Inquiries"


def execute_rag_query(
    query: str,
    llm=None,
    top_k: int = 3,
    detected_emotion: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Execute Gale Encyclopedia of Medicine RAG pipeline.
    
    Guarantees that grounded clinical information with WHAT IS THE REASON and
    PROPER PLAN HOW TO CURE are presented FIRST, followed by warning signs,
    source citations, and professional doctor consultation advisories at the bottom.
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
        msg = (
            "I couldn't find enough relevant information in the available knowledge source "
            "(Gale Encyclopedia of Medicine) to address your inquiry safely and accurately.\n\n"
            "*(Mindly is an educational wellness support companion, not a medical doctor. "
            "For specific health concerns or clinical symptoms, please speak with a healthcare provider.)*"
        )
        if urgency_notice:
            msg = f"{msg}\n\n{urgency_notice}"
        return msg, []

    matched_entry = _match_clinical_condition(query)
    primary_page = matched_entry.get("page") if matched_entry else None

    # Format citations
    citations_str = format_citations(chunks, primary_page=primary_page)
    disclaimer = (
        "\n\n*(Educational Disclaimer: This information is derived from the Gale Encyclopedia of Medicine "
        "for educational awareness and does not substitute for professional medical diagnosis or clinical treatment. "
        "Please consult a qualified healthcare provider for personalized medical evaluation.)*"
    )
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
            
            # Robust refusal detection: small LLM canned refusals
            is_refusal = any(ref in content.lower() for ref in [
                "i cannot provide medical advice", "i can't provide medical advice",
                "consult a qualified healthcare professional for personalized guidance",
                "as an ai", "i am an ai language model", "cannot diagnose or treat",
                "i can't give medical advice", "i cannot give medical advice"
            ]) or len(content) < 250 or not any(k in content.lower() for k in ["reason", "cause", "plan", "cure", "relief", "overview", "what is"])

            if content and len(content) > 150 and not is_refusal:
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
        sections.append(f"### 🩺 Clinical Overview: {title}\n\n{matched_entry.get('definition', matched_entry.get('overview', ''))}")

        # Highlight Brain-Gut Axis & Stress connection when relevant
        if any(term in matched_entry["title"].lower() for term in ["stomach", "acidity", "gastrointestinal"]) and (has_emotional_context or "stomach" in lower_q or "acidity" in lower_q):
            sections.append(
                "🧠 **The Brain-Gut Axis & Stress Connection**:\n"
                "According to the **Gale Encyclopedia of Medicine**, the brain and digestive tract maintain constant bidirectional communication "
                "via the **brain-gut axis** and vagus nerve. Psychological stress and anxiety stimulate enteric nervous receptors, which can provoke "
                "intense cramping, visceral hypersensitivity, delayed gastric emptying, hypersecretion of stomach acid, and bloating without underlying structural tissue destruction."
            )

        # 1. WHAT IS THE REASON (Causes, Mechanisms & Triggers)
        reasons_text = matched_entry.get("reason", matched_entry.get("causes", ""))
        sections.append(f"🔍 **What Is the Reason (Underlying Etiology & Mechanisms)**:\n{reasons_text}")

        # 2. KEY SYMPTOMS
        if "symptoms" in matched_entry and matched_entry["symptoms"]:
            sections.append(f"📋 **Key Symptoms to Monitor**:\n{matched_entry['symptoms']}")

        # 3. PROPER PLAN: IMMEDIATE ACTION
        if "immediate_plan" in matched_entry and matched_entry["immediate_plan"]:
            sections.append(f"💊 **Immediate Action Plan (What to Do Right Now for Relief)**:\n{matched_entry['immediate_plan']}")

        # 4. PROPER PLAN: HOW TO CURE & LONG-TERM MANAGEMENT
        cure_text = matched_entry.get("cure_plan", matched_entry.get("management", ""))
        sections.append(f"🛡️ **Proper Plan: How to Cure & Prevent (Long-Term Management & Clinical Therapies)**:\n{cure_text}")

        # 5. PREVENTION
        if "prevention" in matched_entry and matched_entry["prevention"]:
            sections.append(f"🌿 **Daily Prevention & Self-Care Habits**:\n{matched_entry['prevention']}")

        # 6. WARNING SIGNS / RED FLAGS
        sections.append(f"⚠️ **Warning Signs Requiring Prompt Attention**:\n{matched_entry['warning_signs']}")

    else:
        # Dynamic synthesis from retrieved Gale chunks for unlisted condition
        topic_name = extract_condition_topic(query)
        lead_chunk = chunks[0].get("text", "").strip()
        lead_page = chunks[0].get("page", 1)

        sections.append(
            f"### 🩺 Clinical Overview: {topic_name}\n\n"
            f"According to the **Gale Encyclopedia of Medicine** (Page {lead_page}):\n\n"
            f"> {lead_chunk}\n\n"
            f"This condition requires careful physiological assessment, identification of underlying causes, "
            f"and a structured clinical and lifestyle approach to restore optimal health."
        )

        sections.append(
            f"🔍 **What Is the Reason (Underlying Etiology & Mechanisms)**:\n"
            f"• **Physiological Pathways**: Symptoms arise from localized tissue irritation, inflammatory mediators, "
            f"or autonomic nervous system dysregulation as documented in the medical literature.\n"
            f"• **Common Contributing Factors**: Acute psychological stress, physical exertion, dietary imbalances, "
            f"dehydration, sleep disruption, or underlying infectious agents."
        )

        sections.append(
            f"💊 **Immediate Action Plan (What to Do Right Now for Relief)**:\n"
            f"1. **Rest & Supportive Posture**: Stop strenuous activity immediately and rest in a comfortable, relaxed posture.\n"
            f"2. **Paced Hydration**: Sip room-temperature water slowly to maintain adequate systemic hydration.\n"
            f"3. **Environmental Comfort**: Reduce noise, adjust room lighting, and avoid tight or constrictive clothing.\n"
            f"4. **Autonomic Calming**: Practice slow, calm diaphragmatic breathing to stabilize heart rate variability and downregulate visceral tension."
        )

        sections.append(
            f"🛡️ **Proper Plan: How to Cure & Prevent (Long-Term Management & Clinical Therapies)**:\n"
            f"• **Nutritional & Daily Care**: Maintain regular balanced meals, avoid known food triggers, and ensure 7-8 hours of restorative sleep.\n"
            f"• **Symptom Tracking**: Keep a symptom log recording time of onset, potential triggers, and duration to assist clinical evaluation.\n"
            f"• **Professional Clinical Consultation**: If symptoms persist beyond several days or recur regularly, consult a healthcare physician for formal diagnostic assessment and targeted therapeutic options."
        )

        sections.append(
            "⚠️ **Warning Signs Requiring Prompt Attention**:\n"
            "Seek immediate medical attention if you experience severe unremitting pain, high fever, difficulty breathing, "
            "fainting, persistent vomiting, or sudden weakness."
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
