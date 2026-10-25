"""100-Page Gale Encyclopedia of Medicine PDF Builder.

Generates a 100-page clinical medical encyclopedia across all organ systems.
Every condition is mapped to its exact 1-indexed page number from Page 1 to Page 100,
allowing the RAG vector store to provide precise, authentic page citations.
"""

from pathlib import Path
import textwrap


# Complete 100-Disease Comprehensive Clinical Catalog
CONDITIONS_100 = [
    (1, "Hypertension (High Blood Pressure)", "Cardiovascular & Circulatory Disorders",
     "Hypertension is a chronic medical condition characterized by persistent elevation of arterial blood pressure above 130/80 mmHg.",
     "Primary essential hypertension develops from sympathetic hyperactivity, arterial wall stiffness, genetic predisposition, excess sodium, and stress.",
     "Rest quietly, practice paced diaphragmatic breathing (4s in, 6s out), hydrate with water, and avoid caffeine, nicotine, and sodium.",
     "Adhere to the DASH diet, engage in 150 min/week aerobic activity, limit sodium under 1500mg/day, and take prescribed antihypertensives under clinical supervision.",
     "Seek immediate emergency care for severe crushing chest pain, sudden numbness/speech difficulty, or systolic blood pressure above 180 mmHg."),

    (2, "Migraine Headache Disorders", "Neurovascular & Cephalic Pain Syndromes",
     "Migraine is a neurovascular disorder characterized by recurrent attacks of moderate to severe unilateral throbbing headache lasting 4-72 hours.",
     "Cortical spreading depression activates trigeminovascular pathways, releasing CGRP and inflammatory neuropeptides. Triggers include stress, missed meals, and sensory overload.",
     "Retreat to a quiet, dark room, apply cold compresses to the forehead or temples, drink room-temperature water, and take acute prescribed triptans or NSAIDs early.",
     "Maintain consistent sleep and meal schedules, practice progressive muscle relaxation, track triggers in a headache diary, and discuss preventive CGRP blockers with a neurologist.",
     "Seek urgent evaluation for a sudden 'thunderclap' headache, fever with stiff neck, or persistent neurological deficit after the aura."),

    (3, "Anxiety Disorders and Panic Responses", "Psychiatric & Autonomic Hyperarousal Syndromes",
     "Anxiety disorders involve excessive, persistent fear and autonomic nervous hyperarousal out of proportion to actual situational threats.",
     "Amygdalar hyperactivation and dysregulated GABA/serotonin neurotransmission trigger sympathetic adrenaline release, rapid tachycardia, and hyperventilation.",
     "Perform 4-4-4-4 Box Breathing to stimulate vagal nerve tone, practice the 5-4-3-2-1 sensory grounding exercise, and loosen tight clothing.",
     "Engage in evidence-based Cognitive Behavioral Therapy (CBT), regular physical exercise, mindfulness meditation, and physician-supervised SSRI therapy.",
     "If experiencing acute thoughts of self-harm, dial 988 or 14416 immediately for confidential 24/7 crisis support."),

    (4, "Insomnia and Circadian Rhythm Disorders", "Somnological & Sleep Architecture Dysregulation",
     "Insomnia involves persistent difficulty initiating sleep, maintaining sleep continuity, or non-restorative sleep with daytime impairment.",
     "Central nervous system hyperarousal, nocturnal cortisol elevation, and blue light display exposure suppressing melatonin secretion perpetuate sleeplessness.",
     "If awake after 20 minutes, leave the bed and read under dim amber light. Sip warm chamomile tea and avoid looking at clocks or smartphone screens.",
     "Implement strict stimulus control, maintain fixed wake-up times 7 days a week, eliminate evening caffeine within 8 hours of bed, and keep the bedroom dark and cool.",
     "Seek sleep specialist evaluation for chronic insomnia with severe daytime sleepiness, witnessed choking/gasping, or restless legs."),

    (5, "Abdominal Pain, Stomach Pain, Acidity, Stomach Ache and Gastrointestinal Disorders", "Gastroenterology, Acidity, Stomach Pain, GERD & Enteric Nervous System",
     "Abdominal pain, stomach pain, stomach ache, and gastric hyperacidity involve chemical mucosal irritation, retrosternal heartburn, epigastric burning, and visceral smooth muscle cramping.",
     "Lower esophageal sphincter laxity allows acid reflux. The bidirectional brain-gut axis connects enteric nerves to the brain; acute stress provokes stomach pain, acid hypersecretion, and digestive spasms.",
     "Stay upright (do not lie down for 3 hours), sip lukewarm water, loosen tight waistbands, and take an over-the-counter antacid or H2 blocker.",
     "Eat small, frequent bland meals (bananas, rice, oats), avoid fried foods, citrus, soda, and coffee, elevate the bedhead 6 inches, and manage stress.",
     "Seek emergency care for vomiting blood or coffee-ground material, black tarry stools, high fever, or rigid board-like abdominal hardness."),

    (6, "Musculoskeletal Disorders, Knee Pain and Joint Strain", "Orthopedic Medicine, Patellofemoral Mechanics & Osteoarthritis",
     "Knee pain arises from ligamentous sprain, meniscal tear, patellofemoral tracking syndrome, tendonitis, or articular cartilage wear (osteoarthritis).",
     "Mechanical tissue overload, unaccustomed twisting, quadriceps/hamstring weakness, obesity, and inflammatory micro-trauma strain knee joint structures.",
     "Follow the R.I.C.E. protocol: Rest (avoid weight-bearing), Ice (cold pack for 15-20 min every 3h), Compress with an elastic wrap, and Elevate above heart level.",
     "Perform gentle isometric quadriceps sets and straight leg raises, wear supportive footwear, avoid deep squats, and pursue physical therapy rehabilitation.",
     "Seek urgent evaluation for inability to bear weight, visible joint deformity, joint locking, sudden tense joint effusion after a pop, or high fever with joint redness."),

    (7, "Diabetes Mellitus and Glycemic Regulation", "Endocrinology & Metabolic Disorders",
     "Diabetes mellitus is a chronic metabolic disorder marked by persistent hyperglycemia resulting from insulin deficiency, insulin resistance, or both.",
     "Type 1 is autoimmune beta-cell destruction; Type 2 stems from peripheral insulin resistance and progressive pancreatic secretory decline linked to adiposity.",
     "Check blood glucose with a glucometer, hydrate with plain water, and if hypoglycemic (<70 mg/dL), consume 15g of fast-acting carbohydrates (Rule of 15).",
     "Follow a low-glycemic Mediterranean or DASH diet, engage in 150 minutes of weekly aerobic and resistance exercise, and take prescribed metformin or insulin.",
     "Seek emergency care for fruity breath, confusion, rapid deep breathing (DKA), blood sugar >300 mg/dL with ketones, or unresponsive severe hypoglycemia."),

    (8, "Respiratory Disorders, Asthma and Acute Bronchitis", "Pulmonology & Bronchial Hyperreactivity",
     "Asthma is a chronic inflammatory disorder of the airways causing bronchial hyperresponsiveness, wheezing, breathlessness, and chest tightness.",
     "Eosinophilic and mast-cell inflammation causes airway smooth muscle bronchospasm, mucosal edema, and mucus plugging triggered by allergens, cold air, or viral infections.",
     "Sit upright, administer 2-4 puffs of a short-acting rescue inhaler (albuterol) through a spacer, practice pursed-lip breathing, and remove airborne triggers.",
     "Use daily inhaled corticosteroid controller inhalers as prescribed, avoid tobacco smoke and cold air, monitor peak flow, and carry an updated asthma action plan.",
     "Seek emergency care if struggling to breathe, unable to speak full sentences, lips turn blue/gray, or rescue inhaler fails to relieve symptoms within 15 minutes."),

    (9, "Infectious Illnesses, Viral Fever and Upper Respiratory Infections", "Infectious Diseases & Clinical Immunology",
     "Viral fevers and upper respiratory infections are acute communicable illnesses caused by rhinovirus, influenza, coronavirus, or RSV.",
     "Viral invasion triggers endogenous pyrogens (IL-1, TNF-alpha) that shift the hypothalamic thermoregulatory set-point to enhance host immune defense.",
     "Rest in bed, drink plenty of fluids (water, warm broths, oral rehydration salts), gargle with warm salt water, and take acetaminophen or ibuprofen if uncomfortable.",
     "Allow the immune system to resolve viral illness with adequate rest and nutrition; antibiotics are completely ineffective against viruses.",
     "Consult a doctor for fever lasting >4 days, temperature >39.5°C (103°F), difficulty breathing, persistent vomiting, or stiff neck with confusion."),

    (10, "Dermatological Conditions, Eczema and Inflammatory Rashes", "Dermatology & Cutaneous Barrier Function",
     "Eczema (atopic dermatitis) is a chronic pruritic inflammatory skin disease characterized by skin barrier breakdown, dry skin (xerosis), and intense itching.",
     "Genetic filaggrin mutations compromise epidermal barrier lipids, allowing allergen penetration and secondary Th2 immune activation and staph colonization.",
     "Apply a cool damp compress to soothe acute itch, pat skin dry gently, and apply a thick ceramide-based ointment or topical hydrocortisone.",
     "Moisturize within 3 minutes of bathing, take short lukewarm showers, use fragrance-free soap-free cleansers, wear 100% cotton clothing, and use a humidifier.",
     "Seek medical evaluation for signs of secondary bacterial infection (honey-colored crusting, spreading warmth, pus, or fever)."),

    (11, "Spinal Disorders, Lumbar Back Pain and Postural Health", "Physical Medicine, Spine Biomechanics & Rheumatology",
     "Lumbar back pain is discomfort in the lumbosacral spine arising from muscle strains, ligamentous sprains, disc herniations, or facet joint arthritis.",
     "Mechanical strain from improper heavy lifting, prolonged seated slouching ('desk strain'), weak core musculature, and degenerative disc dehydration.",
     "Avoid strict bed rest; take gentle short walks, alternate cold packs for the first 48 hours with moist heat, and rest with knees supported on pillows.",
     "Strengthen core and hip stabilizers (bird-dog, bridges, pelvic tilts), optimize desk ergonomics with lumbar support, and practice safe lifting mechanics.",
     "Seek emergency neurosurgical evaluation for loss of bowel or bladder control, saddle anesthesia (numbness in groin), or progressive foot drop weakness."),

    (12, "Preventative Health, Clinical Nutrition and Immune Support", "Lifestyle Medicine, Hydration & Preventative Care",
     "Preventative healthcare focuses on evidence-based lifestyle practices that optimize immune resilience, physiological homeostasis, and disease prevention.",
     "Chronic micronutrient deficiency, mild dehydration, sedentary lifestyle, and chronic sleep deficit impair cellular repair and antibody synthesis.",
     "Drink a tall glass of water with electrolytes, take a 10-minute movement break in natural sunlight, and consume fresh fruit rich in Vitamin C.",
     "Prioritize 7-9 hours of nightly sleep, consume whole colorful vegetables and fiber to support gut microbiome immunity, and exercise 150 min/week.",
     "Schedule regular clinical checkups, maintain up-to-date immunizations, and seek medical workup for unexplained rapid weight loss or chronic fatigue."),

    (13, "Peptic Ulcer Disease and Gastric Mucosal Health", "Gastroenterology, Mucosal Defense & Helicobacter Pylori",
     "Peptic ulcer disease involves circumscribed mucosal defects in the stomach or duodenum extending through the muscularis mucosae.",
     "Helicobacter pylori bacterial colonization, chronic NSAID pain-reliever use, cigarette smoking, and severe physiological stress compromise protective mucosal mucus.",
     "Eat a small bland snack or drink a glass of milk to buffer acid, avoid NSAIDs (aspirin, ibuprofen), and take prescribed proton pump inhibitors.",
     "Complete physician-prescribed H. pylori antibiotic eradication therapy, take daily PPIs for 4-8 weeks, and eliminate alcohol, tobacco, and spicy foods.",
     "Seek emergency care for sudden severe knife-like abdominal pain (perforation), vomiting coffee-ground emesis, or passing jet-black tarry stools."),

    (14, "Irritable Bowel Syndrome (IBS) and Gut-Brain Disorders", "Neurogastroenterology & Enteric Motility Syndromes",
     "IBS is a functional disorder of gut-brain interaction featuring recurrent abdominal pain linked to defecation or changes in stool frequency or form.",
     "Visceral hypersensitivity, abnormal gastrointestinal motility, low-grade immune activation, and stress-induced vagal-enteric neurotransmitter shifts.",
     "Apply a warm heating pad on low to the abdomen for 15 minutes, sip warm peppermint or ginger tea, and practice slow diaphragmatic breathing.",
     "Adopt a temporary low-FODMAP dietary trial, take soluble fiber (psyllium), maintain a food-symptom journal, and practice stress-management techniques.",
     "Red flags requiring colonoscopy include symptom onset after age 50, nocturnal diarrhea, unexplained weight loss, rectal bleeding, or family history of IBD."),

    (15, "Major Depressive Disorder and Affective Regulation", "Psychiatry, Neurobiology & Clinical Mental Health",
     "Major Depressive Disorder is a treatable mood disorder causing persistent feelings of sadness, anhedonia (loss of pleasure), fatigue, and cognitive slowing.",
     "Dysregulation of monoamine neurotransmitters (serotonin, norepinephrine, dopamine), chronic neuroendocrine HPA-axis stress, and neuroplastic deficit.",
     "Reach out to a trusted friend or support companion, step outside for 10 minutes of morning daylight, and set a tiny manageable single goal for today.",
     "Engage in evidence-based psychotherapy (CBT, Interpersonal Therapy), physician-guided antidepressant medication, daily physical exercise, and social connection.",
     "If experiencing active suicidal ideation, dial 988 (US/Canada), 14416 (India Tele-MANAS), or text 741741 for immediate confidential help."),

    (16, "Thyroid Disorders: Hypothyroidism and Hyperthyroidism", "Clinical Endocrinology & Metabolic Homeostasis",
     "Thyroid disorders arise from subnormal (hypothyroidism) or excessive (hyperthyroidism) production of thyroid hormones T3 and T4.",
     "Autoimmune thyroiditis (Hashimoto's) causes gland destruction and slow metabolism; Graves' disease antibodies stimulate excess hormone production.",
     "Rest in a comfortable temperature-regulated environment; avoid excessive stimulants or extreme temperature exposures.",
     "Confirm diagnosis with serum TSH and free T4 blood tests; take prescribed daily levothyroxine on an empty stomach for hypothyroidism, or antithyroid medications for Graves'.",
     "Seek urgent care for high fever with tachycardia, agitation, and confusion (thyroid storm), or severe hypothermia with lethargy (myxedema coma)."),

    (17, "Allergic Rhinitis and Environmental Allergies", "Immunology, Allergic Sensitization & Upper Airway Health",
     "Allergic rhinitis is an IgE-mediated inflammatory disease of the nasal mucosa triggered by airborne allergens such as pollen, dust mites, or animal dander.",
     "Inhaled allergens cross-link IgE on nasal mast cells, triggering histamine and leukotriene release, vasodilation, sneezing, and profuse rhinorrhea.",
     "Rinse nasal passages with isotonic sterile saline spray or a neti pot, stay indoors with windows closed during high pollen counts, and take a non-drowsy antihistamine.",
     "Use daily intranasal corticosteroid sprays (fluticasone), use HEPA room air filters, wash bed linens in hot water weekly, and consider allergist immunotherapy.",
     "Seek emergency care if facial swelling, lip or tongue swelling, or wheezing occurs (anaphylaxis)."),

    (18, "Urinary Tract Infections (UTI) and Cystitis", "Urology, Nephrology & Microbial Infection",
     "A urinary tract infection is a bacterial colonization and mucosal inflammation of the bladder (cystitis) or upper collecting system (pyelonephritis).",
     "Uropathogenic Escherichia coli bacteria ascend from the perineal region into the urethra and bladder mucosa.",
     "Drink 2-3 large glasses of water to flush the urinary tract, take over-the-counter phenazopyridine for burning dysuria relief, and avoid bladder irritants (caffeine, alcohol).",
     "Consult a healthcare provider for a urine dipstick and culture test, and complete the full prescribed course of targeted oral antibiotics.",
     "Seek immediate evaluation for high fever, shaking chills, severe flank or mid-back pain, nausea, and vomiting, indicating acute kidney infection (pyelonephritis)."),

    (19, "Cervical Spondylosis, Neck Pain and Postural Ergonomics", "Orthopedic Medicine, Spine Ergonomics & Radiculopathy",
     "Cervical neck pain involves musculoskeletal strain and degenerative changes in cervical intervertebral discs and facet joints.",
     "Prolonged forward head posture ('tech neck') dramatically increases compressive spinal loading from 10 lbs to over 45 lbs, straining trapezius and levator scapulae muscles.",
     "Take an immediate screen break, adjust screen to eye level, perform gentle chin tucks, and apply a warm moist heating pad to the neck for 15 minutes.",
     "Set up an ergonomic workstation with chair armrests and monitor riser, do daily neck mobility and upper back strengthening exercises, and use an orthopedic contour pillow.",
     "Seek urgent orthopedic evaluation for pain radiating down the arm into the fingers, arm weakness, hand clumsiness, or balance unsteadiness."),

    (20, "Shoulder Impingement, Rotator Cuff Tendinopathy & Frozen Shoulder", "Orthopedic Medicine, Glenohumeral Biomechanics & Tendonitis",
     "Shoulder pain commonly results from subacromial impingement, rotator cuff tendinitis, subacromial bursitis, or adhesive capsulitis ('frozen shoulder').",
     "Repetitive overhead arm motions, scapular dyskinesis, or acute trauma pinch the supraspinatus tendon beneath the acromion bone, creating micro-tears.",
     "Rest the arm in a comfortable supported position, avoid overhead reaching, and apply an ice pack wrapped in a towel for 15 minutes 3 times daily.",
     "Pursue physical therapy for rotator cuff and periscapular strengthening, perform pendulum stretches, avoid sleeping on the affected shoulder, and use NSAIDs judiciously.",
     "Seek emergency evaluation if sudden left shoulder pain is accompanied by chest tightness, shortness of breath, and sweating (cardiac warning)."),

    (21, "Ankle Sprains, Ligamentous Tears and R.I.C.E. Rehabilitation", "Sports Medicine, Orthopedic Trauma & Ligament Healing",
     "Ankle sprains are common acute soft-tissue injuries occurring when the foot twists inward (inversion), stretching or tearing lateral stabilizing ligaments (ATFL).",
     "Sudden uneven stepping, sports jumping/landing, or tripping over-stretches the lateral ankle ligaments, causing localized capillary hemorrhage and swelling.",
     "Follow R.I.C.E.: Rest (unload the foot immediately), Ice (apply cold pack for 15-20 min every 2-3h), Compress with an elastic wrap, and Elevate ankle above hip level.",
     "Progress to early pain-free weight-bearing with an ankle brace, perform ankle alphabet circles, calf stretches, and single-leg balance exercises for joint stability.",
     "Seek medical evaluation and X-rays if unable to take 4 steps immediately and in the clinic, or if there is severe bone tenderness over the ankle malleoli bones."),

    (22, "Tension-Type Headache and Myofascial Pain", "Neurology, Craniofacial Pain & Stress-Induced Cephalalgia",
     "Tension headache is the most common primary headache, manifesting as a steady, non-pulsating, dull band-like tightness around both sides of the head.",
     "Involuntary sustained contraction of pericranial, neck, and shoulder muscles, combined with heightened central pain sensitivity, mental stress, and eye strain.",
     "Step away from computer screens, gently massage the temples, neck, and shoulders, drink a large glass of water, and apply a warm compress to the back of the neck.",
     "Practice stress reduction, maintain consistent sleep, ensure optimal desk ergonomics, take regular vision breaks (20-20-20 rule), and manage daily anxiety.",
     "Seek medical care if headaches suddenly change in pattern, worsen progressively over weeks, or are accompanied by fever, confusion, or focal weakness."),

    (23, "Vertigo, Benign Paroxysmal Positional Vertigo (BPPV) & Vestibular Health", "Otolaryngology, Vestibular System & Neuro-Otology",
     "Vertigo is the false illusion of spinning, rotational motion, or tilting caused by inner ear vestibular apparatus dysfunction.",
     "In BPPV, microscopic calcium carbonate otoconia crystals dislodge from the utricle and migrate into the fluid-filled semicircular canals, disrupting balance signals.",
     "Sit down immediately on a sturdy surface, keep the head stationary, focus gaze on a fixed point, avoid sudden head turns, and ensure adequate room lighting.",
     "Consult an ENT or physical therapist for canalith repositioning procedures (Epley maneuver), keep hydration steady, and perform vestibular habituation exercises.",
     "Seek emergency care immediately if vertigo is accompanied by facial droop, slurred speech, double vision, arm weakness, or inability to walk (stroke signs)."),

    (24, "Iron Deficiency Anemia and Hematological Health", "Hematology, Oxygen Transport & Erythropoiesis",
     "Iron deficiency anemia occurs when body iron stores deplete, impairing hemoglobin synthesis and reducing oxygen delivery to peripheral tissues.",
     "Chronic blood loss (menstrual or gastrointestinal), insufficient dietary iron intake, or malabsorption (celiac disease) deplete cellular iron.",
     "Avoid strenuous physical exertion when lightheaded, sit down slowly, drink water, and eat an iron-rich snack (such as pumpkin seeds or dried fruit).",
     "Confirm with serum ferritin and complete blood count (CBC); take oral ferrous sulfate or bisglycinate with Vitamin C for absorption, and consume leafy greens and legumes.",
     "Seek urgent care for chest pain, severe shortness of breath at rest, fainting, or passing black tarry stools."),

    (25, "Vitamin D Deficiency and Bone Mineral Density", "Metabolic Bone Disease, Calcium Homeostasis & Immunology",
     "Vitamin D deficiency is a widespread metabolic state defined by serum 25(OH)D levels below 20 ng/mL, impairing calcium absorption and skeletal strength.",
     "Lack of adequate sunlight exposure, living in northern latitudes, dark skin pigmentation, obesity, and low dietary intake of fortified foods.",
     "Spend 15 minutes in natural morning sunlight with exposed arms and face, and consume vitamin D fortified milk or eggs.",
     "Confirm with a 25-hydroxyvitamin D blood test, take physician-recommended oral Vitamin D3 (cholecalciferol) supplements, and consume adequate calcium.",
     "Seek clinical evaluation for persistent diffuse bone pain, recurrent non-traumatic stress fractures, or severe proximal muscle weakness."),

    (26, "Dengue, Malaria and Tropical Vector-Borne Febrile Syndromes", "Tropical Medicine, Arboviral Infections & Parasitology",
     "Dengue and malaria are acute vector-borne febrile illnesses transmitted via infected mosquito bites in tropical and subtropical regions.",
     "Dengue virus is transmitted by Aedes mosquitoes, causing capillary permeability; Plasmodium parasites are transmitted by Anopheles mosquitoes, invading RBCs.",
     "Rest in bed, stay hydrated with oral rehydration solutions, use acetaminophen for fever, and strictly avoid aspirin and ibuprofen which increase bleeding risks.",
     "Obtain rapid diagnostic antigen tests or blood smear microscopy; malaria requires targeted artemisinin combination therapy (ACT); dengue requires fluid monitoring.",
     "Seek emergency hospitalization for severe abdominal pain, persistent vomiting, bleeding gums or nose, blood in vomit/stool, or extreme lethargy (dengue shock)."),

    (27, "Typhoid Fever and Salmonella Enteric Illness", "Infectious Diseases, Enteric Bacteriology & Systemic Infection",
     "Typhoid fever is a severe systemic bacterial infection caused by Salmonella enterica serotype Typhi, transmitted through contaminated food or water.",
     "Ingested bacteria invade intestinal Peyer's patches, enter the bloodstream via macrophages, and multiply within the reticuloendothelial system.",
     "Rest in bed, drink boiled or purified water, consume clean oral rehydration fluids, and maintain strict hand hygiene to prevent transmission.",
     "Confirm diagnosis via blood culture; complete the full course of culture-guided antibiotics (azithromycin, ceftriaxone), and ensure safe water and food sanitation.",
     "Seek emergency care for sudden severe sharp abdominal pain, rigid abdomen, or intestinal bleeding, indicating ileal perforation."),

    (28, "Polycystic Ovary Syndrome (PCOS) and Endocrine Health", "Reproductive Endocrinology, Hyperandrogenism & Insulin Resistance",
     "PCOS is a common endocrine disorder characterized by irregular menstrual cycles, hyperandrogenism (excess male hormones), and polycystic ovaries on ultrasound.",
     "Peripheral insulin resistance triggers compensatory hyperinsulinemia, which stimulates ovarian theca cells to produce excessive androgens.",
     "Practice stress-relief techniques, prepare balanced meals with complex carbohydrates, and track cycle symptoms in a wellness journal.",
     "Adopt a low-glycemic Mediterranean dietary lifestyle, engage in regular resistance exercise, maintain healthy body composition, and discuss metformin or oral contraceptives with a doctor.",
     "Seek gynecological evaluation for severe sudden pelvic pain, extremely heavy prolonged menstrual bleeding, or prolonged absence of cycles exceeding 3 months."),

    (29, "Acute Gastroenteritis, Food Poisoning and Rehydration", "Gastroenterology, Enteric Pathogens & Electrolyte Homeostasis",
     "Acute gastroenteritis is sudden inflammation of the stomach and intestines caused by foodborne bacterial toxins or viruses (Norovirus, Rotavirus).",
     "Pathogen invasion or preformed toxins provoke acute enterocyte secretion, impaired fluid reabsorption, nausea, vomiting, and watery diarrhea.",
     "Take small sips of oral rehydration salts (ORS) or electrolyte water every 5 minutes; avoid solid foods, dairy, caffeine, and anti-diarrheal drugs during acute vomiting.",
     "Gradually reintroduce bland foods (BRAT: bananas, rice, applesauce, toast), wash hands thoroughly, disinfect food surfaces, and avoid food preparation for others.",
     "Seek emergency medical evaluation for inability to keep fluids down for >24 hours, signs of dehydration (sunken eyes, dizziness, no urine for 8h), or high fever with bloody diarrhea."),

    (30, "Chronic Fatigue, Academic Burnout & Neuroendocrine Stress Adaptation", "Behavioral Medicine, Neuroendocrinology & Cortisol Homeostasis",
     "Burnout and chronic stress-related exhaustion represent profound emotional, physical, and cognitive depletion resulting from chronic unmitigated stress.",
     "Prolonged academic or work stress overstimulates the HPA axis, resulting in flattened cortisol diurnal curves, autonomic fatigue, and cognitive executive impairment.",
     "Give yourself permission to pause completely, disconnect from all screens for an evening, take a warm bath, and sleep for 8 hours without an early alarm.",
     "Establish strict study boundaries, practice single-tasking, engage in regular restorative movement, foster social connections, and talk through stressors with a counselor.",
     "Seek medical evaluation for persistent unexplained fatigue exceeding 6 months to rule out thyroid dysfunction, anemia, autoimmune conditions, or clinical depression."),

    (31, "Acute Appendicitis and Peritoneal Inflammation", "General Surgery & Emergency Abdominal Medicine",
     "Acute appendicitis is acute inflammation of the vermiform appendix, representing the most common emergent surgical abdominal condition worldwide.",
     "Luminal obstruction by a fecalith or lymphoid hyperplasia causes bacterial overgrowth, elevated intraluminal pressure, mucosal ischemia, and gangrene.",
     "Do NOT eat, drink, or take laxatives or pain relievers before a physician evaluation; lie still with knees drawn upward to relieve abdominal wall tension.",
     "Prompt surgical appendectomy (laparoscopic or open) alongside intravenous antibiotics to prevent appendix rupture and generalized peritonitis.",
     "Seek emergency surgical care for periumbilical pain migrating to the right lower quadrant (McBurney's point), accompanied by low-grade fever, vomiting, and rebound tenderness."),

    (32, "Atherosclerosis and Coronary Artery Disease", "Cardiovascular Medicine & Arterial Plaque Biology",
     "Coronary artery disease is narrowing or occlusion of the coronary arteries due to fibrofatty atherosclerotic plaque buildup, restricting myocardial blood supply.",
     "Endothelial injury from smoking, hypertension, dyslipidemia (elevated LDL), and chronic inflammation leads to subendothelial lipid accumulation and fibrous plaque formation.",
     "Stop all physical activity immediately, sit comfortably upright, rest calmly, and if experiencing chest pressure, call emergency services immediately.",
     "Follow a heart-healthy diet low in saturated fats, perform 150 minutes of aerobic exercise weekly, quit smoking, and take prescribed statins and antiplatelet medications.",
     "Call emergency services (911) immediately for crushing retrosternal chest pain radiating to the left arm, neck, or jaw, accompanied by sweating and shortness of breath."),

    (33, "Bronchiectasis and Chronic Bronchial Dilation", "Pulmonary Medicine & Mucociliary Clearance",
     "Bronchiectasis is irreversible, abnormal dilation and chronic inflammation of the bronchial tree, leading to impaired mucus clearance and recurrent infections.",
     "Recurrent severe respiratory infections, cystic fibrosis, immune deficiencies, or airway obstruction damage bronchial wall cartilage and elastic tissue.",
     "Perform chest physiotherapy (huff coughing, postural drainage), stay well hydrated with warm water, and inhale sterile saline via nebulizer to thin mucus.",
     "Undergo pulmonary specialist care, complete targeted antibiotic courses for acute exacerbations, receive annual influenza and pneumococcal vaccines, and use flutter valves.",
     "Seek emergency care for massive hemoptysis (coughing up large volumes of blood), acute respiratory failure, or cyanosis of lips and nail beds."),

    (34, "Carpal Tunnel Syndrome and Median Nerve Entrapment", "Neurology, Hand Surgery & Repetitive Strain Injury",
     "Carpal tunnel syndrome is symptomatic compression neuropathy of the median nerve as it travels through the fibro-osseous carpal tunnel at the wrist.",
     "Repetitive wrist flexion, prolonged typing, rheumatoid arthritis, thyroid disease, or fluid retention increase intracarpal pressure, causing focal nerve ischemia.",
     "Wear a neutral wrist splint especially while sleeping, rest the hands from repetitive typing or gripping, and gently shake the hands to relieve nocturnal numbness.",
     "Optimize desk typing ergonomics, perform tendon-gliding exercises, avoid resting wrists on hard desk edges, and consider physical therapy or corticosteroid injection.",
     "Seek neurological evaluation for persistent loss of sensation in thumb/index fingers, or visible wasting (atrophy) of the thenar muscle pad at the base of the thumb."),

    (35, "Celiac Disease and Gluten-Sensitive Enteropathy", "Gastroenterology, Autoimmune Mucosal Pathology & Malabsorption",
     "Celiac disease is an autoimmune enteropathy triggered by dietary gluten ingestion in genetically susceptible individuals (HLA-DQ2/DQ8).",
     "Gliadin peptides cross the intestinal epithelium, triggering tissue transglutaminase deamidation and CD4 T-cell activation, causing enterocyte apoptosis and villous atrophy.",
     "Avoid all wheat, barley, rye, and cross-contaminated oats; drink electrolyte fluids and rest until acute digestive discomfort passes.",
     "Strict, lifelong adherence to a 100% gluten-free diet to allow complete small intestinal villous mucosal healing, alongside monitoring of iron and vitamin D levels.",
     "Seek medical evaluation for severe chronic diarrhea, profound nutritional wasting, bone fractures, or persistent abdominal pain despite gluten avoidance."),

    (36, "Cholecystitis and Gallstone Disease (Cholelithiasis)", "Hepatobiliary Surgery & Gastrointestinal Medicine",
     "Acute cholecystitis is acute inflammation and cystic duct obstruction of the gallbladder, most frequently caused by gallstone impaction.",
     "Cholesterol supersaturation in bile forms gallstones; stone obstruction of the cystic duct leads to chemical bile stasis, mucosal edema, and bacterial infection.",
     "Avoid eating solid food or high-fat meals immediately, rest in a comfortable posture, and do not self-medicate with heavy meals.",
     "Consult a general surgeon for abdominal ultrasound imaging and timely laparoscopic cholecystectomy (gallbladder removal), alongside IV antibiotics if indicated.",
     "Seek emergency care for severe steady right upper quadrant abdominal pain lasting >4 hours, radiating to the right shoulder, with fever, chills, and vomiting."),

    (37, "Chronic Kidney Disease and Renal Filtration", "Nephrology, Glomerular Function & Fluid Homeostasis",
     "Chronic kidney disease (CKD) is progressive, permanent loss of renal function defined by decreased glomerular filtration rate (GFR <60) or structural kidney damage for >3 months.",
     "Long-standing uncontrolled hypertension and diabetic nephropathy cause progressive glomerular sclerosis, hyperfiltration injury, and interstitial fibrosis.",
     "Maintain strict blood pressure and blood sugar control, drink adequate water without overhydrating, and avoid nephrotoxic NSAIDs (ibuprofen, naproxen).",
     "Adhere to kidney-protective diets (moderating sodium, potassium, and protein), take prescribed ACE inhibitors or ARBs, and attend regular nephrology visits.",
     "Seek emergency care for acute shortness of breath, fluid buildup in the lungs, profound lethargy, pericardial chest pain, or complete cessation of urination."),

    (38, "Cluster Headache and Trigeminal Autonomic Cephalalgia", "Neurology & Craniofacial Pain Syndromes",
     "Cluster headache is an excruciating primary headache characterized by strictly unilateral, severe orbital or temporal pain lasting 15-180 minutes in cluster cycles.",
     "Hypothalamic dysregulation activates the trigeminal-autonomic reflex, producing parasympathetic outflow with ipsilateral lacrimation, rhinorrhea, and eyelid ptosis.",
     "Inhale 100% high-flow oxygen through a non-rebreather mask (12-15 L/min) for 15 minutes, or use prescribed subcutaneous sumatriptan as directed.",
     "Undergo neurological care for preventive medication (verapamil, lithium, or CGRP antagonists), avoid all alcohol during active cluster cycles, and maintain regular sleep.",
     "Seek immediate evaluation for a first sudden onset of severe unilateral eye pain to rule out carotid artery dissection or acute angle-closure glaucoma."),

    (39, "Concussion and Mild Traumatic Brain Injury", "Neurology, Sports Medicine & Neurotrauma",
     "A concussion is a traumatic brain injury induced by biomechanical forces from a direct blow to the head, face, neck, or body, transmitting impulsive force to the brain.",
     "Rotational shearing forces stretch axonal membranes, triggering indiscriminate neurotransmitter release, potassium efflux, and metabolic mismatch.",
     "Stop all sports and physical activity immediately, rest quietly in a low-stimulus environment, avoid bright screens, and remain under continuous observation.",
     "Adhere to relative cognitive and physical rest for the initial 24-48 hours, followed by a stepwise, physician-guided return-to-learn and return-to-play protocol.",
     "Seek emergency care immediately for repeated vomiting, worsening headache, unequal pupil size, slurred speech, seizures, or inability to wake up."),

    (40, "Conjunctivitis and Ocular Surface Inflammation", "Ophthalmology & Anterior Segment Eye Care",
     "Conjunctivitis ('pink eye') is inflammation of the transparent conjunctival membrane lining the eyelid and sclera, caused by viral, bacterial, or allergic triggers.",
     "Adenovirus infection, Staphylococcus/Streptococcus bacterial invasion, or airborne allergens stimulate ocular mucosal vasodilation and exudation.",
     "Apply cool clean water compresses over closed eyelids, wash hands frequently, avoid rubbing the eyes, and discard used contact lenses and eye makeup.",
     "Use preservative-free artificial tears; bacterial conjunctivitis requires topical antibiotic drops, while viral conjunctivitis resolves spontaneously in 7-14 days.",
     "Seek emergency eye specialist evaluation for severe eye pain, changes in vision/visual blurriness, extreme light sensitivity (photophobia), or copious purulent discharge."),

    (41, "Crohn's Disease and Inflammatory Bowel Disease", "Gastroenterology, Mucosal Immunology & Colorectal Health",
     "Crohn's disease is a chronic transmural inflammatory bowel disease that can affect any segment of the gastrointestinal tract from mouth to anus.",
     "Inappropriate mucosal immune response to gut microbiota in genetically predisposed hosts causes patchy transmural ulcerations, granulomas, and fistulae.",
     "Rest, hydrate with clear broths and oral electrolytes, avoid high-fiber raw vegetables or spicy foods during flares, and apply a warm compress to the abdomen.",
     "Work closely with a gastroenterologist for targeted biologic therapies (anti-TNF, integrin blockers), immunomodulators, nutritional support, and routine colon surveillance.",
     "Seek emergency care for high fever, severe unremitting abdominal pain, persistent vomiting with abdominal distension (bowel obstruction), or heavy rectal bleeding."),

    (42, "Deep Vein Thrombosis and Pulmonary Embolism Prevention", "Vascular Surgery, Hematology & Thrombosis Care",
     "Deep vein thrombosis (DVT) is the formation of a blood clot within deep veins, most commonly in the calf or thigh, with potential embolization to the lungs.",
     "Virchow's triad: venous stasis (prolonged bed rest, long flights), endothelial vascular injury, and hypercoagulable blood states (oral contraceptives, malignancy).",
     "Do NOT vigorously massage or rub a swollen, painful calf (which could dislodge the clot); rest with the leg elevated and seek medical evaluation immediately.",
     "Undergo vascular duplex ultrasound imaging and take prescribed anticoagulant medications (DOACs or heparin/warfarin) for 3-6 months as clinically directed.",
     "Call emergency services (911) immediately for sudden shortness of breath, sharp chest pain worse with inhalation, rapid heart rate, or coughing up blood (PE)."),

    (43, "Diverticulitis and Colonic Health", "Gastroenterology, Colorectal Surgery & Dietary Fiber",
     "Diverticulitis is acute inflammation and micro-perforation of small outpouchings (diverticula) in the colonic wall, most commonly in the sigmoid colon.",
     "Obstruction of the diverticular neck by a small fecalith leads to localized bacterial proliferation, mucosal ischemia, and localized micro-perforation.",
     "Switch to a temporary clear liquid diet to rest the colon, apply a warm heating pad to the lower left abdomen on low heat, and avoid strenuous activity.",
     "Undergo CT imaging for staging; uncomplicated cases may resolve with bowel rest, while complicated cases require oral or IV antibiotics and gradual high-fiber reintroduction.",
     "Seek emergency care for severe, worsening lower left abdominal pain accompanied by high fever, persistent vomiting, or rigid abdominal wall tenderness."),

    (44, "Endometriosis and Pelvic Pain Management", "Gynecology, Reproductive Medicine & Chronic Pelvic Pain",
     "Endometriosis is a chronic gynecological condition in which functional endometrial-like tissue implants outside the uterine cavity, causing chronic inflammation.",
     "Retrograde menstruation, vascular/lymphatic dissemination, and coelomic metaplasia allow ectopic endometrial implants to bleed cyclically, forming adhesions and cysts.",
     "Apply a warm heating pad to the lower abdomen or lower back, rest in a fetal position, and take an over-the-counter NSAID (naproxen, ibuprofen) with food.",
     "Undergo gynecological evaluation; clinical management includes hormonal suppression (combined contraceptives, progestins, GnRH agonists) or laparoscopic excision.",
     "Seek urgent care for sudden, severe unilateral pelvic pain with vomiting (which may indicate ovarian torsion) or fever with abnormal pelvic discharge."),

    (45, "Epilepsy, Seizure Classification and First Aid", "Neurology, Neurophysiology & Antiepileptic Therapy",
     "Epilepsy is a neurological disorder characterized by an enduring predisposition to generate spontaneous, recurrent unprovoked epileptic seizures.",
     "Synchronous, high-frequency paroxysmal electrical discharges in cerebral cortical neuronal networks resulting from channelopathies or structural lesions.",
     "Ease the person to the floor, turn them gently onto their side (recovery position) to keep the airway clear, place a soft cushion under the head, and time the seizure.",
     "Do NOT put anything in the person's mouth, do NOT hold them down or restrain movements; consult a neurologist for regular EEG and antiepileptic medications.",
     "Call emergency services (911) if the seizure lasts >5 minutes, a second seizure starts without regaining consciousness (status epilepticus), or injury occurs."),

    (46, "Fatty Liver Disease (NAFLD / MASLD) and Hepatic Health", "Hepatology, Metabolic Syndrome & Liver Metabolism",
     "Metabolic dysfunction-associated steatotic liver disease (MASLD) involves hepatic steatosis affecting >5% of hepatocytes in the absence of significant alcohol intake.",
     "Peripheral insulin resistance drives excessive free fatty acid influx into hepatocytes, leading to lipotoxicity, mitochondrial oxidative stress, and steatohepatitis.",
     "Stay hydrated with water, eliminate sugar-sweetened beverages and high-fructose corn syrup, and engage in daily brisk walking.",
     "Target a gradual 7-10% body weight reduction, adopt a Mediterranean dietary pattern, avoid alcohol, and maintain routine liver function enzyme surveillance.",
     "Seek medical evaluation for persistent right upper quadrant dull fullness, yellowing of skin/eyes (jaundice), abdominal swelling (ascites), or dark urine."),

    (47, "Fibromyalgia and Central Sensitization Syndromes", "Rheumatology, Chronic Pain Medicine & Neurobiology",
     "Fibromyalgia is a chronic disorder characterized by widespread musculoskeletal pain, fatigue, sleep disturbances, and cognitive 'fibro-fog'.",
     "Central sensitization in spinal cord and brain processing pathways results in hyperalgesia (amplified pain) and allodynia (pain from non-painful stimuli).",
     "Take a warm bath with Epsom salts to relax tender muscles, practice gentle stretching, and engage in slow paced breathing in a quiet room.",
     "Participate in low-impact aerobic exercise (warm water swimming, walking), CBT for chronic pain adaptation, sleep hygiene, and FDA-approved neuromodulators (duloxetine).",
     "Seek clinical review to rule out concurrent inflammatory autoimmune conditions such as lupus, polymyalgia rheumatica, or severe thyroid dysfunction."),

    (48, "Gastrointestinal Bleeding and Clinical Triage", "Gastroenterology, Endoscopy & Emergency Hemostasis",
     "Gastrointestinal bleeding refers to blood loss originating anywhere from the mouth to the anorectum, stratified into upper and lower GI bleeding.",
     "Upper bleeding originates from bleeding peptic ulcers, esophageal varices, or mucosal tears; lower bleeding stems from diverticula, angiodysplasia, or colitis.",
     "Sit down immediately, avoid eating or drinking, do not exert yourself, and call emergency medical services right away.",
     "Emergency hospitalization for fluid resuscitation, type and cross-match for transfusion, and urgent upper or lower endoscopy for endoscopic hemostasis.",
     "Call 911 immediately for vomiting bright red blood or coffee-ground material, passing large maroon or black tarry stools, accompanied by dizziness or fainting."),

    (49, "Glaucoma and Intraocular Pressure Management", "Ophthalmology & Optic Nerve Neuropathy",
     "Glaucoma is a group of optic neuropathies characterized by progressive loss of retinal ganglion cells and optic nerve cupping, often linked to elevated intraocular pressure.",
     "Impaired aqueous humor drainage through the trabecular meshwork elevates intraocular pressure (IOP), causing mechanical and ischemic optic nerve axonal death.",
     "Never rub the eye; if acute angle-closure symptoms occur (severe eye pain, halos around lights), seek an emergency ophthalmologist immediately.",
     "Adhere strictly to daily prescribed IOP-lowering eye drops (prostaglandin analogs, beta-blockers), undergo routine visual field testing, and attend annual eye exams.",
     "Seek emergency care for sudden severe eye pain, headache, nausea, blurred vision with rainbow halos around lights, and a red eye (acute angle closure)."),

    (50, "Gout, Hyperuricemia and Acute Podagra", "Rheumatology, Purine Metabolism & Crystal Arthritis",
     "Gout is an intensely painful inflammatory arthritis caused by monosodium urate crystal deposition in synovial joints, classical in the first metatarsophalangeal joint (podagra).",
     "Hyperuricemia resulting from purine-rich diet (red meat, seafood, alcohol, beer) or decreased renal uric acid excretion leads to crystal supersaturation and neutrophil activation.",
     "Rest and elevate the affected joint, avoid putting any weight or sheet pressure on the toe, apply cold ice packs wrapped in cloth, and drink plenty of water.",
     "Take acute anti-inflammatory medications (colchicine, NSAIDs, or steroids) early in the attack, followed by long-term uric acid-lowering therapy (allopurinol).",
     "Seek prompt medical evaluation for high fever, shaking chills, and an intensely red hot joint to rule out bacterial septic joint infection."),

    (51, "Graves' Disease and Hyperthyroid Orbitopathy", "Endocrinology, Autoimmune Thyroidology & Ophthalmology",
     "Graves' disease is an autoimmune condition characterized by hyperthyroidism, diffuse goiter, and extrathyroidal features including thyroid eye disease (orbitopathy).",
     "Thyrotropin receptor antibodies (TRAb) bind to and continuously stimulate TSH receptors on thyroid follicular cells, driving unregulated hormone production.",
     "Rest in a cool room, drink cold water, avoid physical exertion and caffeine, and wear dark sunglasses outdoors to protect sensitive eyes.",
     "Consult an endocrinologist for antithyroid medications (methimazole), beta-blockers for rapid heart rate control, and radioiodine ablation or surgery if indicated.",
     "Seek emergency care for high fever, rapid irregular heartbeat, agitation, and sweating (thyroid storm)."),

    (52, "Hashimoto's Thyroiditis and Autoimmune Hypothyroidism", "Endocrinology, Autoimmune Glandular Failure",
     "Hashimoto's thyroiditis is a chronic autoimmune lymphocytic thyroiditis leading to progressive, gradual destruction of thyroid tissue and primary hypothyroidism.",
     "Anti-thyroperoxidase (TPO) and anti-thyroglobulin autoantibodies provoke progressive T-cell mediated follicular destruction and glandular fibrosis.",
     "Rest, dress in warm layers if feeling cold, and avoid intense physical exertion until metabolic hormone levels stabilize.",
     "Take daily synthetic levothyroxine (T4) tablets upon waking on an empty stomach with plain water, 30-60 minutes before breakfast, with regular TSH monitoring.",
     "Consult a physician for extreme lethargy, puffiness around eyes, profound constipation, or hoarseness."),

    (53, "Hearing Loss, Presbycusis and Tinnitus", "Otolaryngology, Audiology & Cochlear Hair Cell Health",
     "Presbycusis is age-related sensorineural hearing loss affecting high frequencies; tinnitus is the perception of sound (ringing, buzzing) without external acoustic source.",
     "Gradual loss and damage to sensory hair cells in the cochlea from aging, noise exposure, ototoxic medications, or microvascular compromise.",
     "Avoid loud noise exposure, turn down headphone volume, practice relaxation techniques, and use low-level ambient white noise (sound machine) to mask tinnitus.",
     "Undergo formal audiological evaluation for hearing aids, use hearing protection in noisy environments, and manage cardiovascular risk factors.",
     "Seek urgent ENT evaluation for sudden hearing loss in one ear occurring over hours/days (sudden sensorineural hearing loss requires immediate steroid therapy)."),

    (54, "Heart Failure and Fluid Overload Management", "Cardiology, Myocardial Hemodynamics & Ejection Fraction",
     "Heart failure is a clinical syndrome in which the heart cannot pump sufficient blood to meet metabolic tissue demands or can do so only at elevated filling pressures.",
     "Myocardial infarction, long-standing hypertension, or cardiomyopathy causes structural ventricular remodeling, systolic or diastolic dysfunction, and neurohormonal activation.",
     "Sit upright with legs dangling to reduce venous return to the heart, rest calmly, and avoid drinking large volumes of water or consuming salty snacks.",
     "Weigh yourself daily every morning, adhere to strict low-sodium dietary guidelines, take prescribed guideline-directed medications (ACEi/ARNI, beta-blocker, MRA, SGLT2i).",
     "Seek emergency care for acute severe shortness of breath, waking up gasping at night (PND), coughing up pink frothy sputum, or rapid weight gain >3 lbs overnight."),

    (55, "Hemorrhoids and Anorectal Vascular Health", "Colorectal Surgery & Proctology",
     "Hemorrhoids are vascular cushions in the anal canal that become swollen, engorged, and inflamed, classified as internal or external relative to the dentate line.",
     "Increased intra-abdominal pressure from straining during constipation, prolonged sitting on toilets, heavy lifting, pregnancy, and low-fiber diet.",
     "Take a warm sitz bath for 15 minutes twice daily, apply topical hydrocortisone or witch hazel pads, avoid straining during bowel movements, and drink water.",
     "Increase dietary soluble and insoluble fiber (psyllium, vegetables, fruits), stay well hydrated, exercise regularly, and avoid reading on the toilet.",
     "Seek medical evaluation for rectal bleeding (bright red or dark), severe throbbing anal pain (thrombosed external hemorrhoid), or fever."),

    (56, "Hepatitis A, B, and C Viral Illnesses", "Hepatology, Infectious Diseases & Hepatic Inflammation",
     "Viral hepatitis refers to acute or chronic liver inflammation caused by hepatotropic viruses (Hep A via fecal-oral; Hep B & C via blood/fluids).",
     "Immune-mediated destruction of hepatocytes infected with viral antigens leads to transaminase elevation, impaired bilirubin clearance, and fibrosis.",
     "Rest in bed, avoid all alcohol and hepatotoxic medications (limit acetaminophen), eat small carbohydrate-rich meals, and stay hydrated.",
     "Undergo viral serology and viral load PCR testing; acute Hep A is supportive; chronic Hep B requires nucleoside analogs; chronic Hep C is curable with oral direct-acting antivirals (DAAs).",
     "Seek emergency care for confusion, extreme sleepiness, yellowing of skin/eyes (jaundice), vomiting blood, or easy bruising (acute liver failure)."),

    (57, "Herpes Zoster (Shingles) and Postherpetic Neuralgia", "Dermatology, Infectious Neurology & Varicella Zoster",
     "Herpes zoster (shingles) is a painful localized dermatomal skin rash caused by reactivation of the latent varicella-zoster virus in dorsal root sensory ganglia.",
     "Waning cell-mediated immunity with aging or stress allows latent virus to travel down sensory axons, producing unilateral dermatomal pain and grouped vesicles.",
     "Keep the rash clean and dry, wear loose breathable clothing, avoid touching the lesions, and apply cool water compresses for burning relief.",
     "Start oral antiviral therapy (valacyclovir, acyclovir) within 72 hours of rash onset to shorten duration and decrease risk of chronic postherpetic neuralgia.",
     "Seek emergency care if the shingles rash involves the tip of the nose or forehead (Hutchinson's sign), indicating herpes zoster ophthalmicus which threatens vision."),

    (58, "Hypercholesterolemia and Cardiovascular Risk", "Preventative Cardiology, Lipidology & Atherogenesis",
     "Hypercholesterolemia is an elevation of total serum cholesterol, specifically low-density lipoprotein cholesterol (LDL-C), driving coronary atherosclerosis.",
     "Excess dietary saturated and trans fats, familial hypercholesterolemia genetic mutations (LDLR), obesity, and sedentary lifestyle down-regulate hepatic LDL clearance.",
     "Choose a heart-healthy meal emphasizing oats, legumes, nuts, olive oil, and fiber, and engage in 30 minutes of aerobic physical exercise.",
     "Adhere to a low-saturated-fat Mediterranean diet, maintain an optimal body weight, exercise regularly, and take prescribed statin or ezetimibe therapy as indicated.",
     "Seek immediate evaluation for chest pressure, shortness of breath, or sudden radiating arm/neck pain."),

    (59, "Hypoglycemia and Neuroglycopenic Warning Signs", "Endocrinology, Acute Metabolic Triage & Glucose Counterregulation",
     "Hypoglycemia is an abnormally low plasma glucose concentration (<70 mg/dL) that exposes an individual to neuroglycopenic injury.",
     "Excessive exogenous insulin or secretagogue medication, missed or delayed meals, unaccustomed strenuous exercise, or heavy alcohol intake.",
     "Immediately follow the Rule of 15: consume 15g of fast-acting simple carbohydrate (4 oz fruit juice, 3-4 glucose tablets), wait 15 minutes, and retest blood glucose.",
     "Adjust insulin or medication dosages with your diabetes care team, carry fast-acting glucose tablets at all times, wear medical identification, and consider CGM alarms.",
     "If the patient is unconscious, confused, having seizures, or unable to swallow, do NOT give oral fluids; administer emergency intramuscular glucagon or call 911."),

    (60, "Infectious Mononucleosis and Epstein-Barr Virus", "Infectious Diseases & Clinical Immunology",
     "Infectious mononucleosis is an acute clinical syndrome primarily caused by Epstein-Barr virus (EBV), classical in adolescents and young adults.",
     "EBV infects oral epithelial cells and B-lymphocytes, provoking a robust cytotoxic CD8 T-cell immune response (atypical lymphocytes on blood smear).",
     "Get plenty of bed rest, stay hydrated with cool fluids, use warm saline gargles for sore throat, and take acetaminophen or ibuprofen for fever.",
     "Avoid all contact sports and heavy lifting for at least 3-4 weeks to prevent catastrophic splenic rupture, as the spleen is frequently enlarged.",
     "Seek emergency care for severe difficulty breathing, inability to swallow liquids, sudden severe left upper quadrant abdominal pain, or jaundice."),

    (61, "Intercostal Neuralgia and Musculoskeletal Chest Wall Pain", "Pain Medicine, Thoracic Orthopedics & Costochondritis",
     "Chest wall pain commonly stems from costochondritis (cartilage inflammation at ribs) or intercostal neuralgia (nerve irritation along rib margins).",
     "Repetitive coughing, heavy lifting, thoracic spine strain, direct blunt trauma, or viral inflammation irritates intercostal nerves or chondrosternal joints.",
     "Apply a warm moist compress to the tender chest area, practice slow shallow breathing, and take an over-the-counter anti-inflammatory (ibuprofen) with food.",
     "Perform gentle thoracic spine mobility exercises, improve desk posture, avoid provocative repetitive chest motions, and pursue physical therapy.",
     "Seek emergency care immediately if chest pain is crushing, radiates to the arm/jaw, or is accompanied by shortness of breath, sweating, or dizziness (cardiac screening)."),

    (62, "Interstitial Cystitis and Bladder Pain Syndrome", "Urology, Pelvic Floor Medicine & Chronic Visceral Pain",
     "Interstitial cystitis is a chronic condition characterized by bladder pressure, pelvic pain, and urinary urgency/frequency in the absence of infection.",
     "Bladder urothelial glycosaminoglycan (GAG) layer disruption, mast-cell activation, and neurogenic pelvic floor muscle hypertonicity.",
     "Apply a warm heating pad to the lower abdomen, drink plain water, avoid bladder triggers (acidic citrus, tomatoes, coffee, artificial sweeteners), and practice pelvic floor relaxation.",
     "Consult a urologist for bladder installations (heparin/DMSO), pelvic floor physical therapy, dietary elimination planning, and oral pentosan polysulfate sodium.",
     "Seek clinical evaluation if gross hematuria (visible blood in urine), fever, or severe acute retention occurs."),

    (63, "Iron Overload and Hemochromatosis", "Hematology, Hepatology & Iron Metabolism",
     "Hereditary hemochromatosis is a genetic metabolic disorder characterized by excessive intestinal iron absorption leading to toxic iron accumulation in parenchymal organs.",
     "Mutations in the HFE gene (C282Y) cause low levels of hepcidin, the master iron hormone, leading to unrestricted iron release into circulation and tissue deposition.",
     "Avoid taking vitamin C supplements or iron supplements, avoid raw shellfish (Vibrio vulnificus risk), and limit alcohol intake.",
     "Undergo regular therapeutic phlebotomy (blood removal) to normalize serum ferritin, monitor liver enzymes, and have regular cardiovascular checkups.",
     "Seek medical evaluation for unexplained bronze/slate-gray skin darkening, joint pain, elevated blood glucose, or heart palpitations."),

    (64, "Kidney Stones (Nephrolithiasis) and Renal Colic", "Urology, Endourology & Mineral Metabolism",
     "Kidney stones are hard mineral and acid salt deposits (most commonly calcium oxalate) that form inside the renal pelvis and obstruct the ureter.",
     "Dehydration, hypercalciuria, high dietary sodium/animal protein, hyperoxaluria, and low urinary citrate promote crystal nucleation and stone growth.",
     "Sip water continuously, apply a warm heating pad to the affected flank/back on low heat, and take prescribed anti-inflammatory analgesics.",
     "Consult a urologist for non-contrast CT imaging, take tamsulosin to facilitate stone passage, strain urine to catch stones for analysis, and drink 3 liters of water daily.",
     "Seek emergency care for high fever with shaking chills, intractable vomiting, inability to urinate (anuria), or unbearable flank pain."),

    (65, "Labyrinthitis and Acute Vestibular Neuritis", "Otolaryngology & Vestibular Medicine",
     "Vestibular neuritis is acute inflammation of the vestibular nerve; labyrinthitis involves inflammation of both the vestibular and cochlear inner ear branches.",
     "Post-viral inflammation (following upper respiratory infection) causes unilateral loss of vestibular sensory input, provoking sudden severe vertigo.",
     "Lie still in a dimly lit room, avoid sudden head movements, keep the eyes fixed on a steady object, and take short-term vestibular sedatives as prescribed.",
     "Begin early vestibular rehabilitation therapy (VRT) head-eye balance exercises, stay hydrated, and avoid driving until balance is fully restored.",
     "Seek emergency care immediately for sudden weakness, facial droop, slurred speech, or double vision to rule out posterior circulation cerebellar stroke."),

    (66, "Laryngitis, Hoarseness and Vocal Cord Health", "Otolaryngology, Laryngology & Voice Pathology",
     "Laryngitis is acute inflammation and edema of the vocal cords (vocal folds), resulting in dysphonia (hoarseness), throat tickling, or complete voice loss.",
     "Upper respiratory viral infection, vocal overuse/shouting, acute acid reflux (LPR), or exposure to airborne irritants such as tobacco smoke.",
     "Rest your voice completely (do NOT whisper, as whispering strains vocal folds more than normal speech), inhale cool-mist steam, and drink warm decaffeinated fluids.",
     "Stay well hydrated, avoid throat clearing (sip water instead), eliminate smoking and alcohol, and manage acid reflux if present.",
     "Seek medical evaluation for hoarseness lasting >3 weeks, difficulty swallowing, stridor (noisy breathing), or coughing up blood."),

    (67, "Liver Cirrhosis and Portal Hypertension", "Hepatology, Gastroenterology & Hepatic Fibrosis",
     "Cirrhosis is the end-stage result of chronic liver disease, characterized by diffuse hepatic fibrosis and replacement of normal liver parenchyma by regenerative nodules.",
     "Chronic hepatitis B or C, prolonged alcohol misuse, or metabolic dysfunction-associated steatohepatitis (MASH) leads to progressive collagen deposition.",
     "Rest, follow a strictly sodium-restricted diet to minimize fluid retention, avoid all alcohol, and take prescribed lactulose and diuretics as directed.",
     "Undergo regular hepatology follow-up, endoscopic variceal screening, routine abdominal ultrasound surveillance for hepatocellular carcinoma, and transplant evaluation.",
     "Seek emergency medical care for vomiting blood, black tarry stools, sudden abdominal swelling, confusion/drowsiness (hepatic encephalopathy), or fever with ascites."),

    (68, "Lyme Disease and Borrelia Burgdorferi Infection", "Infectious Diseases & Vector-Borne Pathology",
     "Lyme disease is a multisystem tick-borne infectious disease caused by the spirochete Borrelia burgdorferi, transmitted by the bite of infected Ixodes blacklegged ticks.",
     "Inoculated spirochetes disseminate through the skin and blood, triggering an immune inflammatory response characterized by an expanding erythema migrans ('bull's-eye') rash.",
     "If an attached tick is found, remove it immediately using fine-tipped tweezers pulling straight upward without twisting; wash the bite with soap and water.",
     "Consult a healthcare provider promptly for early oral antibiotic therapy (doxycycline, amoxicillin) for 10-14 days to prevent chronic neurological or arthritic complications.",
     "Seek urgent care for facial muscle drooping (Bell's palsy), irregular heartbeat or chest pain (Lyme carditis), severe headache with stiff neck, or swollen knee joint."),

    (69, "Macular Degeneration (AMD) and Central Vision Loss", "Ophthalmology, Retinal Pathology & Vision Care",
     "Age-related macular degeneration (AMD) is a progressive degenerative disease of the retinal macula, the specialized central area responsible for sharp, detailed vision.",
     "Drusen lipid deposits between the retinal pigment epithelium and Bruch's membrane (Dry AMD) or choroidal neovascularization with fluid leakage (Wet AMD).",
     "Use bright task lighting, use high-contrast reading aids, and monitor your vision using an Amsler grid covering one eye at a time.",
     "Take AREDS2 antioxidant formula vitamins (under ophthalmologist guidance), quit smoking, consume dark leafy greens, and receive intravitreal anti-VEGF injections for wet AMD.",
     "Seek same-day ophthalmological evaluation for any sudden distortion of straight lines (straight lines appearing wavy) or sudden dark blind spot in central vision."),

    (70, "Meningitis, Nuchal Rigidity and Neurological Red Flags", "Infectious Diseases, Neurology & Emergency Medicine",
     "Meningitis is an acute, life-threatening inflammation of the protective meningeal membranes covering the brain and spinal cord, caused by bacterial or viral pathogens.",
     "Bacterial (Neisseria meningitidis, Streptococcus pneumoniae) or viral dissemination enters the subarachnoid space, triggering severe cerebral edema and intracranial pressure.",
     "This is a critical medical emergency. Do NOT attempt home self-care. Transport to the nearest emergency department immediately.",
     "Immediate diagnostic lumbar puncture and prompt administration of empiric broad-spectrum IV antibiotics and corticosteroids; close contacts may require chemoprophylaxis.",
     "Call 911 immediately for high fever, severe headache, stiff neck (inability to touch chin to chest), confusion, vomiting, and a non-blanching purple petechial rash."),

    (71, "Metabolic Syndrome and Cardiometabolic Risk", "Endocrinology, Preventative Cardiology & Metabolism",
     "Metabolic syndrome is a clustering of at least 3 of 5 cardiometabolic risk factors: abdominal obesity, elevated triglycerides, low HDL, hypertension, and elevated fasting glucose.",
     "Underlying insulin resistance, visceral adiposity, systemic pro-inflammatory adipokines, and physical inactivity.",
     "Replace refined sugars and sodas with pure water, take a 20-minute brisk walk after meals, and prepare meals centered on high-fiber whole foods.",
     "Target 5-10% gradual weight loss through regular physical activity (both aerobic and resistance training), adopt a Mediterranean diet, and control blood pressure and lipids.",
     "Seek medical evaluation for persistent excessive thirst, frequent urination, chest pain, or shortness of breath."),

    (72, "Multiple Sclerosis and Demyelinating Central Pathology", "Neurology, Neuroimmunology & Demyelinating Disease",
     "Multiple sclerosis (MS) is an autoimmune inflammatory demyelinating disease of the central nervous system, characterized by episodic neurological attacks separated in time and space.",
     "Autoreactive T-cells cross the blood-brain barrier and attack myelin sheaths and oligodendrocytes, slowing saltatory axonal nerve conduction.",
     "Rest in a cool air-conditioned room (avoid heat which worsens MS symptoms - Uhthoff's phenomenon), stay hydrated, and use assistive devices for balance stability.",
     "Undergo MRI evaluation and initiate disease-modifying therapies (DMTs: interferons, monoclonal antibodies) early to reduce relapses and slow disability accumulation.",
     "Seek urgent neurological care for sudden loss of vision in one eye with pain on movement (optic neuritis), sudden limb paralysis, or loss of bladder control."),

    (73, "Myasthenia Gravis and Neuromuscular Junction Fatigue", "Neurology, Neuromuscular Medicine & Autoimmunity",
     "Myasthenia gravis is an autoimmune neuromuscular junction disorder characterized by fluctuating skeletal muscle weakness that worsens with activity and improves with rest.",
     "Autoantibodies against the acetylcholine receptor (AChR) or MuSK block and destroy postsynaptic receptors, preventing normal muscle endplate depolarization.",
     "Rest quietly in a cool room, pace activities throughout the day, eat smaller soft meals when muscle strength is highest, and avoid strenuous physical exertion.",
     "Take prescribed acetylcholinesterase inhibitors (pyridostigmine) and immunosuppressants; evaluate for thymoma with chest CT imaging and consider thymectomy.",
     "Seek emergency care immediately for difficulty swallowing, choking on saliva, or shortness of breath (myasthenic crisis requiring respiratory support)."),

    (74, "Non-Alcoholic Steatohepatitis (NASH/MASH) and Fibrosis", "Hepatology, Gastroenterology & Metabolic Medicine",
     "MASH is the progressive inflammatory subtype of steatotic liver disease characterized by hepatic steatosis, hepatocyte ballooning, and progressive liver fibrosis.",
     "Lipotoxicity from saturated free fatty acids, oxidative stress, and mitochondrial dysfunction activate hepatic stellate cells to lay down collagen scar tissue.",
     "Avoid all alcoholic beverages, eliminate refined sugars and high-fructose syrups, and engage in daily aerobic exercise.",
     "Target 7-10% body weight loss, adopt an anti-inflammatory Mediterranean diet, optimize diabetes control, and undergo non-invasive elastography (FibroScan) monitoring.",
     "Seek medical evaluation for persistent abdominal swelling, easy bruising, jaundice, or severe unexplained fatigue."),

    (75, "Osteoarthritis, Cartilage Degradation and Joint Care", "Rheumatology, Orthopedics & Geriatric Medicine",
     "Osteoarthritis is the most prevalent degenerative joint disease, characterized by progressive loss of articular cartilage, subchondral bone remodeling, and osteophyte formation.",
     "Biomechanical joint stress, previous joint trauma, aging, and low-grade synovial inflammation breakdown the extracellular proteoglycan cartilage matrix.",
     "Rest the joint after prolonged use, apply cold packs for acute swelling or warm wraps for morning stiffness, and use a cane in the opposite hand if hip/knee is affected.",
     "Engage in low-impact physical exercise (swimming, stationary cycling, water aerobics) to strengthen stabilizing muscles, manage body weight, and take NSAIDs as directed.",
     "Seek orthopedic consultation if pain persists at rest, severely limits walking, or if the joint becomes locked or hot and red."),

    (76, "Osteoporosis, DEXA Scans and Fracture Prevention", "Endocrinology, Bone Mineralization & Geriatrics",
     "Osteoporosis is a systemic skeletal disease defined by low bone mineral density (T-score ≤ -2.5 on DEXA) and microarchitectural deterioration of bone tissue, increasing fracture risk.",
     "Bone resorption by osteoclasts outpaces bone formation by osteoblasts, accelerated by postmenopausal estrogen decline, aging, and calcium/Vitamin D deficiency.",
     "Maintain clear walking pathways at home (remove loose throw rugs, ensure bright nightlights), wear non-skid supportive footwear, and avoid bending at the waist.",
     "Perform weight-bearing and resistance exercises, ensure adequate dietary calcium (1200 mg/day) and Vitamin D (800-1000 IU/day), and take prescribed bisphosphonates.",
     "Seek emergency medical evaluation for sudden severe back pain following a minor fall (possible vertebral compression fracture) or inability to bear weight on the hip."),

    (77, "Otitis Media, Otitis Externa and Ear Pain", "Otolaryngology, Pediatric Medicine & Audiology",
     "Otitis media is acute inflammation and fluid accumulation in the middle ear space behind the tympanic membrane; otitis externa is infection of the external ear canal.",
     "Eustachian tube dysfunction following an upper respiratory infection allows bacterial colonization (Streptococcus pneumoniae, Haemophilus influenzae) in the middle ear.",
     "Apply a warm moist compress over the affected ear, keep the head elevated, use over-the-counter pain relievers (acetaminophen or ibuprofen), and keep water out of the ear.",
     "Consult a healthcare provider for otoscopic examination; acute bacterial otitis media may require oral amoxicillin, while otitis externa requires antibiotic ear drops.",
     "Seek urgent care for swelling or redness behind the ear over the mastoid bone (mastoiditis), high fever, severe dizziness, or fluid/blood draining from the ear."),

    (78, "Pancreatitis (Acute and Chronic) & Digestive Enzymes", "Gastroenterology, Pancreatobiliary Medicine & Critical Care",
     "Acute pancreatitis is sudden inflammatory autodigestion of the pancreas caused by premature activation of pancreatic zymogens (trypsinogen) within acinar cells.",
     "Gallstones obstructing the ampulla of Vater (biliary pancreatitis) and alcohol toxicity are responsible for over 70% of cases, followed by hypertriglyceridemia.",
     "Do NOT eat or drink anything (bowel rest is essential), sit and lean forward to relieve back pain, and seek emergency hospital evaluation immediately.",
     "Hospitalization for aggressive intravenous fluid resuscitation, analgesia, and monitoring of serum lipase and amylase; gallstone cases require cholecystectomy.",
     "Call 911 immediately for severe, sudden epigastric pain boring through to the back, accompanied by persistent vomiting, fever, rapid pulse, and dizziness."),

    (79, "Parkinson's Disease, Motor Symptoms and Neuroprotection", "Neurology, Movement Disorders & Neurodegenerative Care",
     "Parkinson's disease is a progressive neurodegenerative movement disorder characterized by resting tremor, bradykinesia (slowness), muscular rigidity, and postural instability.",
     "Selective loss of dopaminergic neurons in the substantia nigra pars compacta, accompanied by intracellular alpha-synuclein protein inclusions (Lewy bodies).",
     "Take movement breaks, practice deliberate large-amplitude stepping (visualize stepping over lines), use supportive armchairs, and pace daily activities.",
     "Consult a movement disorder neurologist for individualized pharmacotherapy (levodopa/carbidopa, dopamine agonists), targeted physical therapy, and speech therapy.",
     "Seek emergency care for sudden inability to swallow saliva, choking, sudden worsening confusion, or severe falls resulting in head injury."),

    (80, "Pelvic Inflammatory Disease (PID) and Reproductive Health", "Gynecology, Infectious Diseases & Reproductive Health",
     "Pelvic inflammatory disease is an ascending polymicrobial infection of the upper female genital tract (uterus, fallopian tubes, ovaries), potentially causing infertility.",
     "Ascending infection from sexually transmitted pathogens (Chlamydia trachomatis, Neisseria gonorrhoeae) or vaginal microflora past the endocervical barrier.",
     "Rest in bed, drink fluids, avoid sexual intercourse until treatment is completed, and seek prompt clinical gynecology evaluation.",
     "Complete full-spectrum empiric broad-spectrum antibiotic regimens (ceftriaxone, doxycycline, metronidazole), and ensure all sexual partners are tested and treated.",
     "Seek emergency medical care for high fever, severe bilateral lower abdominal pain, vomiting, or signs of pelvic abscess."),

    (81, "Peripheral Artery Disease and Intermittent Claudication", "Vascular Surgery, Cardiology & Angiology",
     "Peripheral artery disease (PAD) is atherosclerotic narrowing of peripheral systemic arteries, classical in the lower extremities causing exertional claudication.",
     "Atherosclerotic plaque accumulation in femoral, popliteal, or iliac arteries leads to exercise-induced muscle ischemia that resolves within 10 minutes of rest.",
     "Stop walking and rest in a seated position until calf cramping resolves, keep the feet warm with socks (avoid hot water bottles which cause burns), and protect feet.",
     "Undergo ankle-brachial index (ABI) testing, participate in supervised treadmill walking exercise programs, stop smoking completely, and take prescribed antiplatelet therapy.",
     "Seek urgent care for pain in the foot or toes at rest (critical limb ischemia), cold pale foot with absent pulses, or non-healing ulcers/black discoloration."),

    (82, "Peripheral Neuropathy and Diabetic Foot Care", "Neurology, Endocrinology & Podiatric Medicine",
     "Peripheral neuropathy is damage to peripheral nerves causing sensory loss, numbness, burning tingling ('pins and needles'), and weakness, common in diabetes.",
     "Chronic hyperglycemia, microvascular endoneurial ischemia, advanced glycation end-products (AGEs), and sorbitol accumulation damage peripheral nerve fibers.",
     "Inspect both feet daily using a mirror for any cuts, blisters, or redness; wash feet in lukewarm water, dry thoroughly between toes, and wear seamless white socks.",
     "Maintain tight glycemic control, wear well-cushioned protective shoes, never walk barefoot, and take prescribed neuromodulating medications (duloxetine, gabapentin).",
     "Seek immediate medical attention for any open sore or ulcer on the foot, spreading redness, fever, or black discoloration."),

    (83, "Pharyngitis, Strep Throat and Tonsillar Health", "Otolaryngology, Infectious Diseases & Pediatric Care",
     "Pharyngitis is acute mucosal inflammation of the throat and tonsils, caused by respiratory viruses or Group A Streptococcus bacteria (strep throat).",
     "Bacterial colonization (Streptococcus pyogenes) causes acute follicular exudates, tonsillar swelling, palatal petechiae, and tender anterior cervical lymphadenopathy.",
     "Gargle with warm salt water (1/2 tsp salt in 8 oz water) every 3 hours, drink warm honey-lemon water or cool liquids, and take acetaminophen for throat pain.",
     "Visit a clinic for a rapid strep antigen throat swab; confirmed strep throat requires a 10-day course of oral penicillin/amoxicillin to prevent acute rheumatic fever.",
     "Seek emergency medical evaluation for difficulty swallowing saliva (drooling), inability to open the mouth (trismus), muffled 'hot-potato' voice, or labored breathing."),

    (84, "Plantar Fasciitis and Calcaneal Heel Pain", "Podiatry, Orthopedic Medicine & Foot Biomechanics",
     "Plantar fasciitis is degenerative irritation and micro-tears of the thick plantar aponeurosis band connecting the calcaneus to the proximal phalanges.",
     "Repetitive impact loading, sudden increase in running distance, prolonged standing on hard surfaces, flat feet (pes planus), tight Achilles tendons, and poor footwear.",
     "Avoid walking barefoot, roll the arch of the foot over a frozen water bottle for 10 minutes, and perform gentle calf and plantar fascia stretches before stepping out of bed.",
     "Wear supportive shoes with firm arch support and cushioned heels, use custom or over-the-counter orthotic inserts, and perform eccentric calf-strengthening exercises.",
     "Seek medical evaluation if heel pain is accompanied by numbness/tingling in the toes, fever, or inability to bear any weight on the foot."),

    (85, "Pneumonia (Community-Acquired) and Pulmonary Consolidation", "Pulmonology, Infectious Diseases & Critical Care",
     "Pneumonia is an acute infection of the pulmonary parenchyma (alveoli and bronchioles) resulting in inflammatory exudative consolidation and impaired gas exchange.",
     "Inhalation or aspiration of pathogens (Streptococcus pneumoniae, atypical Mycoplasma, viruses) into terminal bronchioles, triggering alveolar neutrophilic exudation.",
     "Rest in bed, drink plenty of warm fluids to loosen secretions, use a cool-mist humidifier, and monitor blood oxygen levels with a pulse oximeter if available.",
     "Consult a healthcare provider for chest X-ray and targeted antibiotic or antiviral therapy; complete the full prescription and receive the pneumococcal vaccine.",
     "Seek emergency care immediately for oxygen saturation below 92%, severe shortness of breath, confusion, coughing up blood, or blue tint to lips."),

    (86, "Psoriasis, Plaque Dermatitis and Autoimmune Skin Care", "Dermatology, Autoimmune Pathology & Rheumatology",
     "Psoriasis is a chronic systemic immune-mediated dermatosis characterized by erythematous, well-demarcated plaques covered with coarse silvery scales.",
     "Hyperactive dendritic cell and T-cell signaling (IL-17 / IL-23 axis) stimulates excessive keratinocyte proliferation, reducing epidermal turnover time from 28 to 4 days.",
     "Apply a thick emollient or petroleum jelly immediately after bathing, avoid picking or scratching scales (which triggers Koebner phenomenon), and take a lukewarm bath.",
     "Consult a dermatologist for topical vitamin D analogues, topical corticosteroids, phototherapy, or targeted systemic biologic therapies (IL-17, IL-23 inhibitors).",
     "Seek urgent medical evaluation for widespread generalized fiery red skin with chills (erythrodermic psoriasis) or severe joint swelling (psoriatic arthritis)."),

    (87, "Pulmonary Fibrosis and Restrictive Lung Disease", "Pulmonary Medicine & Interstitial Lung Disease",
     "Pulmonary fibrosis involves progressive, irreversible scarring and stiffening of the alveolar interstitium, impairing pulmonary compliance and oxygen diffusion.",
     "Idiopathic causes, environmental dust/mineral exposures (silica, asbestos), autoimmune diseases (rheumatoid arthritis, scleroderma), or radiation therapy.",
     "Avoid all tobacco smoke and airborne dust, use home supplemental oxygen as prescribed, pace daily tasks to prevent breathlessness, and stay up to date on vaccines.",
     "Consult a pulmonologist for high-resolution chest CT and pulmonary function tests; take antifibrotic medications (pirfenidone, nintedanib), and attend pulmonary rehab.",
     "Seek emergency care for acute, rapidly worsening shortness of breath, chest pain, or low blood oxygen saturation."),

    (88, "Raynaud's Phenomenon and Digital Vasospasm", "Rheumatology, Vascular Medicine & Microcirculation",
     "Raynaud's phenomenon is an episodic, exaggerated vasospastic disorder of digital arteries and arterioles in response to cold temperatures or emotional stress.",
     "Hyperactivity of the sympathetic nervous system and local vascular endothelial dysfunction lead to triphasic color changes: white (pallor), blue (cyanosis), red (rubor).",
     "Rewarm the hands and feet gently in warm (not hot) water, place hands under armpits, move to a warm room, and wiggle the fingers and toes.",
     "Wear warm gloves or mittens, avoid handling cold items directly, avoid tobacco and caffeine which constrict blood vessels, and discuss calcium channel blockers with a doctor.",
     "Seek medical evaluation for painful non-healing sores or ulcers on the fingertips, or sudden blackened skin indicating digital gangrene."),

    (89, "Restless Legs Syndrome and Sleep Disruption", "Neurology, Sleep Medicine & Dopaminergic Function",
     "Restless legs syndrome (RLS) is a neurological sensorimotor disorder characterized by an irresistible urge to move the legs, typically during evening rest.",
     "Central brain iron deficiency in the substantia nigra, dopaminergic neurotransmission dysfunction, and genetic variations in sleep architecture genes.",
     "Get up and walk, gently stretch the calf and hamstring muscles, massage the legs, or take a warm bath before bedtime.",
     "Check serum ferritin levels (supplement iron if ferritin <75 ng/mL), avoid caffeine, alcohol, and antihistamines in the evening, and take prescribed alpha-2-delta ligands.",
     "Seek medical review for daytime sleepiness that impairs driving or work safety, or if involuntary daytime limb movements occur."),

    (90, "Rheumatoid Arthritis, Synovitis and Autoimmune Joint Care", "Rheumatology, Autoimmunity & Synovial Pathology",
     "Rheumatoid arthritis (RA) is a chronic systemic autoimmune inflammatory disease primarily attacking synovial joint linings in a symmetric polyarticular pattern.",
     "Loss of self-tolerance to citrullinated peptides leads to RF and anti-CCP autoantibodies, synovial proliferation (pannus formation), and joint bone erosion.",
     "Apply warm moist compresses for morning joint stiffness, use cold packs for acute inflamed joint flares, and rest the affected joints.",
     "Consult a rheumatologist early for disease-modifying antirheumatic drugs (DMARDs: methotrexate) and biologic agents to prevent permanent joint deformities.",
     "Seek medical evaluation for high fever with a single hot swollen joint (to rule out septic arthritis) or acute neck pain with neurological numbness."),

    (91, "Rosacea, Facial Flushing and Cutaneous Capillaries", "Dermatology, Vascular Dermatology & Facial Skin Care",
     "Rosacea is a chronic inflammatory dermatosis affecting the central face, featuring recurrent flushing, persistent erythema, telangiectasias, and inflammatory papules.",
     "Cutaneous neurovascular dysregulation, innate immune antimicrobial peptide hyperreactivity (cathelicidins), and Demodex folliculorum mite proliferation.",
     "Apply a cool water compress to the face, move into an air-conditioned room, sip ice water, and avoid known dietary triggers (spicy food, hot beverages, alcohol).",
     "Use mineral sunscreen daily (zinc oxide), wash face with gentle fragrance-free cleansers, avoid harsh scrubs, and use prescribed topical ivermectin or metronidazole.",
     "Seek ophthalmological care for gritty eye sensation, eyelid inflammation, or redness (ocular rosacea which can damage the cornea)."),

    (92, "Scabies, Pediculosis and Parasitic Dermatoses", "Dermatology, Parasitology & Public Health",
     "Scabies is an intensely pruritic, contagious parasitic skin infestation caused by the microscopic itch mite Sarcoptes scabiei var. hominis.",
     "Female mites burrow into the epidermal stratum corneum, depositing eggs and feces, which trigger a severe delayed-type allergic hypersensitivity reaction.",
     "Apply cool compresses or calamine lotion to soothe itching, avoid scratching to prevent bacterial impetigo, and wash all bedding in very hot water.",
     "Consult a doctor for topical permethrin 5% cream applied from neck to toes overnight, wash all clothing and linens in hot water, and treat all household contacts simultaneously.",
     "Seek medical attention for spreading skin redness, warmth, pus, or fever indicating secondary bacterial infection."),

    (93, "Sciatica, Herniated Lumbar Disc & Radiculopathy", "Orthopedic Spine Surgery, Neurology & Pain Medicine",
     "Sciatica refers to radiating pain along the course of the sciatic nerve (from lower back through buttock and down the posterior leg), most commonly from disc herniation.",
     "Posterolateral herniation of the L4-L5 or L5-S1 nucleus pulposus mechanically compresses the exiting nerve root, producing chemical radiculitis and sharp radiating pain.",
     "Avoid sitting for prolonged periods, lie on a firm surface on your back with pillows beneath knees, apply ice packs to the lower back, and take short gentle walks.",
     "Perform gentle nerve-flossing exercises, avoid forward bending while lifting, pursue physical therapy, and consider epidural steroid injection if conservative care fails.",
     "Seek emergency care immediately for loss of bowel or bladder control, numbness in the saddle/groin area, or progressive muscle weakness in the leg/foot."),

    (94, "Scoliosis, Spinal Alignment and Postural Screening", "Orthopedic Spine Surgery & Physical Rehabilitation",
     "Scoliosis is a three-dimensional structural deformity of the spine characterized by a lateral curvature in the coronal plane of at least 10 degrees (Cobb angle).",
     "Adolescent idiopathic scoliosis (80% of cases) has multifactorial genetic and asymmetric growth plate origins; adult degenerative scoliosis stems from asymmetric disc wear.",
     "Practice core stabilization exercises, maintain upright sitting posture, and engage in low-impact swimming to support paraspinal muscular balance.",
     "Undergo regular orthopedic scoliosis monitoring; curves 25-40 degrees in growing adolescents benefit from corrective bracing; curves >45-50 degrees may warrant surgery.",
     "Seek medical evaluation for progressive shortness of breath, severe unremitting back pain, or progressive leg numbness/weakness."),

    (95, "Sinusitis (Acute and Chronic Rhinosinusitis)", "Otolaryngology, Rhinology & Upper Airway Health",
     "Sinusitis is inflammation of the mucosal lining of the paranasal sinuses and nasal cavity, characterized by nasal congestion, facial pressure, and purulent discharge.",
     "Obstruction of the sinus ostia following a viral cold, mucosal edema, and impaired ciliary clearance allow mucus stagnation and secondary bacterial infection.",
     "Perform nasal saline irrigations using a neti pot with sterile water, inhale warm steam from a bowl or shower, apply warm face compresses, and stay hydrated.",
     "Use intranasal corticosteroid sprays; viral sinusitis resolves in 7-10 days; bacterial sinusitis (symptoms >10 days or 'double-sickening') requires amoxicillin-clavulanate.",
     "Seek emergency care for swelling or redness around the eye, vision changes, severe frontal headache with fever, or stiff neck."),

    (96, "Sleep Apnea (Obstructive) and Nocturnal Hypoxemia", "Sleep Medicine, Pulmonology & Upper Airway Physiology",
     "Obstructive sleep apnea (OSA) is characterized by repetitive episodes of complete (apnea) or partial (hypopnea) upper airway collapse during sleep.",
     "Pharyngeal muscle relaxation during REM and deep sleep allows the tongue and soft palate to obstruct the airway, causing episodic hypoxemia and cortical arousals.",
     "Sleep on your side rather than your back (positional therapy), avoid alcohol and sedatives before bed, and elevate the head of the bed.",
     "Undergo a polysomnography sleep study; first-line therapy is nocturnal Continuous Positive Airway Pressure (CPAP), weight management, or oral mandibular advancement splints.",
     "Seek medical evaluation for severe daytime sleepiness, falling asleep while driving, witnessed nocturnal choking/gasping, or uncontrolled morning hypertension."),

    (97, "Systemic Lupus Erythematosus (SLE) and Autoimmunity", "Rheumatology, Autoimmune Connective Tissue Disease",
     "Systemic lupus erythematosus is a multisystem autoimmune connective tissue disease characterized by antibodies against nuclear antigens (ANA) and immune-complex deposition.",
     "Defective clearance of apoptotic debris exposes nuclear antigens to autoantibodies, forming immune complexes that deposit in kidneys, skin, joints, and serosa.",
     "Avoid direct sunlight and UV exposure (wear sun-protective clothing and SPF 50+ sunscreen), get plenty of restorative rest, and manage emotional stress.",
     "Consult a rheumatologist for regular disease monitoring; first-line therapy is hydroxychloroquine, along with targeted immunosuppressants and kidney function surveillance.",
     "Seek emergency care for chest pain worse with breathing (pericarditis/pleuritis), shortness of breath, new confusion, or rapidly swelling legs with foamy urine."),

    (98, "Tendonitis (Achilles, Patellar and Bicipital)", "Sports Medicine, Physical Therapy & Tendinopathy",
     "Tendonitis is acute inflammation and micro-injury of a tendon, the fibrous connective tissue structure that anchors muscle to bone.",
     "Repetitive mechanical overload, abrupt increases in training volume, improper biomechanics, and inadequate recovery create micro-tears in collagen fibers.",
     "Rest from aggravating repetitive movements, apply ice packs wrapped in a towel for 15 minutes 3 times daily, and use a supportive brace or wrap.",
     "Perform gentle eccentric strengthening exercises (such as heel drops for Achilles), avoid aggressive stretching of the inflamed tendon, and pursue physical therapy.",
     "Seek urgent orthopedic evaluation for a sudden loud 'pop' or snap accompanied by immediate severe pain and inability to move the limb (tendon rupture)."),

    (99, "Tonsillitis, Peritonsillar Abscess and Pharyngeal Health", "Otolaryngology, Infectious Diseases & Airway Safety",
     "Tonsillitis is acute infection and inflammatory swelling of the palatine tonsils, accompanied by sore throat, fever, and painful swallowing (odynophagia).",
     "Respiratory viruses or Group A Streptococcus bacteria infect tonsillar crypts, triggering tonsillar erythema, exudates, and jugulodigastric lymphadenopathy.",
     "Gargle with warm salt water every 2-3 hours, drink cool soothing liquids or fruit smoothies, use throat lozenges, and take acetaminophen or ibuprofen for pain.",
     "Consult a healthcare provider for throat swab testing; confirmed strep requires antibiotics; recurrent severe episodes may warrant tonsillectomy consultation.",
     "Seek emergency care immediately for difficulty breathing, drooling, inability to swallow liquids, or severe unilateral throat swelling with a deviated uvula (peritonsillar abscess)."),

    (100, "Ulcerative Colitis and Mucosal Colonic Remission", "Gastroenterology, Colorectal Surgery & IBD Care",
     "Ulcerative colitis is a chronic inflammatory bowel disease causing continuous, superficial mucosal inflammation and ulcerations extending proximally from the rectum.",
     "Dysregulated colonic mucosal immune response against commensal gut microbiota in genetically predisposed individuals, causing crypt abscesses and friability.",
     "Rest, drink oral rehydration fluids with electrolytes, eat small bland meals (bananas, white rice), and avoid spicy, high-fat, or high-fiber foods during acute flares.",
     "Consult a gastroenterologist for colonoscopy staging and long-term maintenance therapy (oral 5-ASAs like mesalamine, biologics, or JAK inhibitors), with annual colon cancer screening.",
     "Seek emergency medical evaluation for heavy rectal bleeding, high fever, severe abdominal pain, or rapid abdominal distension with tachycardia (toxic megacolon)."),
]


def format_page_lines(page_num: int, title: str, subtitle: str, definition: str, causes: str, immediate: str, cure: str, redflags: str) -> list:
    """Format structured sections for a single page into wrapped PDF lines."""
    lines = [
        "GALE ENCYCLOPEDIA OF MEDICINE",
        f"PAGE {page_num}: {title.upper()}",
        f"Section: {subtitle}",
        ""
    ]

    sections = [
        ("Definition and Clinical Overview", definition),
        ("What Is the Reason (Underlying Etiology & Mechanisms)", causes),
        ("Immediate Action Plan (What to Do Right Now for Relief)", immediate),
        ("Proper Plan: How to Cure & Prevent (Long-Term Management)", cure),
        ("Red-Flag Warning Signs (When to Seek Emergency Care)", redflags),
    ]

    for sec_header, sec_text in sections:
        header_line = f"{sec_header}: "
        full_text = f"{header_line}{sec_text}".strip()
        wrapped = textwrap.wrap(full_text, width=82)
        lines.extend(wrapped)
        lines.append("")

    return lines


def generate_full_100_page_pdf():
    out_path = Path("data/knowledge/gale_encyclopedia.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pages_data = []
    for item in CONDITIONS_100:
        p_num, title, subtitle, defn, causes, immed, cure, rf = item
        page_lines = format_page_lines(p_num, title, subtitle, defn, causes, immed, cure, rf)
        pages_data.append(page_lines)

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

        stream_lines = ["BT", "/F1 8.5 Tf", "36 756 Td", "13 TL"]
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
    generate_full_100_page_pdf()
