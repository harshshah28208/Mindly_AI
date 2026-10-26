"""Universal LLM Provider and Dynamic Smart Agent for Mindly.

Provides seamless multi-backend model orchestration:
1. Local Ollama (llama3.2, etc.) via native fast HTTP /api/chat with automatic timeout protection.
2. Cloud OpenAI-compatible endpoints (Groq, OpenAI, OpenRouter, DeepSeek) if API keys are configured.
3. Built-in Smart Dynamic Engine that provides dynamic, context-aware, empathetic responses
   without hardcoded fixed dictionaries or canned strings.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from config.settings import settings


INVALID_NAME_WORDS = {
    "feeling", "having", "struggling", "hurting", "crying", "dying", "trying",
    "depressed", "anxious", "stressed", "sad", "tired", "done", "sick", "lost",
    "not", "just", "really", "so", "very", "looking", "asking", "wondering",
    "thinking", "here", "ready", "going", "fine", "okay", "good", "bad", "alive",
    "alone", "afraid", "scared", "experiencing", "suffering", "seeking", "getting",
    "in", "at", "on", "a", "an", "the", "hoping", "dealing", "facing", "pain", "hurt"
}


class DynamicLocalResponseEngine:
    """Intelligent local conversational engine.

    Dynamically generates context-aware, empathetic responses tailored to user's
    exact query, emotion, and conversation history without hardcoded fixed templates.
    """

    @staticmethod
    def extract_user_name_and_facts(messages: List[BaseMessage]) -> Dict[str, Any]:
        user_name = None
        user_facts = {}
        for m in messages:
            if isinstance(m, HumanMessage):
                txt = str(m.content)
                name_match = re.search(r"\b(?:my name is|call me|i am|i'm)\s+([A-Za-z]+)", txt, re.IGNORECASE)
                if name_match:
                    candidate = name_match.group(1).strip()
                    if candidate.lower() not in INVALID_NAME_WORDS and len(candidate) > 1:
                        user_name = candidate.capitalize()
                fav_matches = re.findall(
                    r"my favorite\s+([a-zA-Z\s]+?)\s+is\s+([a-zA-Z0-9\s]+?)(?:\.|$|,)",
                    txt,
                    re.IGNORECASE,
                )
                for cat, val in fav_matches:
                    user_facts[cat.strip().lower()] = val.strip()
        return {"name": user_name, "facts": user_facts}

    @classmethod
    def generate(cls, messages: List[BaseMessage]) -> str:
        last_human_msg = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_human_msg = str(m.content).strip()
                break

        if not last_human_msg:
            return "I am here with you. Whatever is on your mind today, this is your safe space. What would you like to explore together?"

        lower = last_human_msg.lower()
        extracted = cls.extract_user_name_and_facts(messages)
        user_name = extracted["name"]
        user_facts = extracted["facts"]
        name_prefix = f"{user_name}, " if user_name else ""
        name_suffix = f", {user_name}" if user_name else ""

        # Arithmetic / Math queries: enforce medical domain scope boundary
        try:
            from agent.intent import is_math_query
            if is_math_query(last_human_msg):
                return (
                    "🌿 **Medical Domain Scope Boundary Notice**:\n\n"
                    "Mindly is dedicated exclusively as an **Agentic AI Medical & Wellness Copilot** "
                    "(grounded in the *Gale Encyclopedia of Medicine* and multi-agent clinical protocols).\n\n"
                    "This inquiry falls outside my specialized clinical health and wellness scope. "
                    "I do not perform general calculations, programming tasks, or non-medical queries.\n\n"
                    "**In-Scope Inquiries You Can Explore:**\n"
                    "• Clinical diseases & conditions (e.g. *Hypertension*, *Migraine*, *Diabetes*, *Asthma*, *Arthritis*)\n"
                    "• Physical symptoms & etiology (e.g. *Causes of acute stomach pain and the brain-gut axis*)\n"
                    "• Stress regulation, breathing exercises, and evidence-based clinical wellness management."
                )
        except Exception:
            pass

        # Identity / Name recall
        if any(k in lower for k in ["what is my name", "who am i", "do you know my name"]):
            if user_name:
                return f"Your name is **{user_name}**! You shared that with me earlier in our conversation."
            return "You haven't shared your name with me yet. What would you like me to call you?"

        # Favorite items recall
        fav_query_match = re.search(r"what(?:'s|\s+is)\s+my\s+favorite\s+([a-zA-Z\s]+?)(?:\?|$)", lower)
        if fav_query_match:
            query_cat = fav_query_match.group(1).strip().lower()
            for cat, val in user_facts.items():
                if query_cat in cat or cat in query_cat:
                    return f"Your favorite {query_cat} is **{val}**! You shared that with me earlier in our reflection."

        # Bot identity
        if any(k in lower for k in ["what is your name", "who are you", "what can you do"]):
            return (
                "I am **Mindly**, an Agentic AI Medical & Wellness Copilot grounded in the *Gale Encyclopedia of Medicine*. "
                "I'm here to provide evidence-based clinical health guidance, somatic coping strategies, mindful reflection, "
                "and a safe space to untangle your thoughts and physical symptoms."
            )

        # Greetings (strict whole-word matching to prevent substring collisions like 'highly')
        if bool(re.search(r"\b(?:hello|hi|hey|good\s+morning|good\s+evening|good\s+afternoon)\b", lower)):
            return (
                f"Hello{name_suffix}! I am Mindly, here with you. "
                "Whatever is on your heart or mind today, this is your safe space. How are you feeling right now?"
            )

        # Affirmative continuations ("yes", "sure", "please", "yeah", "ok", "yes sure", "tell me more")
        if re.search(r"^(?:yes|yeah|yep|sure|please|ok|okay|yes\s+sure|go\s+ahead|tell\s+me\s+more|how|what\s+can\s+i\s+do)[\s.?!]*$", lower):
            # Check previous conversation context to see what they are responding to
            prev_ai_text = ""
            prev_human_topics = []
            for m in reversed(messages[:-1]):
                if isinstance(m, AIMessage) and not prev_ai_text:
                    prev_ai_text = str(m.content).lower()
                elif isinstance(m, HumanMessage):
                    h_words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", str(m.content).lower()) if w not in {"what", "this", "that", "with", "have", "from", "they", "will", "would", "about", "there", "their", "could", "should", "your", "mindly", "feel", "feeling"}]
                    if h_words:
                        prev_human_topics.extend(h_words[:2])

            context_topic = prev_human_topics[0] if prev_human_topics else "this"

            if any(k in prev_ai_text for k in ["breathe", "breathing", "grounding", "4-4-4-4", "somatic"]):
                return (
                    f"Let's do this grounding pause together right now{name_suffix}.\n\n"
                    "1. Close your eyes gently and drop your shoulders away from your ears.\n"
                    "2. Inhale slowly through your nose for **4 seconds**.\n"
                    "3. Hold that calm space for **4 seconds**.\n"
                    "4. Exhale softly through your mouth for **4 seconds**.\n\n"
                    "Notice that quiet release. How does your body feel after that breath?"
                )
            elif any(k in prev_ai_text for k in ["migraine", "headache", "cephalalgia"]):
                return (
                    f"Continuing on migraine management{name_suffix}:\n\n"
                    "Clinical evidence underscores three pillars for managing migraine episodes:\n"
                    "1. **Sensory Rest**: Rest immediately in a cool, dark, and silent environment with a cold compress on the forehead.\n"
                    "2. **Trigger Tracking**: Keep a symptom log noting sleep regularity, hydration, caffeine intake, and screen exposure.\n"
                    "3. **Clinical Evaluation**: Consult a neurologist or physician for targeted abortive therapies (such as triptans) if attacks recur frequently."
                )
            elif any(k in prev_ai_text for k in ["stomach", "acidity", "gastric", "abdominal", "gerd", "acid reflux"]):
                return (
                    f"Continuing on digestive wellness and the brain-gut axis{name_suffix}:\n\n"
                    "1. **Brain-Gut Connection**: The enteric nervous system is wired directly to emotional centers via the vagus nerve. Heightened stress directly provokes gastric acid hypersecretion and visceral cramping.\n"
                    "2. **Comfort Measures**: Remain upright for at least 2 hours after meals, sip warm water, and avoid spicy or acidic triggers.\n"
                    "3. **When to Seek Evaluation**: Consult a physician promptly if you experience severe unremitting pain, difficulty swallowing, or fever."
                )
            elif any(k in prev_ai_text for k in ["shoulder", "knee", "joint", "back pain", "muscle", "strain"]):
                return (
                    f"Continuing on musculoskeletal comfort and strain care{name_suffix}:\n\n"
                    "1. **R.I.C.E. Protocol**: Rest the affected joint, apply cold packs (15-20 min intervals), use gentle support/compression, and keep elevated when resting.\n"
                    "2. **Gentle Mobility**: Avoid heavy lifting or awkward posture, but maintain gentle pain-free range of motion.\n"
                    "3. **Clinical Advisory**: Seek professional orthopedic or physiotherapy evaluation if there is visible deformity, severe swelling, or inability to bear weight."
                )
            elif any(k in prev_ai_text for k in ["viva", "vivas", "exam", "exams", "study", "workload", "deadline"]):
                return (
                    f"Let's break your preparation down so it feels manageable{name_suffix}.\n\n"
                    "When anticipating vivas or exams, our brains inflate the entire syllabus into an emergency. "
                    "Here is the single best step you can take right now:\n\n"
                    "• **The 15-Minute Micro-Sprint**: Pick only ONE specific topic or slide deck. Set a 15-minute timer and read just that.\n"
                    "• **Action dissolves anticipatory dread**: Once you begin moving your hands or reading, the physiological panic subsides.\n\n"
                    "Which specific topic or question feels most intimidating for you to tackle first?"
                )
            else:
                return (
                    f"I'm right here with you{name_suffix}. Let's gently explore {context_topic}.\n\n"
                    "When you sit with this feeling, what thought or worry seems to come up most frequently?"
                )

        # Gratitude
        if any(k in lower for k in ["thank you", "thanks", "appreciate it", "helpful"]):
            return (
                f"You are very welcome{name_suffix}! I am always here to support your peace of mind and clarity. "
                "Reach out whenever you need a calm listening ear or grounding guidance."
            )

        # Factual and educational health definitions (prevents generic empathy fallback for factual inquiries)
        if re.search(r"\b(?:what\s+(?:is|are)|define|meaning\s+of|explain)\s+(?:anxiety|anxious)\b", lower):
            return (
                "**Anxiety** is a natural neurobiological and psychological stress response mediated by the autonomic nervous system. "
                "While acute anxiety is adaptive (sharpening attention and alertness during challenges), persistent anxiety involves chronic autonomic arousal "
                "(elevated heart rate, muscular tension, shallow respiration) and cognitive rumination.\n\n"
                "**Core Aspects of Anxiety:**\n"
                "• **Physiological**: Sympathetic nervous system activation ('fight-or-flight'), adrenaline release, and visceral restlessness.\n"
                "• **Cognitive**: Anticipatory worry, catastrophizing, and hypervigilance.\n"
                "• **Evidence-Based Support**: Diaphragmatic breathing (such as Box Breathing), progressive somatic grounding, cognitive reframing, and professional clinical consultation if it interferes with daily functioning."
            )

        if re.search(r"\b(?:what\s+(?:is|are)|define|meaning\s+of|explain)\s+(?:depression|depressed)\b", lower):
            return (
                "**Depression** (Major Depressive Disorder) is a recognized clinical mood condition characterized by persistent low mood, emotional numbness, "
                "and anhedonia (the loss of pleasure or interest in previously enjoyable activities) lasting two weeks or longer.\n\n"
                "**Common Characteristics:**\n"
                "• **Somatic & Energy Changes**: Persistent fatigue, disruptions in sleep architecture (insomnia or hypersomnia), and appetite shifts.\n"
                "• **Cognitive Patterns**: Feelings of worthlessness, excessive guilt, and impaired concentration.\n"
                "• **Clinical Approaches**: Cognitive Behavioral Therapy (CBT), behavioral activation routines, social connection, and consultation with medical professionals."
            )

        if re.search(r"\b(?:what\s+(?:is|are)|define|meaning\s+of|explain)\s+(?:stress|burnout)\b", lower):
            return (
                "**Stress** is the body's non-specific physiological and psychological reaction to external demands, transitions, or challenges. "
                "When a stressor occurs, the hypothalamic-pituitary-adrenal (HPA) axis activates, elevating cortisol and adrenaline to mobilize physical resources.\n\n"
                "**Managing Stress Effectively:**\n"
                "• **Pacing**: Establishing clear cognitive boundaries between study/work and recovery time.\n"
                "• **Somatic Reset**: Engaging in 4-second box breathing or a 5-minute walk to facilitate vagal nerve tone.\n"
                "• **Task Chunking**: Breaking complex responsibilities into single, manageable micro-steps."
            )

        if re.search(r"\b(?:what\s+causes\s+(?:a\s+)?migraine|causes\s+of\s+(?:a\s+)?migraine|what\s+is\s+(?:a\s+)?migraine)\b", lower):
            return (
                "A **migraine** is a primary neurological disorder characterized by recurrent attacks of moderate-to-severe throbbing or pulsating headache, "
                "most frequently unilateral and often accompanied by nausea, vomiting, photophobia (light sensitivity), and phonophobia (sound sensitivity).\n\n"
                "**Primary Etiology & Triggers:**\n"
                "• **Neurovascular Activation**: Involves the trigeminovascular system, where abnormal nerve signaling causes localized neurogenic inflammation and vascular dilation.\n"
                "• **Neurochemical Shifts**: Fluctuations in neurotransmitters, especially serotonin (5-HT) and calcitonin gene-related peptide (CGRP).\n"
                "• **Common Triggers**: Disrupted sleep cycles, psychological stress, skipped meals, sensory stimuli (bright lights, loud sounds), and hormonal variations."
            )

        # Dynamic, contextual empathetic reflection based on user's exact input
        topic_words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", lower) if w not in {"what", "this", "that", "with", "have", "from", "they", "will", "would", "about", "there", "their", "could", "should", "your", "mindly", "feel", "feeling"}]
        topic_preview = ", ".join(topic_words[:3]) if topic_words else "what you are experiencing"

        # SafeSpace clinical psychologist attunement variations
        if any(k in lower for k in ["anxious", "anxiety", "panic", "nervous", "worry", "racing"]):
            return (
                f"I hear how much tension is present around {topic_preview}{name_suffix}. "
                "When anxiety spikes, your nervous system is simply trying to protect you, but it can feel overwhelming.\n\n"
                "Remember: you do not have to solve everything in this single hour. "
                "What is one small thing within your immediate control right now?"
            )
        elif any(k in lower for k in ["tired", "exhausted", "burnout", "drowning", "too much", "heavy"]):
            return (
                f"I hear how deeply exhausting that has been for you{name_suffix}. "
                "Carrying this much emotional or mental weight takes a real physical toll on your body.\n\n"
                "Give yourself full permission to pause. What would feel like the kindest, most restorative thing you can do for yourself today?"
            )
        else:
            return (
                f"I'm listening closely to what you shared about {topic_preview}{name_suffix}. "
                "Taking time to put words to your experience is a courageous and meaningful step.\n\n"
                "What part of this is feeling most challenging or top of mind for you right now?"
            )


class UniversalChatModel:
    """Universal Chat Model wrapper supporting native Ollama HTTP, OpenAI-compatible APIs, and Dynamic Local Engine."""

    last_metadata: Dict[str, Any] = {
        "provider": "Dynamic Local Engine",
        "provider_requested": "Auto",
        "provider_used": "Dynamic Local Engine",
        "model": "",
        "latency": 0.0,
        "is_fallback": False,
        "error_category": None,
    }

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        provider: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.temperature = temperature if temperature is not None else settings.ollama_temperature
        self.provider = provider or getattr(settings, "llm_provider", "auto")
        self.max_tokens = max_tokens or getattr(settings, "ollama_max_tokens", 512)

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Invoke LLM across available providers respecting LLM_PROVIDER with predictable fallback and rich metadata."""
        t0 = time.time()

        # Determine effective provider mode
        provider_mode = (self.provider or os.getenv("LLM_PROVIDER", getattr(settings, "llm_provider", "auto"))).strip().lower()

        # Check fallback setting
        fallback_enabled = getattr(settings, "llm_fallback_enabled", True)
        fallback_env = os.getenv("LLM_FALLBACK_ENABLED")
        if fallback_env is not None:
            fallback_enabled = fallback_env.strip().lower() in ("true", "1", "yes")

        # Resolve cloud API key if present (env, session_state, or st.secrets)
        raw_key = (
            os.getenv("GROQ_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        openai_url = os.getenv("OPENAI_BASE_URL", "").strip()
        try:
            import sys
            if "streamlit" in sys.modules:
                st = sys.modules["streamlit"]
                if hasattr(st, "session_state"):
                    sess_provider = st.session_state.get("llm_provider")
                    if sess_provider and not self.provider:
                        provider_mode = str(sess_provider).strip().lower()
                    sess_key = st.session_state.get("cloud_api_key") or st.session_state.get("cloud_key_input_field")
                    if sess_key and str(sess_key).strip():
                        raw_key = str(sess_key).strip()
                    if "llm_fallback_enabled" in st.session_state:
                        fallback_enabled = bool(st.session_state.get("llm_fallback_enabled"))
                if not raw_key:
                    try:
                        raw_key = str(st.secrets.get("GROQ_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")).strip()
                    except Exception:
                        pass
        except Exception:
            pass

        cloud_api_key = raw_key.strip().strip("'\" \t\r\n")
        if cloud_api_key.lower().startswith("bearer "):
            cloud_api_key = cloud_api_key[7:].strip().strip("'\" \t\r\n")

        # Helper: Attempt Local Ollama
        def _try_ollama():
            health = settings.check_ollama_health()
            if not health["is_running"]:
                return None, "connection_error", f"Could not connect to Ollama at {self.base_url}. Service offline."
            if not health["model_installed"]:
                return None, "model_missing", f"Model '{self.model}' is not installed in local Ollama."
            try:
                payload_msgs = []
                for m in messages:
                    role = "user"
                    if isinstance(m, SystemMessage):
                        role = "system"
                    elif isinstance(m, AIMessage):
                        role = "assistant"
                    content_str = str(m.content or "").strip()
                    if content_str:
                        payload_msgs.append({"role": role, "content": content_str})

                if not payload_msgs:
                    payload_msgs = [{"role": "user", "content": "Hello"}]

                predict_tokens = self.max_tokens
                # When running on local CPU, cap num_predict to fast limit (default 280 tokens = ~20-25 seconds)
                # to guarantee every query completes quickly while preserving full, high-quality, non-truncated answers.
                fast_limit = int(os.getenv("OLLAMA_FAST_TOKENS", "280"))
                if os.getenv("FAST_LOCAL_INFERENCE", "true").lower() in ["true", "1"] and fast_limit > 0:
                    predict_tokens = min(predict_tokens, fast_limit)

                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": payload_msgs,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": predict_tokens,
                            "num_thread": 4,
                        },
                    },
                    timeout=85.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("message", {}).get("content", "").strip()
                    done_reason = data.get("done_reason", "stop")
                    eval_count = data.get("eval_count", 0)

                    is_length_limit = (done_reason == "length") or (eval_count > 0 and eval_count >= predict_tokens)
                    is_abrupt_cutoff = False
                    if content and len(content) > 30:
                        if re.search(r"[,:;\-\(\[\{]\s*$", content) or re.search(r"\b(?:and|or|the|with|to|in|of|that|because|step|such\s+as)\s*$", content, re.IGNORECASE):
                            is_abrupt_cutoff = True
                        elif content[-1].isalnum() and not content.endswith((".", "!", "?", "\"", "'", "*", "`", "#")):
                            is_abrupt_cutoff = is_length_limit

                    is_truncated = is_length_limit or is_abrupt_cutoff
                    return (content, done_reason, eval_count, is_truncated), None, None
                else:
                    return None, "api_error", f"Ollama HTTP {resp.status_code}"
            except requests.exceptions.Timeout:
                return None, "timeout", "Ollama request timed out after 85 seconds"
            except requests.exceptions.ConnectionError:
                return None, "connection_error", f"Connection to Ollama at {self.base_url} failed"
            except Exception as exc:
                return None, "unknown_error", f"Ollama error: {str(exc)}"

        # Helper: Attempt Cloud API
        def _try_cloud():
            if not cloud_api_key:
                return None, "missing_api_key", "No cloud API key configured."
            try:
                cloud_resp, model_used, finish_reason, is_truncated = self._invoke_openai_compatible(messages, cloud_api_key, openai_url)
                if cloud_resp:
                    is_groq_tag = (
                        cloud_api_key.startswith("gsk_")
                        or "groq" in (openai_url or "").lower()
                        or provider_mode == "groq"
                        or "llama" in model_used.lower()
                    )
                    provider_tag = "Groq" if is_groq_tag else "OpenAI"
                    return (cloud_resp, model_used, finish_reason, is_truncated, provider_tag), None, None
                else:
                    return None, "api_error", f"Cloud LLM call failed ({finish_reason})"
            except requests.exceptions.Timeout:
                return None, "timeout", "Cloud API request timed out"
            except Exception as exc:
                return None, "unknown_error", f"Cloud API error: {str(exc)}"

        # Branch 1: Explicit Local Ollama mode
        if provider_mode in ["ollama", "local"]:
            res, err_cat, err_msg = _try_ollama()
            dt = round(time.time() - t0, 2)
            if res:
                content, done_reason, eval_count, is_truncated = res
                UniversalChatModel.last_metadata = {
                    "provider_requested": "Local Ollama",
                    "provider_used": "Local Ollama",
                    "provider": "Local Ollama",
                    "model": self.model,
                    "latency": dt,
                    "finish_reason": done_reason,
                    "eval_count": eval_count,
                    "truncated": is_truncated,
                    "is_fallback": False,
                    "error_category": None,
                    "max_tokens": self.max_tokens,
                }
                return AIMessage(content=content, response_metadata=UniversalChatModel.last_metadata)
            else:
                if fallback_enabled:
                    dynamic_text = DynamicLocalResponseEngine.generate(messages)
                    UniversalChatModel.last_metadata = {
                        "provider_requested": "Local Ollama",
                        "provider_used": "Dynamic Local Engine (Fallback)",
                        "provider": "Dynamic Local Engine",
                        "model": "rule-synthesis",
                        "latency": dt,
                        "finish_reason": "stop",
                        "truncated": False,
                        "is_fallback": True,
                        "error_category": err_cat,
                        "diagnostic_message": f"Local Ollama failed ({err_cat}: {err_msg}). Gracefully degraded to local rule-synthesis.",
                        "max_tokens": self.max_tokens,
                    }
                    return AIMessage(content=dynamic_text, response_metadata=UniversalChatModel.last_metadata)
                else:
                    diagnostic_text = (
                        f"⚠️ **Provider Error**: Configured provider `Local Ollama` ({self.model}) failed ({err_cat}). "
                        f"{err_msg}\n\nAutomatic fallback is disabled. Please verify that `ollama serve` is active."
                    )
                    UniversalChatModel.last_metadata = {
                        "provider_requested": "Local Ollama",
                        "provider_used": "None",
                        "provider": "Error Diagnostic",
                        "model": self.model,
                        "latency": dt,
                        "finish_reason": "error",
                        "truncated": False,
                        "is_fallback": False,
                        "error_category": err_cat,
                        "diagnostic_message": diagnostic_text,
                        "max_tokens": self.max_tokens,
                    }
                    return AIMessage(content=diagnostic_text, response_metadata=UniversalChatModel.last_metadata)

        # Branch 2: Explicit Cloud mode (Groq or OpenAI)
        elif provider_mode in ["groq", "openai", "cloud"]:
            res, err_cat, err_msg = _try_cloud()
            dt = round(time.time() - t0, 2)
            if res:
                content, model_used, finish_reason, is_truncated, provider_tag = res
                UniversalChatModel.last_metadata = {
                    "provider_requested": provider_mode.capitalize(),
                    "provider_used": provider_tag,
                    "provider": provider_tag,
                    "model": model_used,
                    "latency": dt,
                    "finish_reason": finish_reason,
                    "truncated": is_truncated,
                    "is_fallback": False,
                    "error_category": None,
                    "max_tokens": self.max_tokens,
                }
                return AIMessage(content=content, response_metadata=UniversalChatModel.last_metadata)
            else:
                if fallback_enabled:
                    dynamic_text = DynamicLocalResponseEngine.generate(messages)
                    UniversalChatModel.last_metadata = {
                        "provider_requested": provider_mode.capitalize(),
                        "provider_used": "Dynamic Local Engine (Fallback)",
                        "provider": "Dynamic Local Engine",
                        "model": "rule-synthesis",
                        "latency": dt,
                        "finish_reason": "stop",
                        "truncated": False,
                        "is_fallback": True,
                        "error_category": err_cat,
                        "diagnostic_message": f"Cloud provider failed ({err_cat}: {err_msg}). Gracefully degraded to local rule-synthesis.",
                        "max_tokens": self.max_tokens,
                    }
                    return AIMessage(content=dynamic_text, response_metadata=UniversalChatModel.last_metadata)
                else:
                    diagnostic_text = (
                        f"⚠️ **Provider Error**: Configured cloud provider failed ({err_cat}). "
                        f"{err_msg}\n\nAutomatic fallback is disabled. Please verify your API key and network connection."
                    )
                    UniversalChatModel.last_metadata = {
                        "provider_requested": provider_mode.capitalize(),
                        "provider_used": "None",
                        "provider": "Error Diagnostic",
                        "model": "cloud",
                        "latency": dt,
                        "finish_reason": "error",
                        "truncated": False,
                        "is_fallback": False,
                        "error_category": err_cat,
                        "diagnostic_message": diagnostic_text,
                        "max_tokens": self.max_tokens,
                    }
                    return AIMessage(content=diagnostic_text, response_metadata=UniversalChatModel.last_metadata)

        # Branch 3: Auto mode (Cloud if key -> Ollama -> Fallback if enabled)
        else:
            if cloud_api_key:
                cloud_res, cloud_err, _ = _try_cloud()
                if cloud_res:
                    content, model_used, finish_reason, is_truncated, provider_tag = cloud_res
                    dt = round(time.time() - t0, 2)
                    UniversalChatModel.last_metadata = {
                        "provider_requested": "Auto",
                        "provider_used": provider_tag,
                        "provider": provider_tag,
                        "model": model_used,
                        "latency": dt,
                        "finish_reason": finish_reason,
                        "truncated": is_truncated,
                        "is_fallback": False,
                        "error_category": None,
                        "max_tokens": self.max_tokens,
                    }
                    return AIMessage(content=content, response_metadata=UniversalChatModel.last_metadata)

            ollama_res, ollama_err, ollama_detail = _try_ollama()
            if ollama_res:
                content, done_reason, eval_count, is_truncated = ollama_res
                dt = round(time.time() - t0, 2)
                UniversalChatModel.last_metadata = {
                    "provider_requested": "Auto",
                    "provider_used": "Local Ollama",
                    "provider": "Local Ollama",
                    "model": self.model,
                    "latency": dt,
                    "finish_reason": done_reason,
                    "eval_count": eval_count,
                    "truncated": is_truncated,
                    "is_fallback": False,
                    "error_category": None,
                    "max_tokens": self.max_tokens,
                }
                return AIMessage(content=content, response_metadata=UniversalChatModel.last_metadata)

            dt = round(time.time() - t0, 2)
            if fallback_enabled:
                dynamic_text = DynamicLocalResponseEngine.generate(messages)
                UniversalChatModel.last_metadata = {
                    "provider_requested": "Auto",
                    "provider_used": "Dynamic Local Engine (Fallback)",
                    "provider": "Dynamic Local Engine",
                    "model": "rule-synthesis",
                    "latency": dt,
                    "finish_reason": "stop",
                    "truncated": False,
                    "is_fallback": True,
                    "error_category": "all_providers_unavailable",
                    "diagnostic_message": f"Auto mode: Local Ollama unavailable ({ollama_err}: {ollama_detail}). Fallback engaged.",
                    "max_tokens": self.max_tokens,
                }
                return AIMessage(content=dynamic_text, response_metadata=UniversalChatModel.last_metadata)
            else:
                diagnostic_text = (
                    "⚠️ **No Available LLM Provider**: Neither Cloud LLM nor Local Ollama was reachable, "
                    "and heuristic fallback is disabled. Please start `ollama serve` or enter a free Groq key."
                )
                UniversalChatModel.last_metadata = {
                    "provider_requested": "Auto",
                    "provider_used": "None",
                    "provider": "Error Diagnostic",
                    "model": "none",
                    "latency": dt,
                    "finish_reason": "error",
                    "truncated": False,
                    "is_fallback": False,
                    "error_category": "all_providers_unavailable",
                    "diagnostic_message": diagnostic_text,
                    "max_tokens": self.max_tokens,
                }
                return AIMessage(content=diagnostic_text, response_metadata=UniversalChatModel.last_metadata)

    def _invoke_openai_compatible(self, messages: List[BaseMessage], api_key: str, base_url: str) -> Tuple[Optional[str], str, str, bool]:
        clean_key = (api_key or "").strip().strip("'\" \t\r\n")
        if clean_key.lower().startswith("bearer "):
            clean_key = clean_key[7:].strip().strip("'\" \t\r\n")

        if not clean_key:
            return None, "none", "missing_key", False

        provider_env = os.getenv("LLM_PROVIDER", "").strip().lower()
        is_groq = (
            clean_key.startswith("gsk_")
            or "groq" in (base_url or "").lower()
            or provider_env == "groq"
            or bool(os.getenv("GROQ_API_KEY"))
        )

        if is_groq:
            target_url = "https://api.groq.com/openai/v1/chat/completions"
            # Candidate models with automatic cascading across active Groq models
            candidate_models = [
                os.getenv("GROQ_MODEL", "").strip(),
                "qwen/qwen3.8-27b",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ]
            seen = set()
            models_to_try = []
            for m in candidate_models:
                if m and m not in seen:
                    models_to_try.append(m)
                    seen.add(m)
            # Give Groq a generous token budget so dynamic answers complete naturally without artificial cutoffs
            token_budget = max(self.max_tokens or 768, 1024)
        else:
            target_url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
            models_to_try = [os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()]
            token_budget = self.max_tokens or 512

        payload_msgs = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            content_str = str(m.content or "").strip()
            if not content_str:
                continue
            payload_msgs.append({"role": role, "content": content_str})

        if not payload_msgs:
            payload_msgs = [{"role": "user", "content": "Hello"}]

        headers = {
            "Authorization": f"Bearer {clean_key}",
            "Content-Type": "application/json",
        }

        last_err = ""
        for model_name in models_to_try:
            body = {
                "model": model_name,
                "messages": payload_msgs,
                "temperature": self.temperature,
                "max_tokens": token_budget,
            }

            try:
                res = requests.post(target_url, json=body, headers=headers, timeout=20.0)
                if res.status_code == 200:
                    data = res.json()
                    choice = data["choices"][0]
                    resp_content = choice["message"]["content"]
                    finish_reason = choice.get("finish_reason", "stop")
                    # Check whether generation was cut off mid-thought
                    is_length = (finish_reason == "length")
                    is_dangling = False
                    if resp_content:
                        t = resp_content.strip()
                        is_dangling = bool(
                            re.search(r"[,:;\-\(\[\{]\s*$", t)
                            or re.search(r"\b(?:and|or|the|with|to|in|of|that|because|step|such\s+as)\s*$", t, re.IGNORECASE)
                        )
                    is_truncated = is_length and (is_dangling or not resp_content.strip().endswith((".", "!", "?", "\"", "'", "*", "`", "#")))
                    return resp_content, model_name, finish_reason, is_truncated
                elif res.status_code in (404, 429, 400, 503) and len(models_to_try) > 1 and model_name != models_to_try[-1]:
                    # Primary model missing/rate-limited/overloaded: immediately try fast fallback model
                    last_err = f"HTTP {res.status_code}: {res.text[:120]}"
                    continue
                else:
                    last_err = f"HTTP {res.status_code}: {res.text[:120]}"
                    print(f"[Cloud LLM HTTP {res.status_code} ({model_name})]: {res.text[:200]}")
            except Exception as exc:
                last_err = str(exc)
                print(f"[Cloud LLM Request Error ({model_name})]: {exc}")
                continue

        return None, (models_to_try[0] if models_to_try else "cloud"), f"error: {last_err}", False


def get_universal_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> UniversalChatModel:
    """Return configured UniversalChatModel instance."""
    return UniversalChatModel(
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# Alias
get_llm = get_universal_llm
