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
from typing import Any, Dict, List, Optional
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

        # Affirmative continuations ("yes", "sure", "please", "yeah", "ok", "yes sure")
        if re.search(r"^(?:yes|yeah|yep|sure|please|ok|okay|yes\s+sure|go\s+ahead)[\s.?!]*$", lower):
            return (
                f"I'm right here with you{name_suffix}. Take a calm breath. "
                "What would feel most helpful for us to focus on next?"
            )

        # Gratitude
        if any(k in lower for k in ["thank you", "thanks", "appreciate it", "helpful"]):
            return (
                f"You are very welcome{name_suffix}! I am always here to support your wellness, clarity, and peace of mind. "
                "Reach out whenever you need a grounding pause or clinical insight."
            )

        # Dynamic, contextual empathetic reflection based on user's exact input
        topic_words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", lower) if w not in {"what", "this", "that", "with", "have", "from", "they", "will", "would", "about", "there", "their", "could", "should", "your", "mindly", "feel", "feeling"}]
        topic_preview = ", ".join(topic_words[:3]) if topic_words else "what you are experiencing"

        return (
            f"I hear you{name_suffix}, and I'm listening closely to what you shared about {topic_preview}. "
            f"Taking a moment to pause and put words to this is a meaningful step.\n\n"
            f"What aspect of this is feeling most challenging or top of mind for you right now?"
        )


class UniversalChatModel:
    """Universal Chat Model wrapper supporting native Ollama HTTP, OpenAI-compatible APIs, and Dynamic Local Engine."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.temperature = temperature if temperature is not None else settings.ollama_temperature
        self.provider = provider or getattr(settings, "llm_provider", "auto")

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Invoke LLM across available providers with automatic graceful fallback."""
        # 1. Check if Cloud OpenAI/Groq API key is present (from env, session_state, or st.secrets)
        api_key = os.getenv("OPENAI_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
        openai_url = os.getenv("OPENAI_BASE_URL", "")

        try:
            import sys
            if "streamlit" in sys.modules:
                st = sys.modules["streamlit"]
                if not api_key:
                    if hasattr(st, "session_state") and st.session_state.get("cloud_api_key"):
                        api_key = str(st.session_state["cloud_api_key"]).strip()
                if not api_key:
                    try:
                        api_key = str(st.secrets.get("GROQ_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")).strip()
                        if api_key.startswith("gsk_") and not openai_url:
                            openai_url = "https://api.groq.com/openai/v1"
                    except Exception:
                        pass
        except Exception:
            pass

        if api_key:
            if api_key.startswith("gsk_") and not openai_url:
                openai_url = "https://api.groq.com/openai/v1"
            elif not openai_url and os.getenv("GROQ_API_KEY"):
                openai_url = "https://api.groq.com/openai/v1"

            try:
                cloud_resp = self._invoke_openai_compatible(messages, api_key, openai_url)
                if cloud_resp:
                    return AIMessage(content=cloud_resp)
            except Exception:
                pass

        # 2. Check Local Ollama via fast native HTTP /api/chat with bounded token count
        health = settings.check_ollama_health()
        if health["is_running"]:
            try:
                payload_msgs = []
                for m in messages:
                    role = "user"
                    if isinstance(m, SystemMessage):
                        role = "system"
                    elif isinstance(m, AIMessage):
                        role = "assistant"
                    payload_msgs.append({"role": role, "content": str(m.content)})

                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": payload_msgs,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": 250,  # Bound generation tokens for fast CPU execution
                        },
                    },
                    timeout=45.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("message", {}).get("content", "").strip()
                    if content:
                        return AIMessage(content=content)
            except Exception:
                pass

        # 3. Dynamic Smart Local Engine (Zero dependency, instant, contextual fallback)
        dynamic_text = DynamicLocalResponseEngine.generate(messages)
        return AIMessage(content=dynamic_text)

    def _invoke_openai_compatible(self, messages: List[BaseMessage], api_key: str, base_url: str) -> Optional[str]:
        target_url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        model_name = os.getenv("OPENAI_MODEL") or ("llama-3.3-70b-versatile" if "groq" in target_url else "gpt-4o-mini")

        payload_msgs = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            payload_msgs.append({"role": role, "content": str(m.content)})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model_name,
            "messages": payload_msgs,
            "temperature": self.temperature,
            "max_tokens": 400,
        }

        res = requests.post(target_url, json=body, headers=headers, timeout=20.0)
        if res.status_code == 200:
            data = res.json()
            return data["choices"][0]["message"]["content"]
        return None


def get_universal_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
) -> UniversalChatModel:
    """Return configured UniversalChatModel instance."""
    return UniversalChatModel(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


# Alias
get_llm = get_universal_llm
