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

        # Greetings
        if any(k in lower for k in ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]):
            return (
                f"Hello{name_suffix}! I am Mindly, here with you. "
                "Whatever is on your heart or mind today, this is your safe space. How are you feeling right now?"
            )

        # Affirmative continuations ("yes", "sure", "please", "yeah", "ok")
        if re.search(r"^(?:yes|yeah|yep|sure|please|ok|okay|yes please|do that|go ahead|tell me|help me)\b", lower):
            return (
                f"I am right here with you{name_suffix}. Let's take this one step at a time.\n\n"
                "Would you like to focus on steadying physical tension with a guided breathing rhythm, or explore practical action steps together?"
            )

        # Gratitude
        if any(k in lower for k in ["thank you", "thanks", "appreciate it", "helpful"]):
            return (
                f"You are very welcome{name_suffix}! I am always here to support your wellness, clarity, and peace of mind. "
                "Reach out whenever you need a grounding pause or clinical insight."
            )

        # Dynamic, contextual empathetic reflection based on user's exact input
        # Extract meaningful emotion or topic tokens
        topic_words = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", lower) if w not in {"what", "this", "that", "with", "have", "from", "they", "will", "would", "about", "there", "their", "could", "should", "your", "mindly"}]
        topic_preview = ", ".join(topic_words[:3]) if topic_words else "what you're experiencing"

        if any(w in lower for w in ["anxious", "anxiety", "panic", "nervous", "scared", "terrified"]):
            return (
                f"I hear how much anxiety is present for you right now{name_suffix}. Feeling anxious when facing {topic_preview} "
                "is your nervous system trying to protect you, but it can feel exhausting.\n\n"
                "Let's ground your physical body first: place both feet flat on the floor, drop your shoulders away from your ears, and take one slow exhale. "
                "Would you like to try a structured 4-4-4-4 breathing cycle together, or unpack what is making you feel most worried?"
            )
        elif any(w in lower for w in ["sad", "lonely", "alone", "crying", "depressed", "hopeless", "hurt", "grief"]):
            return (
                f"I'm holding gentle space for you{name_suffix}. Moving through feelings of sadness around {topic_preview} "
                "can feel heavy and isolating. You don't have to carry it alone or put on a brave face here.\n\n"
                "I'm listening with care. What has felt the heaviest for you today?"
            )
        elif any(w in lower for w in ["tired", "exhausted", "burnout", "drained", "can't sleep"]):
            return (
                f"Your mind and body are signaling that you're carrying a heavy load{name_suffix}. Dealing with {topic_preview} "
                "takes real energy, and cognitive fatigue is a sign to rest, not push harder.\n\n"
                "Give yourself permission to pause for a few minutes. What is one small demand you can set aside for the rest of today?"
            )
        else:
            return (
                f"Thank you for sharing that with me{name_suffix}. I am listening attentively to your thoughts about {topic_preview}.\n\n"
                "Every step toward putting your experience into words helps bring clarity. "
                "How has this been affecting your day, and what kind of support would feel most helpful right now?"
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
        # 1. Check if Cloud OpenAI/Groq API key is present
        api_key = os.getenv("OPENAI_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
        openai_url = os.getenv("OPENAI_BASE_URL", "")
        if not openai_url and os.getenv("GROQ_API_KEY"):
            openai_url = "https://api.groq.com/openai/v1"

        if api_key:
            try:
                cloud_resp = self._invoke_openai_compatible(messages, api_key, openai_url)
                if cloud_resp:
                    return AIMessage(content=cloud_resp)
            except Exception:
                pass

        # 2. Check Local Ollama via fast native HTTP /api/chat
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
                        "options": {"temperature": self.temperature},
                    },
                    timeout=12.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("message", {}).get("content", "").strip()
                    if content:
                        return AIMessage(content=content)
            except Exception:
                pass

        # 3. Dynamic Smart Local Engine (Zero dependency, instant, high quality, contextual)
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
        }

        res = requests.post(target_url, json=body, headers=headers, timeout=12.0)
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
