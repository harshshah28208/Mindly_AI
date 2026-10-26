"""Centralized configuration management for Mindly.

Reads environment variables from .env using python-dotenv.
Avoids hardcoding models, URLs, or secrets throughout the application.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

# Resolve project root and load .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
else:
    load_dotenv()


class Settings:
    """Application settings class for Mindly."""

    def __init__(self) -> None:
        # LLM & Multi-Provider Settings
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        self.llm_fallback_enabled: bool = os.getenv("LLM_FALLBACK_ENABLED", "true").strip().lower() in ("true", "1", "yes")
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_base_url: str = os.getenv("OPENAI_BASE_URL", "").strip()
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()

        self.ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b").strip()
        self.coping_model: str = os.getenv("COPING_MODEL", self.ollama_model).strip()
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", self.ollama_model).strip()
        self.vision_model: str = os.getenv("VISION_MODEL", "llava").strip()
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        
        try:
            self.ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
        except ValueError:
            self.ollama_temperature = 0.7

        # Configurable Generation Token Budgets (Phase 8: 100-200 for greetings, 300-512 normal, 700-1024 detailed)
        try:
            self.ollama_max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "512"))
        except ValueError:
            self.ollama_max_tokens = 512

        try:
            self.token_budget_greetings: int = int(os.getenv("TOKEN_BUDGET_GREETINGS", "150"))
        except ValueError:
            self.token_budget_greetings = 150

        try:
            self.token_budget_normal: int = int(os.getenv("TOKEN_BUDGET_NORMAL", "512"))
        except ValueError:
            self.token_budget_normal = 512

        try:
            self.token_budget_detailed: int = int(os.getenv("TOKEN_BUDGET_DETAILED", "1024"))
        except ValueError:
            self.token_budget_detailed = 1024

        try:
            self.max_eval_retries: int = int(os.getenv("MAX_EVAL_RETRIES", "2"))
        except ValueError:
            self.max_eval_retries = 2

        # Knowledge & RAG
        self.knowledge_dir: Path = Path(os.getenv("KNOWLEDGE_DIR", str(PROJECT_ROOT / "data" / "knowledge")))
        self.vector_db_dir: Path = Path(os.getenv("VECTOR_DB_DIR", str(PROJECT_ROOT / "data" / "vectorstore")))
        self.gale_pdf_filename: str = os.getenv("GALE_PDF_FILENAME", "gale_encyclopedia.pdf").strip()

        # Database & Memory
        self.db_path: Path = Path(os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "app.db")))

        # Voice & Multimodal
        self.stt_provider: str = os.getenv("STT_PROVIDER", "local").strip()
        self.whisper_model: str = os.getenv("WHISPER_MODEL", "small").strip()
        self.tts_provider: str = os.getenv("TTS_PROVIDER", "local").strip()

        # Human Supervisor Escalation
        self.escalation_mode: str = os.getenv("ESCALATION_MODE", "demo").strip().lower()
        self.supervisor_name: str = os.getenv("SUPERVISOR_NAME", "Project Owner").strip()
        self.supervisor_phone: str = os.getenv("SUPERVISOR_PHONE", "+1-555-0199").strip()

        # UI & App
        self.app_title: str = os.getenv("APP_TITLE", "Mindly - Multimodal Mental-Wellness Companion")
        self.app_name: str = "Mindly"
        self.database_path: Path = self.db_path
        self.audio_dir: Path = self.knowledge_dir.parent / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.escalation_log_file: Path = self.knowledge_dir.parent / "supervisor_escalations.json"

        # Ensure core runtime directories exist
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def gale_pdf_path(self) -> Path:
        """Absolute path to the Gale Encyclopedia of Medicine PDF."""
        return self.knowledge_dir / self.gale_pdf_filename

    def check_ollama_health(self, test_generation: bool = False) -> Dict[str, Any]:
        """Perform a quick health check against the local Ollama instance.
        
        Args:
            test_generation: If True, tests whether configured model can actually generate a token.

        Returns:
            dict containing:
                - is_running (bool): True if Ollama service responds
                - available_models (list[str]): Names of downloaded local models
                - model_installed (bool): True if configured primary model is available locally
                - can_generate (bool): True if model can actively generate completions
                - error_message (str | None): Friendly description if check fails
        """
        tags_url = f"{self.ollama_base_url}/api/tags"
        try:
            response = requests.get(tags_url, timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                models_info = data.get("models", [])
                available_models: List[str] = [m.get("name", "") for m in models_info if m.get("name")]
                
                model_installed = any(
                    self.ollama_model == m or self.ollama_model == m.split(":")[0]
                    for m in available_models
                )
                can_generate = False
                gen_error = None
                if model_installed and test_generation:
                    try:
                        probe = requests.post(
                            f"{self.ollama_base_url}/api/generate",
                            json={"model": self.ollama_model, "prompt": "ping", "stream": False, "options": {"num_predict": 1}},
                            timeout=3.5,
                        )
                        if probe.status_code == 200:
                            can_generate = True
                        else:
                            gen_error = f"Model generation probe returned HTTP {probe.status_code}"
                    except Exception as probe_err:
                        gen_error = f"Model generation probe failed: {str(probe_err)}"
                
                return {
                    "is_running": True,
                    "available_models": available_models,
                    "model_installed": model_installed,
                    "can_generate": can_generate if test_generation else model_installed,
                    "error_message": gen_error,
                }
            else:
                return {
                    "is_running": False,
                    "available_models": [],
                    "model_installed": False,
                    "can_generate": False,
                    "error_message": f"Ollama returned HTTP status {response.status_code}",
                }
        except requests.exceptions.ConnectionError:
            return {
                "is_running": False,
                "available_models": [],
                "model_installed": False,
                "can_generate": False,
                "error_message": (
                    f"Could not connect to Ollama at {self.ollama_base_url}. "
                    "Make sure Ollama is installed and running (`ollama serve`)."
                ),
            }
        except requests.exceptions.Timeout:
            return {
                "is_running": False,
                "available_models": [],
                "model_installed": False,
                "can_generate": False,
                "error_message": f"Connection to Ollama at {self.ollama_base_url} timed out.",
            }
        except Exception as exc:
            return {
                "is_running": False,
                "available_models": [],
                "model_installed": False,
                "can_generate": False,
                "error_message": f"Unexpected error checking Ollama: {str(exc)}",
            }

    def check_vision_health(self) -> Dict[str, Any]:
        """Check if configured vision model is installed locally."""
        health = self.check_ollama_health()
        if not health["is_running"]:
            return {"vision_available": False, "reason": "Ollama service offline"}
        available = health.get("available_models", [])
        has_vision = any(
            self.vision_model == m or self.vision_model == m.split(":")[0]
            or "llava" in m.lower() or "vision" in m.lower()
            for m in available
        )
        return {
            "vision_available": has_vision,
            "configured_model": self.vision_model,
            "reason": None if has_vision else f"Model '{self.vision_model}' not found in Ollama.",
        }

    def determine_token_budget(self, user_text: str, intent: Optional[str] = None) -> int:
        """Dynamically determine token budget based on inquiry depth and intent:
        - Greetings and simple conversation: ~100-200 tokens (token_budget_greetings = 150)
        - Detailed explanations or step-by-step guidance: ~700-1024 tokens (token_budget_detailed = 1024)
        - Normal questions: ~300-512 tokens (token_budget_normal = 512)
        """
        clean = (user_text or "").strip().lower()
        if not clean:
            return self.token_budget_greetings

        # 1. Explicit requests for detailed explanations, multi-step guidance, or clinical health guides (budget: 700-1024)
        detailed_triggers = [
            "detail", "detailed", "in detail", "explain thoroughly", "comprehensive",
            "step by step", "step-by-step", "steps", "walk me through", "deep dive",
            "elaborate", "full plan", "action plan", "list all", "numbered list",
            "break down every step", "detailed explanation", "exercise", "grounding",
            "routine", "protocol", "guide", "how to cure", "how to handle", "what to do",
            "what is the reason", "cure", "treatment",
        ]
        is_step_pattern = bool(re.search(r"\b\d+[-\s]?steps?\b", clean) or re.search(r"\b(?:five|four|three|six|ten)[-\s]?steps?\b", clean))
        if is_step_pattern or any(trig in clean for trig in detailed_triggers):
            return self.token_budget_detailed

        # 2. Greetings and simple conversation (budget: 100-200)
        is_greeting = clean in [
            "hello", "hi", "hey", "how are you", "what is your name", "who are you",
            "good morning", "good evening", "tell me a joke", "thank you", "thanks", "bye", "goodbye"
        ] or (intent == "GENERAL_CONVERSATION" and len(clean.split()) <= 6)
        if is_greeting:
            return self.token_budget_greetings

        # 3. Standard normal questions (budget: 300-512)
        return self.token_budget_normal


# Singleton settings instance
settings = Settings()
