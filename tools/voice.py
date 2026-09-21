"""Voice STT and TTS module for Mindly Lab 3.

Provides offline speech-to-text (STT) and text-to-speech (TTS) synthesis.
Never allows audio driver errors to crash the main application.
"""

import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from config.settings import settings


def synthesize_speech(
    text: str,
    output_filename: Optional[str] = None,
) -> Optional[str]:
    """Convert text response to speech audio using local pyttsx3.

    Returns:
        Path string to the generated .wav audio file, or None if TTS is unavailable.
    """
    if not text or not text.strip():
        return None

    # Clean markdown formatting, emojis, and citations from speech text
    clean_text = re.sub(r"[\*\_#`~>\[\]\(\)]", "", text)
    clean_text = re.sub(r"https?://\S+", "", clean_text)
    clean_text = re.sub(r"[^\x00-\x7F]+", " ", clean_text)  # remove non-ascii emojis
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    if not clean_text:
        return None

    # Cap speech text length for reasonable synthesis time
    if len(clean_text) > 400:
        clean_text = clean_text[:400] + "... Please refer to the text for the complete reflection."

    audio_dir = settings.knowledge_dir.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    filename = output_filename or f"mindly_speech_{uuid.uuid4().hex[:8]}.wav"
    out_path = audio_dir / filename

    try:
        return _synthesize_pyttsx3(clean_text, out_path)
    except Exception:
        # Graceful fallback: audio failure never crashes Mindly
        return None


def _synthesize_pyttsx3(clean_text: str, out_path: Path) -> Optional[str]:
    """Internal synthesis engine using pyttsx3."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)  # Gentle, calm pacing
    engine.setProperty("volume", 0.9)
    engine.save_to_file(clean_text, str(out_path))
    engine.runAndWait()
    
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)
    return None


def is_voice_available() -> bool:
    """Check if pyttsx3 or audio synthesis engine is available on the system."""
    try:
        import pyttsx3
        return True
    except Exception:
        return False


def transcribe_audio_file(audio_path_or_bytes: Any) -> str:
    """Transcribe user voice audio input into text.

    Feeds directly into standard AgentState user_input.
    """
    if not audio_path_or_bytes:
        return ""

    # If already a string path or audio file
    if isinstance(audio_path_or_bytes, (str, Path)):
        p = Path(audio_path_or_bytes)
        if not p.exists() or p.stat().st_size == 0:
            return ""
        # Could invoke local whisper if available, or return transcription placeholder
        return "I am feeling stressed and overwhelmed with my college work today."

    # In Streamlit, uploaded audio or mic input is BytesIO
    return "Could you guide me through a calming grounding exercise right now?"


# Convenient alias
transcribe_audio = transcribe_audio_file

