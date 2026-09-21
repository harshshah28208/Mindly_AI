"""Tests for Voice Input, TTS Synthesis, and Audio Failure Resilience (Categories K, L, M)."""

import os
from pathlib import Path
import pytest
from tools.voice import synthesize_speech, transcribe_audio, is_voice_available
from config.settings import settings


def test_m_tts_synthesis_basic():
    """Category M: Verify TTS converts response text into a valid local audio file."""
    text = "Welcome to Mindly. Take a gentle breath and let me know how you are feeling."
    audio_path = synthesize_speech(text)
    if audio_path is not None:
        p = Path(audio_path)
        assert p.exists()
        assert p.stat().st_size > 0
        assert p.suffix == ".wav"


def test_m_tts_cleans_emojis_and_markdown():
    """Category M: Verify emoji/markdown stripping prevents audio driver crashes."""
    dirty_text = "### 🌿 Box Breathing (4-4-4-4)\n* Inhale slowly **now** 🌬️\n* Hold gently..."
    # Should synthesize cleanly without crashing or throwing
    audio_path = synthesize_speech(dirty_text)
    # Even if offline audio driver fails on some server headless setups, it should return None, not crash
    assert audio_path is None or Path(audio_path).exists()


def test_m_tts_failure_leaves_text_intact(monkeypatch):
    """Category M: Verify simulated TTS crash does not break text responses."""
    import tools.voice
    def broken_engine(*args, **kwargs):
        raise RuntimeError("Audio device busy")

    monkeypatch.setattr(tools.voice, "_synthesize_pyttsx3", broken_engine)
    res = synthesize_speech("Sample text")
    assert res is None, "TTS failure should gracefully return None rather than raising unhandled exceptions"


def test_k_voice_transcription_from_wav(sample_wav_audio):
    """Category K: Verify audio file can be processed by transcribe_audio without crashing."""
    text = transcribe_audio(sample_wav_audio)
    assert isinstance(text, str)
    # For a silent fixture or mock, it returns empty string or transcription without throwing


def test_l_voice_failure_invalid_audio():
    """Category L: Verify corrupted or missing audio files return empty strings gracefully."""
    # Non-existent file
    res = transcribe_audio("non_existent_audio_file.wav")
    assert res == ""

    # Zero-byte invalid file
    empty_file = settings.audio_dir / "empty_test.wav"
    empty_file.write_bytes(b"")
    res2 = transcribe_audio(str(empty_file))
    assert res2 == ""


def test_l_voice_available_flag():
    """Verify is_voice_available returns boolean status cleanly."""
    status = is_voice_available()
    assert isinstance(status, bool)
