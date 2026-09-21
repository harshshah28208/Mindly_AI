"""Pytest configuration and shared fixtures for Mindly Lab 3."""

import base64
import os
import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from graph import build_mindly_graph
from langgraph.checkpoint.memory import MemorySaver


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def simulated_graph():
    """Graph configured to run with deterministic simulation without external LLM latency."""
    memory = MemorySaver()
    return build_mindly_graph(checkpointer=memory, use_simulation=True)


@pytest.fixture
def clean_test_user():
    """Unique test user ID to avoid pollution in persistent SQLite database."""
    import uuid
    return f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_base64_image() -> str:
    """Minimal valid 1x1 transparent PNG encoded in base64."""
    # 1x1 PNG bytes
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02"
        b"\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base64.b64encode(png_bytes).decode("ascii")


@pytest.fixture
def sample_wav_audio(tmp_path) -> str:
    """Generate a minimal valid 0.5-second silent PCM 16-bit 16kHz WAV file."""
    import wave
    import struct
    
    wav_path = tmp_path / "test_sample.wav"
    with wave.open(str(wav_path), "w") as wav_file:
        wav_file.setnchannels(1)       # Mono
        wav_file.setsampwidth(2)       # 16-bit
        wav_file.setframerate(16000)   # 16kHz
        # 0.25 seconds of silence
        silence_frames = [0] * int(16000 * 0.25)
        packed_data = struct.pack(f"<{len(silence_frames)}h", *silence_frames)
        wav_file.writeframes(packed_data)
    return str(wav_path)
