"""Dedicated test suite for disease image analysis, multi-run stability, and mic mode."""

import io
import struct
import wave
from PIL import Image
import pytest

from tools.vision import analyze_image, is_vision_available
from tools.voice import transcribe_audio, is_voice_available
from graph import build_mindly_graph, send_message
from langgraph.checkpoint.memory import MemorySaver


def test_disease_image_erythema_detection():
    """Verify that an image presenting acute cutaneous redness triggers erythematous pattern analysis."""
    # Create an image simulating an acute red erythematous skin rash (120x120 RGB with high red channel)
    im = Image.new("RGB", (120, 120), color=(220, 48, 42))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    res = analyze_image(img_bytes, prompt="What is this red rash on my arm?")
    assert res["success"] is True
    assert res["has_image"] is True
    summary = res["summary"]

    # 1. Morphological & Visual Observation
    assert "Vision Perception & Morphological Inspection" in summary
    assert "erythema" in summary.lower() or "inflammatory" in summary.lower()

    # 2. Gale Encyclopedia Grounding (Page 10)
    assert "Gale Encyclopedia of Medicine" in summary
    assert "Page 10" in summary
    assert "Contact Dermatitis" in summary or "Eczema" in summary

    # 3. Non-Pharmacological Comfort Steps
    assert "Comfort Steps" in summary or "compress" in summary.lower()

    # 4. Red-Flag Warning Signs
    assert "Warning Signs" in summary or "Red-Flag" in summary

    # 5. Educational Disclaimer
    assert "Educational Note" in summary


def test_vision_multiple_consecutive_runs_no_loop():
    """Verify that vision analysis can be called multiple consecutive times without state pollution or infinite loops."""
    im = Image.new("RGB", (80, 80), color=(180, 60, 60))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    for idx in range(3):
        res = analyze_image(img_bytes, prompt=f"Clinical inspection iteration {idx + 1}")
        assert res["success"] is True
        assert res["has_image"] is True
        assert len(res["summary"]) > 200
        assert "Gale Encyclopedia of Medicine" in res["summary"]


def test_mic_mode_silent_audio_returns_empty():
    """Verify that recording silence or background noise produces an empty string rather than hallucinating prompts."""
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        silence_frames = [0] * int(16000 * 0.3)  # 0.3s of PCM silence
        wf.writeframes(struct.pack(f"<{len(silence_frames)}h", *silence_frames))
    
    transcript = transcribe_audio(buf.getvalue())
    assert transcript == "", "Silent audio must return empty string to prevent hallucinated user input"


def test_mic_mode_empty_or_none():
    """Verify that empty, None, or zero-byte audio returns empty string."""
    assert transcribe_audio(None) == ""
    assert transcribe_audio(b"") == ""
    assert transcribe_audio("non_existent_file.wav") == ""


def test_full_graph_with_disease_image():
    """Verify end-to-end LangGraph StateGraph execution with a disease image input."""
    memory = MemorySaver()
    graph = build_mindly_graph(checkpointer=memory, use_simulation=True)

    im = Image.new("RGB", (90, 90), color=(210, 50, 50))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    res = send_message(
        user_text="What is this skin rash?",
        thread_id="test_disease_graph_thread",
        input_type="image",
        image_data=img_bytes,
        graph=graph,
        use_simulation=True,
    )
    assert res["action"] == "USE_VISION_TOOL"
    assert res["tool"] == "vision_analyzer"
    assert res["evaluation"] == "PASS"
    assert "Gale Encyclopedia of Medicine" in res["response"]
    assert "Page 10" in res["response"]
