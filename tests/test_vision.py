"""Tests for Multimodal Vision and Modality Combinations (Categories N and O)."""

import pytest
from tools.vision import analyze_image, is_vision_available
from graph import build_mindly_graph


def test_n_vision_execution_with_image(sample_base64_image):
    """Category N: Verify vision analysis function processes image fixture safely."""
    res = analyze_image(
        image_input=sample_base64_image,
        prompt="Describe what you see in this drawing.",
    )
    assert isinstance(res, dict)
    assert "summary" in res
    assert "has_image" in res
    assert res["has_image"] is True
    assert "disclaimer" in res
    # Must include non-clinical disclaimer
    assert "Educational" in res["disclaimer"]


def test_n_vision_fallback_when_model_missing(monkeypatch, sample_base64_image):
    """Category N: Verify graceful guidance when local vision model is not installed."""
    import requests
    def mock_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("Ollama vision model not pulled")

    monkeypatch.setattr(requests, "post", mock_post)
    res = analyze_image(image_input=sample_base64_image)
    assert "ollama pull" in res["summary"].lower() or "vision" in res["summary"].lower()


def test_o_multimodal_combinations_in_graph(simulated_graph, sample_base64_image):
    """Category O: Verify all supported combinations (TEXT, TEXT+IMAGE, VOICE, VOICE+IMAGE) execute smoothly."""
    config = {"configurable": {"thread_id": "test_multimodal_thread"}}

    # 1. TEXT only
    state_text = simulated_graph.invoke({
        "user_input": "Hello Mindly",
        "input_type": "text",
        "messages": [],
    }, config=config)
    assert state_text.get("final_response") is not None
    assert state_text.get("selected_action") == "DIRECT_RESPONSE"

    # 2. TEXT + IMAGE
    state_image = simulated_graph.invoke({
        "user_input": "Look at my art",
        "input_type": "image",
        "image_data": sample_base64_image,
        "messages": [],
    }, config=config)
    assert state_image.get("final_response") is not None
    assert state_image.get("selected_action") == "USE_VISION_TOOL"

    # 3. VOICE
    state_voice = simulated_graph.invoke({
        "user_input": "I am feeling stressed about finals",
        "input_type": "voice",
        "messages": [],
    }, config=config)
    assert state_voice.get("final_response") is not None

    # 4. VOICE + IMAGE
    state_voice_img = simulated_graph.invoke({
        "user_input": "Here is what I drew while feeling anxious",
        "input_type": "voice",
        "image_data": sample_base64_image,
        "messages": [],
    }, config=config)
    assert state_voice_img.get("final_response") is not None
    assert state_voice_img.get("selected_action") == "USE_VISION_TOOL"
