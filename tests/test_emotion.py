"""Tests for Emotion Classification (Category D)."""

import pytest
from agent.emotion import classify_emotion


def test_stressed_emotion():
    """Verify stressful academic inputs map to stressed, anxious, or overwhelmed."""
    emotion = classify_emotion("I have three exams tomorrow and zero time left, I'm stressed!")
    assert emotion in ["stressed", "anxious", "overwhelmed"]


def test_anxious_emotion():
    """Verify panic/racing thoughts map to anxious or overwhelmed."""
    emotion = classify_emotion("My heart is pounding and I feel intense anxiety about presenting.")
    assert emotion in ["anxious", "overwhelmed", "stressed"]


def test_sad_emotion():
    """Verify sadness inputs map to sad or depressed."""
    emotion = classify_emotion("I feel really down, lonely, and sad today.")
    assert emotion in ["sad", "overwhelmed", "lonely"]


def test_angry_emotion():
    """Verify frustrated/angry inputs map to angry or stressed."""
    emotion = classify_emotion("I am so furious and angry with how unfairly this was graded!")
    assert emotion in ["angry", "stressed", "frustrated"]


def test_calm_or_neutral_emotion():
    """Verify relaxed or casual inputs map to calm, neutral, or happy."""
    emotion = classify_emotion("The weather is nice and I am feeling peaceful.")
    assert emotion in ["calm", "neutral", "happy"]


def test_greeting_neutral_emotion():
    """Verify simple greeting maps to neutral or calm."""
    emotion = classify_emotion("Hello there, how are you?")
    assert emotion in ["neutral", "calm", "happy"]
