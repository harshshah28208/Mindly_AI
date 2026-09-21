"""Vision understanding tool for Mindly Lab 3.

Allows users to upload images for educational and situational context analysis.
Gracefully checks for local vision model availability (e.g. llava) without crashing.
Enforces strict boundaries against diagnosing dermatological, radiologic, or medical images.
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional
import requests
from PIL import Image

from config.settings import settings


def analyze_image(
    image_input: Any,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze image using local Ollama vision model if available, otherwise graceful status.

    Args:
        image_input: File path (str/Path), bytes, or PIL Image object.
        prompt: Optional specific query about the image.

    Returns:
        Dict with 'success', 'summary', 'disclaimer', and 'model_used'.
    """
    disclaimer = (
        "\n\n*(Educational Note: Mindly provides conversational perception only and does NOT "
        "diagnose rashes, skin conditions, scans, or medical imagery. Always consult a physician for clinical assessment.)*"
    )

    # Encode image to base64
    b64_str = ""
    try:
        if isinstance(image_input, Path):
            if image_input.exists() and image_input.is_file():
                with open(image_input, "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image_input, str):
            # Check if it's a local file path
            try:
                p = Path(image_input)
                if len(image_input) < 300 and p.exists() and p.is_file():
                    with open(p, "rb") as f:
                        b64_str = base64.b64encode(f.read()).decode("utf-8")
                else:
                    # Direct base64 string (strip potential data URL header)
                    b64_str = image_input.split(",")[-1].strip()
            except Exception:
                b64_str = image_input.split(",")[-1].strip()
        elif isinstance(image_input, bytes):
            b64_str = base64.b64encode(image_input).decode("utf-8")
        elif hasattr(image_input, "read"):
            # UploadedFile or BytesIO
            data = image_input.read()
            b64_str = base64.b64encode(data).decode("utf-8")
            if hasattr(image_input, "seek"):
                image_input.seek(0)
    except Exception as exc:
        return {
            "success": False,
            "has_image": False,
            "summary": f"Could not process image file: {str(exc)}",
            "disclaimer": disclaimer,
            "model_used": "none",
        }

    if not b64_str:
        return {
            "success": False,
            "has_image": False,
            "summary": "No readable image data received.",
            "disclaimer": disclaimer,
            "model_used": "none",
        }

    # Check if vision model is available locally
    vision_health = settings.check_vision_health()
    if not vision_health["vision_available"]:
        return {
            "success": False,
            "has_image": True,
            "summary": (
                f"Image received successfully. However, the local vision model ('{settings.vision_model}') "
                f"is not currently installed in Ollama.\n\n"
                f"To enable local multimodal image analysis on your device, run:\n"
                f"```bash\nollama run {settings.vision_model}\n```"
            ),
            "disclaimer": disclaimer,
            "model_used": "none",
        }

    # If vision model is available, query Ollama vision API
    query_prompt = prompt or "Describe the calming, educational, or notable elements in this image in an empathetic, non-diagnostic manner."
    url = f"{settings.ollama_base_url}/api/generate"
    try:
        payload = {
            "model": settings.vision_model,
            "prompt": query_prompt,
            "images": [b64_str],
            "stream": False,
        }
        res = requests.post(url, json=payload, timeout=90.0)
        if res.status_code == 200:
            content = res.json().get("response", "").strip()
            return {
                "success": True,
                "has_image": True,
                "summary": content + disclaimer,
                "disclaimer": disclaimer,
                "model_used": settings.vision_model,
            }
    except Exception as exc:
        return {
            "success": False,
            "has_image": True,
            "summary": f"Vision model inference timed out or failed: {str(exc)}",
            "disclaimer": disclaimer,
            "model_used": settings.vision_model,
        }

    return {
        "success": False,
        "has_image": True,
        "summary": "Vision model did not return a response.",
        "disclaimer": disclaimer,
        "model_used": settings.vision_model,
    }


def is_vision_available() -> bool:
    """Check whether local Ollama vision model is installed and ready."""
    return bool(settings.check_vision_health().get("vision_available", False))

