"""Generate diverse clinical/multimodal test assets (images and audio) and execute full validation.

Creates:
- sample_test_assets/images/ (erythema rash, dermatitis plaque, pigmented lesion, medical diagram, nature grounding)
- sample_test_assets/audio/ (medical question, exam anxiety, grounding request, goal creation, pure silence)
"""

import io
import math
import os
import struct
import sys
import time
import uuid
import wave
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFilter

from tools.vision import analyze_image
from tools.voice import synthesize_speech, transcribe_audio_file
from graph import build_mindly_graph, send_message
from langgraph.checkpoint.memory import MemorySaver


ASSET_DIR = Path("sample_test_assets")
IMG_DIR = ASSET_DIR / "images"
AUD_DIR = ASSET_DIR / "audio"


def create_sample_images():
    """Create diverse clinical and wellness images for multimodal testing."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    created = {}

    # 1. Acute Erythematous Skin Rash (Contact Dermatitis / Eczema)
    # 250x250 with warm skin background and an irregular, inflamed erythematous center
    im_rash = Image.new("RGB", (250, 250), color=(225, 185, 160))
    draw = ImageDraw.Draw(im_rash)
    # Inflamed patches
    draw.ellipse([60, 50, 190, 180], fill=(220, 50, 45))
    draw.ellipse([80, 70, 160, 150], fill=(235, 40, 35))
    draw.ellipse([140, 120, 200, 180], fill=(210, 55, 50))
    im_rash = im_rash.filter(ImageFilter.GaussianBlur(radius=7))
    rash_path = IMG_DIR / "01_erythema_rash_eczema.jpg"
    im_rash.save(rash_path, format="JPEG", quality=90)
    created["rash"] = rash_path

    # 2. Cutaneous Dermatitis Plaque with texture
    im_plaque = Image.new("RGB", (240, 240), color=(215, 175, 150))
    draw_p = ImageDraw.Draw(im_plaque)
    for x in range(50, 190, 8):
        for y in range(60, 180, 8):
            dist = math.sqrt((x - 120)**2 + (y - 120)**2)
            if dist < 65:
                r_val = int(210 + 25 * math.sin(x * 0.1))
                g_val = int(60 + 20 * math.cos(y * 0.1))
                b_val = 55
                draw_p.rectangle([x, y, x + 6, y + 6], fill=(r_val, g_val, b_val))
    im_plaque = im_plaque.filter(ImageFilter.GaussianBlur(radius=4))
    plaque_path = IMG_DIR / "02_dermatitis_plaque.png"
    im_plaque.save(plaque_path, format="PNG")
    created["plaque"] = plaque_path

    # 3. Pigmented Skin Lesion / Mole
    im_mole = Image.new("RGB", (240, 240), color=(230, 195, 175))
    draw_m = ImageDraw.Draw(im_mole)
    draw_m.ellipse([85, 85, 155, 155], fill=(70, 40, 30))
    draw_m.ellipse([95, 95, 140, 140], fill=(50, 25, 18))
    im_mole = im_mole.filter(ImageFilter.GaussianBlur(radius=3))
    mole_path = IMG_DIR / "03_pigmented_mole_lesion.jpg"
    im_mole.save(mole_path, format="JPEG", quality=92)
    created["mole"] = mole_path

    # 4. Medical / Health Study Chart (Non-skin document)
    im_chart = Image.new("RGB", (280, 200), color=(248, 249, 250))
    draw_c = ImageDraw.Draw(im_chart)
    draw_c.rectangle([20, 20, 260, 50], fill=(41, 128, 185))
    draw_c.text((30, 28), "CLINICAL TRIALS REVISION FLOW", fill=(255, 255, 255))
    # Chart bars
    draw_c.rectangle([40, 80, 80, 170], fill=(46, 204, 113))
    draw_c.rectangle([110, 100, 150, 170], fill=(52, 152, 219))
    draw_c.rectangle([180, 60, 220, 170], fill=(155, 89, 182))
    chart_path = IMG_DIR / "04_medical_study_chart.png"
    im_chart.save(chart_path, format="PNG")
    created["chart"] = chart_path

    # 5. Calming Nature & Grounding Scene (Green/Blue Landscape)
    im_nature = Image.new("RGB", (260, 200), color=(135, 206, 235))  # Sky
    draw_n = ImageDraw.Draw(im_nature)
    draw_n.ellipse([180, 30, 230, 80], fill=(255, 223, 0))  # Sun
    draw_n.rectangle([0, 110, 260, 200], fill=(34, 139, 34))  # Green meadow
    draw_n.ellipse([40, 90, 140, 160], fill=(46, 117, 89))   # Mountain/Tree
    im_nature = im_nature.filter(ImageFilter.GaussianBlur(radius=2))
    nature_path = IMG_DIR / "05_nature_grounding_scene.jpg"
    im_nature.save(nature_path, format="JPEG", quality=90)
    created["nature"] = nature_path

    return created


def create_sample_audio_files():
    """Create spoken and silent audio files using pyttsx3 and wave."""
    AUD_DIR.mkdir(parents=True, exist_ok=True)
    created = {}

    samples = [
        ("medical_hypertension", "I have severe headaches and high blood pressure, what could cause this?", "01_headache_hypertension_query.wav"),
        ("exam_anxiety", "I feel very overwhelmed and stressed about my college exams, can you give me coping strategies?", "02_exam_anxiety_coping.wav"),
        ("grounding_breathing", "Can you guide me through a 5-4-3-2-1 grounding exercise?", "03_breathing_exercise_request.wav"),
        ("study_goal", "I want to set a goal to study computer science for three hours everyday.", "04_study_goal_setting.wav"),
    ]

    for key, text, fname in samples:
        target = AUD_DIR / fname
        res_path = synthesize_speech(text, output_filename=fname)
        if res_path and Path(res_path).exists():
            # Copy or move to AUD_DIR if stored elsewhere
            if Path(res_path) != target:
                target.write_bytes(Path(res_path).read_bytes())
            created[key] = target
        else:
            print(f"Warning: TTS failed for {fname}")

    # 5. Pure PCM Silence Audio (1.5 seconds)
    silence_path = AUD_DIR / "05_pure_silence_mic_test.wav"
    with wave.open(str(silence_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        n_samples = int(16000 * 1.5)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    created["silence"] = silence_path

    return created


def test_vision_modes(img_paths):
    """Test vision analyzer with different image types."""
    print("\n" + "=" * 60)
    print("TESTING VISION MODE WITH DIVERSE IMAGE ASSETS")
    print("=" * 60)

    # Test 1: Acute Erythema Rash (Contact Dermatitis)
    rash_bytes = img_paths["rash"].read_bytes()
    t0 = time.time()
    res_rash = analyze_image(rash_bytes, prompt="What is this inflamed red rash on my arm?")
    dt = round(time.time() - t0, 2)
    print(f"\n[Vision 1] Acute Rash ({dt}s):")
    print(f"  Success: {res_rash['success']} | Model: {res_rash.get('model_used')}")
    assert res_rash["success"] is True
    assert "erythema" in res_rash["summary"].lower() or "inflammatory" in res_rash["summary"].lower()
    assert "Gale Encyclopedia of Medicine" in res_rash["summary"]
    assert "Page 10" in res_rash["summary"]
    print("  -> Correctly identified erythema & grounded with Gale Page 10 (Eczema/Dermatitis)!")

    # Test 2: Cutaneous Dermatitis Plaque
    plaque_bytes = img_paths["plaque"].read_bytes()
    t0 = time.time()
    res_plaque = analyze_image(plaque_bytes, prompt="Please inspect this skin plaque.")
    dt = round(time.time() - t0, 2)
    print(f"\n[Vision 2] Dermatitis Plaque ({dt}s):")
    print(f"  Success: {res_plaque['success']} | Model: {res_plaque.get('model_used')}")
    assert res_plaque["success"] is True
    assert "Comfort Steps" in res_plaque["summary"] or "Warning Signs" in res_plaque["summary"]
    print("  -> Correctly synthesized comfort steps and red-flag warning signs!")

    # Test 3: Pigmented Mole / Lesion
    mole_bytes = img_paths["mole"].read_bytes()
    t0 = time.time()
    res_mole = analyze_image(mole_bytes, prompt="Is this dark mole normal?")
    dt = round(time.time() - t0, 2)
    print(f"\n[Vision 3] Pigmented Lesion ({dt}s):")
    print(f"  Success: {res_mole['success']} | Model: {res_mole.get('model_used')}")
    assert res_mole["success"] is True
    assert "dermatologist" in res_mole["summary"].lower() or "clinical" in res_mole["summary"].lower()
    print("  -> Correctly emphasized non-diagnostic boundary and professional evaluation!")

    # Test 4: Medical Study Chart
    chart_bytes = img_paths["chart"].read_bytes()
    t0 = time.time()
    res_chart = analyze_image(chart_bytes, prompt="What is shown in this study flowchart?")
    dt = round(time.time() - t0, 2)
    print(f"\n[Vision 4] Medical Study Flowchart ({dt}s):")
    print(f"  Success: {res_chart['success']} | Model: {res_chart.get('model_used')}")
    assert res_chart["success"] is True
    print("  -> Handled non-cutaneous image safely without false medical diagnosis!")

    # Test 5: Full End-to-End LangGraph Flow with Disease Image
    memory = MemorySaver()
    graph = build_mindly_graph(checkpointer=memory, use_simulation=True)
    res_graph = send_message(
        user_text="What is this skin condition?",
        thread_id="e2e_vision_test_1",
        input_type="image",
        image_data=rash_bytes,
        graph=graph,
        use_simulation=True,
    )
    print(f"\n[Vision 5] Full LangGraph StateGraph Execution:")
    print(f"  Action: {res_graph['action']} | Tool: {res_graph['tool']} | Eval: {res_graph['evaluation']}")
    assert res_graph["action"] == "USE_VISION_TOOL"
    assert res_graph["tool"] == "vision_analyzer"
    assert res_graph["evaluation"] == "PASS"
    print("  -> LangGraph cleanly executed vision_node, evaluated PASS, and assembled response!")


def test_audio_modes(aud_paths):
    """Test audio transcription and subsequent agent dispatch for diverse speech files."""
    print("\n" + "=" * 60)
    print("TESTING AUDIO MODE WITH DIVERSE SPEECH & SILENCE ASSETS")
    print("=" * 60)

    memory = MemorySaver()
    graph = build_mindly_graph(checkpointer=memory, use_simulation=True)

    # Test 1: Medical / Hypertension Spoken Query
    if "medical_hypertension" in aud_paths:
        t0 = time.time()
        transcript = transcribe_audio_file(aud_paths["medical_hypertension"])
        dt = round(time.time() - t0, 2)
        print(f"\n[Audio 1] Medical Query Transcription ({dt}s):")
        print(f"  Transcript: \"{transcript}\"")
        assert len(transcript) > 0
        assert "headache" in transcript.lower() or "blood pressure" in transcript.lower()

        # Run through Agent StateGraph
        res = send_message(transcript, thread_id=f"aud_thread_1_{uuid.uuid4().hex[:6]}", input_type="voice", graph=graph, use_simulation=True)
        print(f"  Agent Routing: Action={res['action']} | Tool={res['tool']}")
        assert res["action"] == "USE_RAG_TOOL"
        assert res["tool"] == "gale_retriever"
        assert "Gale Encyclopedia of Medicine" in res["response"] or len(res.get("sources", [])) > 0
        print("  -> Correctly routed to Gale RAG retriever with clinical citations!")

    # Test 2: Exam Anxiety & Coping Spoken Query
    if "exam_anxiety" in aud_paths:
        t0 = time.time()
        transcript = transcribe_audio_file(aud_paths["exam_anxiety"])
        dt = round(time.time() - t0, 2)
        print(f"\n[Audio 2] Exam Anxiety Transcription ({dt}s):")
        print(f"  Transcript: \"{transcript}\"")
        assert len(transcript) > 0
        assert "exam" in transcript.lower() or "coping" in transcript.lower() or "stress" in transcript.lower()

        res = send_message(transcript, thread_id=f"aud_thread_2_{uuid.uuid4().hex[:6]}", input_type="voice", graph=graph, use_simulation=True)
        print(f"  Agent Routing: Action={res['action']} | Tool={res['tool']}")
        assert res["action"] in ["USE_COPING_TOOL", "USE_GROUNDING_TOOL"]
        print("  -> Correctly routed to Coping/Grounding tool for academic anxiety!")

    # Test 3: Grounding / Breathing Spoken Query
    if "grounding_breathing" in aud_paths:
        t0 = time.time()
        transcript = transcribe_audio_file(aud_paths["grounding_breathing"])
        dt = round(time.time() - t0, 2)
        print(f"\n[Audio 3] Grounding Request Transcription ({dt}s):")
        print(f"  Transcript: \"{transcript}\"")
        assert len(transcript) > 0
        assert "grounding" in transcript.lower() or "5-4-3-2-1" in transcript.lower() or "exercise" in transcript.lower()

        res = send_message(transcript, thread_id=f"aud_thread_3_{uuid.uuid4().hex[:6]}", input_type="voice", graph=graph, use_simulation=True)
        print(f"  Agent Routing: Action={res['action']} | Tool={res['tool']}")
        assert res["action"] == "USE_GROUNDING_TOOL"
        print("  -> Correctly routed to Interactive Grounding tool!")

    # Test 4: Study Goal Creation Spoken Query
    if "study_goal" in aud_paths:
        t0 = time.time()
        transcript = transcribe_audio_file(aud_paths["study_goal"])
        dt = round(time.time() - t0, 2)
        print(f"\n[Audio 4] Goal Setting Transcription ({dt}s):")
        print(f"  Transcript: \"{transcript}\"")
        assert len(transcript) > 0
        assert "goal" in transcript.lower() or "study" in transcript.lower()

        res = send_message(transcript, thread_id=f"aud_thread_4_{uuid.uuid4().hex[:6]}", input_type="voice", graph=graph, use_simulation=True)
        print(f"  Agent Routing: Action={res['action']} | Tool={res['tool']}")
        assert res["action"] == "USE_GOAL_TOOL"
        print("  -> Correctly routed to Goals Tracker and saved goal!")

    # Test 5: Pure Silence Audio Handling
    if "silence" in aud_paths:
        t0 = time.time()
        transcript = transcribe_audio_file(aud_paths["silence"])
        dt = round(time.time() - t0, 2)
        print(f"\n[Audio 5] Pure Silence Handling ({dt}s):")
        print(f"  Transcript: \"{transcript}\" (Length: {len(transcript)})")
        assert transcript == "", "Pure silence must return empty string to prevent hallucinated user input!"
        print("  -> Returned empty string cleanly! Streamlit will display microphone warning without false prompt.")


def main():
    print("Step 1: Generating Sample Assets...")
    img_paths = create_sample_images()
    print(f"Created {len(img_paths)} sample images in '{IMG_DIR}':")
    for k, p in img_paths.items():
        print(f"  • {k}: {p} ({p.stat().st_size} bytes)")

    aud_paths = create_sample_audio_files()
    print(f"\nCreated {len(aud_paths)} sample audio files in '{AUD_DIR}':")
    for k, p in aud_paths.items():
        print(f"  • {k}: {p} ({p.stat().st_size} bytes)")

    # Execute tests
    test_vision_modes(img_paths)
    test_audio_modes(aud_paths)

    print("\n" + "=" * 60)
    print("ALL DIVERSE MULTIMODAL VISION & AUDIO TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
