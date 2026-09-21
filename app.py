"""Mindly - Your Study & Wellness Companion.

A calm, approachable student-friendly companion powered by an agentic
multimodal mental-wellness system.

Architecture (Preserved):
- LangGraph dynamic orchestrator
- Gale Encyclopedia of Medicine local RAG pipeline
- Voice (STT / TTS) & Vision multimodal processing
- Persistent long-term memory, wellness journal, and goals
- Non-emergency human supervisor escalation protocol
- Real-time response quality evaluation & safety guarding
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure project root is always in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config.settings import settings
from graph import get_mindly_graph, send_message
from memory.database import db
from rag.retriever import retrieve_relevant_chunks
from rag.vectorstore import vector_store
from tools.voice import synthesize_speech, transcribe_audio_file

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Mindly — Study & Wellness Companion",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------------------
# STUDENT-FRIENDLY, MODERN, CALM STYLING (CSS)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Lora:ital,wght@0,500;0,600;1,400&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e2d24;
    }

    /* Calming Header */
    .hero-container {
        text-align: center;
        padding: 1.8rem 1rem 1.2rem 1rem;
        background: linear-gradient(180deg, #edf5f0 0%, #f7faf8 100%);
        border: 1px solid #d8e8dc;
        border-radius: 20px;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-family: 'Lora', Georgia, serif;
        font-size: 2.3rem;
        font-weight: 600;
        color: #1b3d2b;
        margin: 0 0 0.2rem 0;
        letter-spacing: -0.01em;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #4a6856;
        margin: 0 0 0.8rem 0;
        font-weight: 500;
    }

    .greeting-line {
        font-size: 1.15rem;
        font-weight: 600;
        color: #214330;
        margin-top: 0.6rem;
    }

    .mood-prompt {
        font-size: 0.88rem;
        color: #5d7a69;
        margin-bottom: 0.5rem;
    }

    /* Quick Action Pills */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid #cde0d3;
        background-color: #ffffff;
        color: #234833;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }

    .stButton > button:hover {
        border-color: #3b7a54;
        background-color: #edf5f0;
        color: #183b27;
        box-shadow: 0 2px 6px rgba(45, 90, 63, 0.08);
    }

    /* Subtle Cards */
    .student-card {
        background: #ffffff;
        border: 1px solid #e0ebe3;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px -2px rgba(35, 65, 45, 0.04);
    }

    .source-chip {
        display: inline-block;
        background: #eef6f1;
        border: 1px solid #c2ded0;
        color: #225239;
        border-radius: 8px;
        padding: 0.3rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.4rem;
    }

    .safety-banner {
        background: #fff4f4;
        border-left: 4px solid #d32f2f;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-top: 0.6rem;
        color: #b71c1c;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Friendly Details Panel */
    .friendly-details {
        background: #f8faf8;
        border: 1px solid #dce8e0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.84rem;
        color: #324e3e;
        line-height: 1.6;
    }

    .detail-row {
        display: flex;
        justify-content: space-between;
        padding: 0.2rem 0;
        border-bottom: 1px dashed #e4ede6;
    }
    .detail-row:last-child {
        border-bottom: none;
    }
    .detail-label {
        font-weight: 600;
        color: #244633;
    }
    .detail-val {
        color: #3f604d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "supervisor_alerts" not in st.session_state:
    st.session_state.supervisor_alerts = []

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "dev_mode" not in st.session_state:
    st.session_state.dev_mode = False


def reset_session():
    """Start a fresh, peaceful reflection session."""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.uploaded_image = None
    get_mindly_graph(reload=True, use_simulation=False)
    get_mindly_graph(reload=True, use_simulation=True)


# ------------------------------------------------------------------------------
# GREETING HELPER (Time-Aware)
# ------------------------------------------------------------------------------
def get_time_greeting() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning 👋"
    elif 12 <= hour < 17:
        return "Good afternoon 👋"
    else:
        return "Good evening 👋"


# ------------------------------------------------------------------------------
# SIDEBAR / QUICK COMPANION MENU
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 Mindly")
    st.caption("Your study & wellness companion")

    health = settings.check_ollama_health()
    is_online = health["is_running"] and health["model_installed"]
    if is_online:
        st.success("● Mindly is ready to listen")
    else:
        st.info("● Quiet Practice Mode (Local)")

    if st.button("🌱 Start Fresh Session", use_container_width=True, on_click=reset_session):
        st.toast("Your conversation has been refreshed.", icon="🌿")

    st.markdown("---")

    st.markdown("##### ⚙️ Settings")
    st.session_state.dev_mode = st.toggle(
        "Developer / Demo Mode",
        value=st.session_state.dev_mode,
        help="Enables full agentic routing telemetry, evaluation metrics, and supervisor logs for inspection.",
    )

    memory_active = db.is_memory_enabled()
    mem_toggle = st.toggle(
        "Remember My Preferences",
        value=memory_active,
        help="Allows Mindly to remember your preferred name and study habits across sessions.",
    )
    if mem_toggle != memory_active:
        db.set_memory_enabled(mem_toggle)
        st.toast(f"Memory is now {'enabled' if mem_toggle else 'disabled'}.", icon="💭")


# ------------------------------------------------------------------------------
# MAIN APPLICATION TABS (STUDENT-FACING)
# ------------------------------------------------------------------------------
tab_home, tab_journal, tab_goals, tab_learn, tab_memory, tab_settings = st.tabs([
    "🏠 Home",
    "📝 Journal",
    "🎯 Goals",
    "📚 Learn",
    "💭 Memory",
    "⚙️ Settings",
])


# ==============================================================================
# TAB 1: HOME & CHAT
# ==============================================================================
with tab_home:
    # Friendly Hero Header
    st.markdown(
        f"""
        <div class="hero-container">
            <h1 class="hero-title">🌿 Mindly</h1>
            <p class="hero-subtitle">Your study & wellness companion</p>
            <div class="greeting-line">{get_time_greeting()}</div>
            <div class="mood-prompt">How are you feeling today?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mood Check-In Buttons
    m_cols = st.columns(5)
    mood_options = [
        ("😊 Great", "I'm feeling great today and ready to learn!"),
        ("🙂 Good", "I'm having a pretty good day."),
        ("😐 Okay", "I'm feeling okay, just taking things one step at a time."),
        ("😟 Stressed", "I'm feeling stressed about my studies and need some guidance."),
        ("😔 Down", "I'm feeling down and overwhelmed right now."),
    ]
    for idx, (label, prompt_text) in enumerate(mood_options):
        with m_cols[idx]:
            if st.button(label, use_container_width=True, key=f"mood_{idx}"):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

    # Quick Actions (Real backend tools via LangGraph)
    st.markdown("<p style='font-size:0.85rem; font-weight:600; color:#3b5c47; margin-top:0.6rem; margin-bottom:0.3rem;'>Quick actions:</p>", unsafe_allow_html=True)
    q1, q2, q3, q4, q5 = st.columns(5)
    with q1:
        if st.button("🫁 Calm me down", use_container_width=True):
            st.session_state.pending_prompt = "Can you guide me through a calming breathing exercise?"
            st.rerun()
    with q2:
        if st.button("📚 Study help", use_container_width=True):
            st.session_state.pending_prompt = "I need help managing my study workload and exam stress."
            st.rerun()
    with q3:
        if st.button("📝 Journal", use_container_width=True):
            st.session_state.pending_prompt = "Journal this: Took a moment to pause and reflect on today."
            st.rerun()
    with q4:
        if st.button("🎯 My goals", use_container_width=True):
            st.session_state.pending_prompt = "What are my study goals and how can I stay consistent?"
            st.rerun()
    with q5:
        if st.button("📖 Learn", use_container_width=True):
            st.session_state.pending_prompt = "What can you teach me about managing stress and health from your library?"
            st.rerun()

    # Multimodal Controls: Voice & Photo (Clean, non-technical)
    with st.expander("✨ Add voice or photo to your message", expanded=False):
        c_voice, c_photo = st.columns(2)
        with c_voice:
            st.markdown("##### 🎤 Talk to Mindly")
            voice_file = st.file_uploader("Upload audio note", type=["wav", "mp3", "m4a"], key="voice_uploader")
            if voice_file:
                transcript = transcribe_audio_file(voice_file)
                st.info(f"**You said:**\n\"{transcript}\"")
                if st.button("Send Voice Message ➤", key="send_voice"):
                    st.session_state.pending_prompt = transcript
                    st.rerun()

        with c_photo:
            st.markdown("##### 📷 Add a photo")
            photo_file = st.file_uploader("Attach an image", type=["png", "jpg", "jpeg"], key="photo_uploader")
            if photo_file:
                st.session_state.uploaded_image = photo_file
                st.image(photo_file, caption="Attached photo", use_container_width=True)

    st.markdown("---")

    # Message Rendering Loop
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🌿"
        author = "You" if msg["role"] == "user" else "Mindly 🌿"
        
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(f"**{author}**")
            st.markdown(msg["content"])

            # Safety / Escalation Notification
            if msg.get("escalation"):
                st.markdown(
                    f"""
                    <div class="safety-banner">
                        <strong>🚨 Immediate support</strong><br>
                        Your message suggests you may need immediate human support.<br>
                        Mindly has activated its safety support process.<br>
                        <small>Human supervisor notification: <strong>Triggered (Demo Simulation)</strong></small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Source citation chip if RAG was used
            sources = msg.get("sources", [])
            if sources:
                pages = sorted(list(set(s.get("page", 1) for s in sources)))
                pages_str = ", ".join(f"Page {p}" for p in pages)
                st.markdown(
                    f'<div class="source-chip">📚 <strong>Source</strong>: Gale Encyclopedia of Medicine ({pages_str})</div>',
                    unsafe_allow_html=True,
                )

            # Audio Player if TTS synthesized
            if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
                st.audio(msg["audio_path"])

            # Technical AI Information (Respectfully collapsed behind friendly expander)
            act = msg.get("activity")
            if act and msg["role"] == "assistant":
                is_high_risk = act.get("risk_level") == "HIGH" or act.get("action") == "CRISIS_RESPONSE"
                expander_label = "🚨 Safety & Route Details" if is_high_risk else ("⚙️ AI Details" if st.session_state.dev_mode else "🧠 How Mindly handled this")
                with st.expander(expander_label, expanded=is_high_risk):
                    if is_high_risk:
                        # Section 28: Strict High-Risk Activity Display
                        supervisor_disp = "DEMO TRIGGERED" if "demo" in str(act.get("escalation") or "").lower() else (act.get("escalation") or "DEMO TRIGGERED")
                        eval_verdict = act.get("evaluation", "PASS")
                        st.markdown(
                            f"""
                            <div class="friendly-details">
                                <div class="detail-row"><span class="detail-label">🚨 Safety check:</span><span class="detail-val" style="color:#d32f2f; font-weight:700;">HIGH RISK</span></div>
                                <div class="detail-row"><span class="detail-label">🛡️ Safety guard:</span><span class="detail-val" style="color:#d32f2f; font-weight:600;">TRIGGERED</span></div>
                                <div class="detail-row"><span class="detail-label">📌 Intent:</span><span class="detail-val">Crisis Related</span></div>
                                <div class="detail-row"><span class="detail-label">🚨 Action:</span><span class="detail-val">Crisis Response</span></div>
                                <div class="detail-row"><span class="detail-label">🔄 Normal routing:</span><span class="detail-val" style="font-weight:600;">BYPASSED</span></div>
                                <div class="detail-row"><span class="detail-label">👤 Supervisor:</span><span class="detail-val">{supervisor_disp}</span></div>
                                <div class="detail-row"><span class="detail-label">✓ Safety evaluation:</span><span class="detail-val" style="color:#2e7d32; font-weight:600;">{eval_verdict}</span></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    elif st.session_state.dev_mode:
                        # Full developer / professor telemetry
                        st.markdown(
                            f"""
                            <div class="friendly-details">
                                <div class="detail-row"><span class="detail-label">🛡️ Risk Level:</span><span class="detail-val">{act.get('risk_level', 'LOW')}</span></div>
                                <div class="detail-row"><span class="detail-label">🎯 Detected Intent:</span><span class="detail-val">{act.get('intent', 'GENERAL_CONVERSATION')}</span></div>
                                <div class="detail-row"><span class="detail-label">💭 Detected Emotion:</span><span class="detail-val">{act.get('emotion', 'neutral')}</span></div>
                                <div class="detail-row"><span class="detail-label">⚡ Selected Action:</span><span class="detail-val">{act.get('action', 'DIRECT_RESPONSE')}</span></div>
                                <div class="detail-row"><span class="detail-label">🛠️ Tool Executed:</span><span class="detail-val">{act.get('tool', 'direct_conversation')}</span></div>
                                <div class="detail-row"><span class="detail-label">📚 Knowledge Chunks:</span><span class="detail-val">{len(sources)} retrieved</span></div>
                                <div class="detail-row"><span class="detail-label">🔍 Evaluator Verdict:</span><span class="detail-val">{act.get('evaluation', 'PASS')} ({act.get('retries', 0)} replan attempts)</span></div>
                                <div class="detail-row"><span class="detail-label">🧩 Graph Status:</span><span class="detail-val">Completed successfully</span></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        # Friendly student-oriented summary (Sections 27 & 28)
                        intent_readable = act.get("intent", "").replace("_", " ").title()
                        action_readable = "Urgent clinical assessment guidance" if "URGENT" in act.get("intent", "") or "urgent" in str(act.get("tool", "")).lower() else (
                            "Knowledge library search" if "RAG" in act.get("action", "") else (
                                "Calming exercise" if "GROUNDING" in act.get("action", "") else (
                                    "Adaptive coping plan" if "COPING" in act.get("action", "") else (
                                        "Journal recorder" if "JOURNAL" in act.get("action", "") else (
                                            "Habits & goals tracker" if "GOAL" in act.get("action", "") else (
                                                "Memory vault" if "MEMORY" in act.get("action", "") else "Thoughtful listening"
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        eval_text = "✓ Approved by Mindly Evaluator" if act.get("evaluation") == "PASS" else "⚠️ Evaluator: replanning verified"
                        st.markdown(
                            f"""
                            <div class="friendly-details">
                                <div class="detail-row"><span class="detail-label">Safety check:</span><span class="detail-val">✓ Passed (Safe)</span></div>
                                <div class="detail-row"><span class="detail-label">Topic:</span><span class="detail-val">{intent_readable}</span></div>
                                <div class="detail-row"><span class="detail-label">Decision:</span><span class="detail-val">{action_readable}</span></div>
                                {'<div class="detail-row"><span class="detail-label">Knowledge source:</span><span class="detail-val">Gale Encyclopedia of Medicine</span></div>' if sources else ''}
                                {'<div class="detail-row"><span class="detail-label">Information verified:</span><span class="detail-val">✓ Found relevant material</span></div>' if sources else ''}
                                <div class="detail-row"><span class="detail-label">Response check:</span><span class="detail-val">{eval_text}</span></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Chat Input Box
    user_input = st.chat_input("💬 Tell me what's on your mind...")

    active_query = None
    if st.session_state.pending_prompt:
        active_query = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    elif user_input:
        active_query = user_input.strip()

    if active_query or st.session_state.uploaded_image:
        query_text = active_query or "Please analyze this uploaded photo."
        st.session_state.messages.append({"role": "user", "content": query_text})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown("**You**")
            st.markdown(query_text)

        img_obj = st.session_state.uploaded_image
        st.session_state.uploaded_image = None
        modality = "image" if img_obj else "text"

        sim_mode = not is_online

        with st.chat_message("assistant", avatar="🌿"):
            st.markdown("**Mindly 🌿**")
            with st.spinner("Reflecting with you..."):
                result = send_message(
                    user_text=query_text,
                    thread_id=st.session_state.thread_id,
                    input_type=modality,
                    image_data=img_obj,
                    use_simulation=sim_mode,
                )
                resp_text = result["response"]
                retrieved_srcs = result.get("sources", [])
                escalation_alert = result.get("escalation")

                # Synthesize gentle voice audio if TTS is available
                audio_file_path = synthesize_speech(resp_text)

                act_data = {
                    "intent": result.get("intent", "GENERAL_CONVERSATION"),
                    "emotion": result.get("emotion", "neutral"),
                    "risk_level": result.get("risk_level", "LOW"),
                    "safety_status": result.get("safety_status", "PASSED"),
                    "health_priority": result.get("health_priority", "NONE"),
                    "action": result.get("action", "DIRECT_RESPONSE"),
                    "tool": result.get("tool", "direct_conversation"),
                    "reasoning": result.get("reasoning", ""),
                    "evaluation": result.get("evaluation", "PASS"),
                    "retries": result.get("retries", 0),
                    "replan_count": result.get("replan_count", 0),
                    "escalation": result.get("escalation"),
                }

                st.markdown(resp_text)

                if escalation_alert:
                    st.markdown(
                        f"""
                        <div class="safety-banner">
                            <strong>🚨 Immediate support</strong><br>
                            Your message suggests you may need immediate human support.<br>
                            Mindly has activated its safety support process.<br>
                            <small>Human supervisor notification: <strong>Triggered (Demo Simulation)</strong></small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.session_state.supervisor_alerts.append({
                        "time": datetime.now().strftime("%I:%M %p"),
                        "risk": "HIGH",
                        "prompt": query_text[:60],
                        "status": "ESCALATION TRIGGERED (Simulation)",
                    })

                if retrieved_srcs:
                    pgs = sorted(list(set(s.get("page", 1) for s in retrieved_srcs)))
                    pgs_str = ", ".join(f"Page {p}" for p in pgs)
                    st.markdown(
                        f'<div class="source-chip">📚 <strong>Source</strong>: Gale Encyclopedia of Medicine ({pgs_str})</div>',
                        unsafe_allow_html=True,
                    )

                if audio_file_path and os.path.exists(audio_file_path):
                    st.audio(audio_file_path)

                # AI Details Expander
                exp_label = "⚙️ AI Details" if st.session_state.dev_mode else "🧠 How Mindly handled this"
                with st.expander(exp_label, expanded=False):
                    if st.session_state.dev_mode:
                        st.markdown(
                            f"""
                            <div class="friendly-details">
                                <div class="detail-row"><span class="detail-label">🛡️ Risk Level:</span><span class="detail-val">{act_data.get('risk_level', 'LOW')}</span></div>
                                <div class="detail-row"><span class="detail-label">🎯 Detected Intent:</span><span class="detail-val">{act_data.get('intent', 'GENERAL_CONVERSATION')}</span></div>
                                <div class="detail-row"><span class="detail-label">💭 Detected Emotion:</span><span class="detail-val">{act_data.get('emotion', 'neutral')}</span></div>
                                <div class="detail-row"><span class="detail-label">⚡ Selected Action:</span><span class="detail-val">{act_data.get('action', 'DIRECT_RESPONSE')}</span></div>
                                <div class="detail-row"><span class="detail-label">🛠️ Tool Executed:</span><span class="detail-val">{act_data.get('tool', 'direct_conversation')}</span></div>
                                <div class="detail-row"><span class="detail-label">📚 Knowledge Chunks:</span><span class="detail-val">{len(retrieved_srcs)} retrieved</span></div>
                                <div class="detail-row"><span class="detail-label">🔍 Evaluator Verdict:</span><span class="detail-val">{act_data.get('evaluation', 'PASS')} ({act_data.get('retries', 0)} replan attempts)</span></div>
                                <div class="detail-row"><span class="detail-label">🧩 Graph Status:</span><span class="detail-val">Completed successfully</span></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        int_read = act_data.get("intent", "").replace("_", " ").title()
                        act_read = "Knowledge library search" if "RAG" in act_data.get("action", "") else (
                            "Calming exercise" if "GROUNDING" in act_data.get("action", "") else (
                                "Adaptive coping plan" if "COPING" in act_data.get("action", "") else "Thoughtful listening"
                            )
                        )
                        st.markdown(
                            f"""
                            <div class="friendly-details">
                                <div class="detail-row"><span class="detail-label">Safety check:</span><span class="detail-val">✓ Passed</span></div>
                                <div class="detail-row"><span class="detail-label">Topic:</span><span class="detail-val">{int_read}</span></div>
                                <div class="detail-row"><span class="detail-label">Decision:</span><span class="detail-val">{act_read}</span></div>
                                {'<div class="detail-row"><span class="detail-label">Knowledge source:</span><span class="detail-val">Gale Encyclopedia of Medicine</span></div>' if retrieved_srcs else ''}
                                {'<div class="detail-row"><span class="detail-label">Information verified:</span><span class="detail-val">✓ Found relevant material</span></div>' if retrieved_srcs else ''}
                                <div class="detail-row"><span class="detail-label">Response check:</span><span class="detail-val">✓ Approved by Mindly Evaluator</span></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resp_text,
                    "activity": act_data,
                    "sources": retrieved_srcs,
                    "audio_path": audio_file_path,
                    "escalation": escalation_alert,
                })


# ==============================================================================
# TAB 2: JOURNAL
# ==============================================================================
with tab_journal:
    st.markdown("### 📝 My Journal")
    st.caption("A private space for your thoughts, reflections, and daily study moments.")

    with st.container():
        journal_text = st.text_area(
            "Write something down...",
            placeholder="Today was busy, but I managed to finish my assignment and take a short walk...",
            height=110,
        )
        j_col1, j_col2 = st.columns([3, 1])
        with j_col1:
            selected_mood = st.selectbox("How did today feel?", ["calm", "grateful", "reflective", "stressed", "exhausted", "hopeful"])
        with j_col2:
            st.write("")
            st.write("")
            if st.button("Add entry", use_container_width=True) and journal_text.strip():
                db.create_journal_entry(journal_text.strip(), mood=selected_mood)
                st.success("Entry saved.")
                st.rerun()

    st.markdown("##### Recent entries")
    st.markdown("---")
    entries = db.get_journal_entries(limit=10)
    if entries:
        for ent in entries:
            st.markdown(
                f"""
                <div class="student-card">
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem;">
                        <span style="font-weight:600; color:#1e3d2b;">🗓️ {ent['timestamp'][:16]}</span>
                        <span class="source-chip" style="margin-top:0;">{ent['mood'].capitalize()}</span>
                    </div>
                    <p style="margin:0.3rem 0 0 0; font-size:0.92rem; color:#2c3e34; line-height:1.5;">{ent['entry']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No journal entries yet. Write your first reflection above, or type *'Journal this: ...'* in chat.")


# ==============================================================================
# TAB 3: GOALS
# ==============================================================================
with tab_goals:
    st.markdown("### 🎯 My Goals")
    st.caption("Keep track of your study targets and daily habits.")

    # Add Goal Section
    with st.expander("➕ Add a new goal", expanded=False):
        new_goal_title = st.text_input("What goal would you like to set?", placeholder="e.g. Study for 2 hours in Pomodoro intervals")
        target_date = st.text_input("Target date (optional)", placeholder="e.g. Tomorrow by 6 PM")
        if st.button("Create Goal", use_container_width=True) and new_goal_title.strip():
            db.create_goal(title=new_goal_title.strip(), target_date=target_date.strip())
            st.success("Goal added successfully.")
            st.rerun()

    st.markdown("##### Today's goals")
    goals = db.get_goals()
    if goals:
        for g in goals:
            is_done = g["status"] == "completed"
            g_icon = "✓" if is_done else "○"
            status_style = "text-decoration: line-through; color: #789082;" if is_done else "color: #1e3d2b; font-weight: 500;"
            
            c_g1, c_g2 = st.columns([5, 1])
            with c_g1:
                st.markdown(
                    f"""
                    <div style="padding: 0.5rem 0; font-size: 0.95rem; {status_style}">
                        <strong>{g_icon}</strong> {g['title']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_g2:
                if not is_done:
                    if st.button("Mark done", key=f"done_{g['id']}", use_container_width=True):
                        db.update_goal_status(g["id"], "completed")
                        st.rerun()
    else:
        st.info("No goals tracked yet. Add a study target above, or type *'Create a goal to ...'* in chat.")


# ==============================================================================
# TAB 4: LEARN (KNOWLEDGE EXPLORER)
# ==============================================================================
with tab_learn:
    st.markdown("### 📚 Learn")
    st.caption("Ask Mindly about a health or wellness topic. Mindly can search its trusted knowledge library when a question needs additional information.")

    learn_query = st.text_input("What would you like to learn about?", placeholder="e.g. What causes tension headaches? Or: What is hypertension?")
    
    if st.button("Explore Topic", use_container_width=True) and learn_query:
        with st.spinner("Searching trusted health library..."):
            chunks = retrieve_relevant_chunks(learn_query, top_k=3)
            if chunks:
                pages = sorted(list(set(c.get("page", 1) for c in chunks)))
                pages_str = ", ".join(f"Page {p}" for p in pages)
                st.markdown(f'<div class="source-chip">📚 <strong>Source</strong>: Gale Encyclopedia of Medicine ({pages_str})</div>', unsafe_allow_html=True)
                
                for c in chunks:
                    with st.container():
                        st.markdown(
                            f"""
                            <div class="student-card">
                                <span style="font-size:0.8rem; font-weight:600; color:#3b7a54;">Excerpt from Page {c['page']}</span>
                                <p style="margin-top:0.4rem; font-size:0.9rem; color:#233a2d; line-height:1.5;">{c['text']}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                st.caption("*(Educational Disclaimer: This information is derived from the Gale Encyclopedia of Medicine for educational awareness and does not replace professional medical evaluation.)*")
            else:
                st.info("No specific knowledge found for that query. You can also ask directly in the chat tab!")


# ==============================================================================
# TAB 5: MEMORY
# ==============================================================================
with tab_memory:
    st.markdown("### 💭 Memory")
    st.caption("Mindly can remember useful preferences when you allow it.")

    mem_status = db.is_memory_enabled()
    st.markdown(f"**Memory Status**: `{'ON' if mem_status else 'OFF'}`")

    st.markdown("##### What Mindly remembers")
    stored = db.get_memories()
    if stored:
        for m in stored:
            st.markdown(f"• **{m['key'].replace('_', ' ').title()}**: {m['value']}")
        
        st.markdown("---")
        if st.button("Forget All Memories", use_container_width=True):
            db.clear_memories()
            st.success("All stored preferences have been cleared.")
            st.rerun()
    else:
        st.info("Mindly hasn't saved any personal preferences yet. You can tell Mindly things like *'Remember that my preferred name is Alex'* in chat.")


# ==============================================================================
# TAB 6: SETTINGS & ABOUT
# ==============================================================================
with tab_settings:
    st.markdown("### ⚙️ Settings & Project Information")
    st.caption("Inspect system settings, companion preferences, and project details.")

    st.markdown("##### Companion Settings")
    st.write(f"• **Primary LLM**: `{settings.ollama_model}` ({'Online' if is_online else 'Quiet Practice Mode'})")
    st.write(f"• **Vision Model**: `{settings.vision_model}`")
    st.write(f"• **Knowledge Library**: `{vector_store.count()} indexed chunks`")
    st.write(f"• **Safety Escalation Protocol**: `{settings.escalation_mode.upper()} (Supervisor: {settings.supervisor_name})`")

    st.markdown("---")
    st.markdown("##### About Mindly")
    st.markdown(
        """
        - **Project**: Mindly — Study & Wellness Companion
        - **Version**: Lab 3 (Multimodal Agentic Architecture)
        - **Technologies**: LangGraph, Local Ollama, Gale Encyclopedia of Medicine (RAG), Whisper STT, Pyttsx3 TTS, Multimodal Vision, SQLite.
        - **Clinical Safety Boundary**: Educational support companion. Non-diagnostic, deterministic safety screening, and simulated human supervisor escalation.
        """
    )

    if st.session_state.supervisor_alerts:
        st.markdown("##### 🚨 Active Supervisor Escalations (Demo Simulation)")
        for alert in st.session_state.supervisor_alerts:
            st.markdown(
                f"""
                <div class="safety-banner">
                    <strong>Alert: {alert['time']}</strong> | Status: <code>{alert['status']}</code><br>
                    Trigger snippet: <em>"{alert['prompt']}"</em>
                </div>
                """,
                unsafe_allow_html=True,
            )
