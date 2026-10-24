"""Mindly - Agentic AI Medical & Wellness System.

An autonomous, multimodal multi-agent architecture built on LangGraph:
- Autonomous Multi-Agent Routing & Execution Graph
- Clinical RAG Pipeline over the Gale Encyclopedia of Medicine
- Deterministic Safety Interceptor & Human-in-the-Loop Supervisor Escalations
- Self-Reflective Evaluator with Bounded Replanning Loops (MAX_RETRIES = 2)
- Multimodal Processing (Text, Voice STT/TTS, Medical Vision Perception)
- Zero-Dependency Smart Dynamic Engine with Ollama & Cloud LLM Orchestration
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from agent.safety import assess_safety, screen_urgent_medical
from agent.intent import is_out_of_scope, evaluate_math_expression
from tools.voice import synthesize_speech, transcribe_audio_file

# ------------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Mindly — Agentic AI Medical & Wellness Copilot",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# MODERN AGENTIC AI DESIGN SYSTEM (CSS)
# ------------------------------------------------------------------------------
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }

    .stButton > button:hover {
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    </style>""",
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

if "enable_tts" not in st.session_state:
    st.session_state.enable_tts = False


def reset_session():
    """Start a fresh session with new thread ID and refreshed graph."""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.uploaded_image = None
    get_mindly_graph(reload=True, use_simulation=False)
    get_mindly_graph(reload=True, use_simulation=True)


# ------------------------------------------------------------------------------
# SIDEBAR: SYSTEM & AGENT CONTROLS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌿 Mindly Agentic AI")
    st.caption("Clinical & Wellness Multi-Agent System")

    health = settings.check_ollama_health()
    has_api = bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))

    if has_api:
        st.success("🟢 Cloud LLM Connected (Groq/OpenAI)")
    elif health["is_running"] and health["model_installed"]:
        st.success(f"🟢 Local Ollama Active ({settings.ollama_model})")
    elif health["is_running"]:
        st.warning("🟡 Ollama Online (Model Pulling)")
    else:
        st.info("🔵 Dynamic Local Engine (Zero-Server Mode)")

    st.markdown("---")
    st.markdown("##### 🧩 Architecture Status")
    st.markdown(f"• **Graph Orchestrator:** `LangGraph StateGraph`")
    st.markdown(f"• **Knowledge Engine:** `Gale Encyclopedia ({vector_store.count()} chunks)`")
    st.markdown(f"• **Safety Screen:** `Deterministic Guard + Screen`")
    st.markdown(f"• **Evaluator Loop:** `Self-Reflection (Max Retries: 2)`")
    st.markdown(f"• **Active Thread:** `{st.session_state.thread_id[:8]}...`")

    st.markdown("---")
    st.session_state.enable_tts = st.toggle(
        "🔊 Voice Audio (TTS)",
        value=st.session_state.enable_tts,
        help="Synthesizes voice audio response for messages.",
    )

    if st.button("🔄 Reset Agent State", use_container_width=True, on_click=reset_session):
        st.toast("Agent conversation state refreshed.", icon="🌿")


# ------------------------------------------------------------------------------
# HERO HEADER (CLEAN UNINDENTED HTML)
# ------------------------------------------------------------------------------
st.markdown(
"""<div style="padding: 1.5rem 1.2rem; background: linear-gradient(135deg, #1b3d2b 0%, #28543b 100%); border-radius: 16px; margin-bottom: 1.2rem; color: #ffffff;">
    <h1 style="color: #ffffff; margin: 0 0 0.3rem 0; font-size: 2.1rem; font-weight: 700;">🌿 Mindly — Agentic AI Medical & Wellness Copilot</h1>
    <p style="color: #d1f2e1; margin: 0; font-size: 1.0rem; line-height: 1.5;">Autonomous Hierarchical Multi-Agent System with Gale Encyclopedia of Medicine RAG, Deterministic Safety Interceptors, and Evaluator Reflection Loops</p>
</div>""",
unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# AGENTIC TELEMETRY RENDERER (NO RAW HTML TAG LEAKS)
# ------------------------------------------------------------------------------
def render_agent_trace(act: Dict[str, Any], sources: List[Dict[str, Any]]):
    """Renders agent reasoning trace using native Streamlit layout without raw HTML leaks."""
    is_high_risk = act.get("risk_level") == "HIGH" or act.get("action") == "CRISIS_RESPONSE"
    expander_title = "🚨 Safety Guard & Route Alert" if is_high_risk else "🧠 LangGraph Agent Execution Trace"

    with st.expander(expander_title, expanded=is_high_risk):
        c1, c2 = st.columns(2)
        with c1:
            risk = act.get("risk_level", "LOW")
            risk_color = "red" if risk == "HIGH" else ("orange" if risk == "MEDIUM" else "green")
            st.markdown(f"**🛡️ Risk Level:** :{risk_color}[**{risk}**]")
            st.markdown(f"**🎯 Intent Detected:** `{act.get('intent', 'GENERAL_CONVERSATION')}`")
            st.markdown(f"**💭 Emotional State:** `{act.get('emotion', 'neutral')}`")
            st.markdown(f"**⚡ Planned Action:** `{act.get('action', 'DIRECT_RESPONSE')}`")
        with c2:
            st.markdown(f"**🛠️ Tool Executed:** `{act.get('tool', 'direct_conversation')}`")
            st.markdown(f"**📚 Gale Chunks:** `{len(sources)} retrieved`")
            eval_val = act.get("evaluation", "PASS")
            eval_color = "green" if eval_val == "PASS" else "orange"
            retries = act.get("retries", 0)
            st.markdown(f"**🔍 Evaluator:** :{eval_color}[**{eval_val}**] `({retries} replans)`")
            esc = act.get("escalation")
            if esc:
                st.markdown(f"**👤 Supervisor:** :red[**{esc}**]")
            else:
                st.markdown("**🧩 Graph State:** :green[**Completed**]")


# ------------------------------------------------------------------------------
# MAIN APPLICATION TABS (PURE AGENTIC AI)
# ------------------------------------------------------------------------------
tab_chat, tab_graph, tab_rag, tab_safety, tab_diagnostics = st.tabs([
    "💬 Agentic Copilot",
    "🧠 Multi-Agent Architecture",
    "📚 Clinical Knowledge RAG",
    "🛡️ Safety & Guardrails",
    "⚡ Model Diagnostics",
])


# ==============================================================================
# TAB 1: AGENTIC COPILOT
# ==============================================================================
with tab_chat:
    st.markdown("##### 💡 Quick Test Prompts (Agentic Capabilities)")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    with col_p1:
        if st.button("🩺 Clinical RAG Test", use_container_width=True):
            st.session_state.pending_prompt = "What are the common causes and symptoms of hypertension?"
            st.rerun()
    with col_p2:
        if st.button("🔬 Brain-Gut Symptom", use_container_width=True):
            st.session_state.pending_prompt = "Why does my stomach hurt badly when I get anxious before exams?"
            st.rerun()
    with col_p3:
        if st.button("🧮 Math / Scope Test", use_container_width=True):
            st.session_state.pending_prompt = "What is 2+2?"
            st.rerun()
    with col_p4:
        if st.button("🚨 Urgent Medical Test", use_container_width=True):
            st.session_state.pending_prompt = "I have sudden severe crushing chest pain, difficulty breathing, and fainting."
            st.rerun()

    # Multimodal Inputs Bar
    with st.expander("🎙️ / 📷 Multimodal Input Controls (Voice & Vision)", expanded=False):
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.markdown("**🎙️ Audio Voice Message**")
            audio_upload = st.file_uploader("Upload audio recording (.wav, .mp3, .ogg):", type=["wav", "mp3", "ogg"], key="audio_uploader")
            if audio_upload is not None:
                audio_bytes = audio_upload.read()
                st.audio(audio_bytes)
                with st.spinner("Transcribing audio message..."):
                    transcript = transcribe_audio_file(audio_bytes)
                    if transcript:
                        st.info(f"Transcribed: *\"{transcript}\"*")
                        if st.button("Send Audio Transcript to Agent", use_container_width=True):
                            st.session_state.pending_prompt = transcript
                            st.rerun()

        with c_v2:
            st.markdown("**📷 Medical Image Inspection**")
            img_upload = st.file_uploader("Upload symptom or clinical image (.jpg, .png):", type=["jpg", "jpeg", "png"], key="img_uploader")
            if img_upload is not None:
                st.session_state.uploaded_image = img_upload.read()
                st.image(st.session_state.uploaded_image, caption="Uploaded Image for Vision Analyzer", width=220)

    st.markdown("---")

    # Render Chat History
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg.get("image"):
                st.image(msg["image"], caption="Clinical Inspection Image", width=260)
            
            st.markdown(msg["content"])

            # Sources chip
            sources = msg.get("sources", [])
            if sources:
                pages = sorted(list(set(c.get("page", 1) for c in sources)))
                pages_str = ", ".join(f"Page {p}" for p in pages)
                st.info(f"📚 **Clinical Knowledge Source**: *Gale Encyclopedia of Medicine* ({pages_str})")

            # Audio output if available
            if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
                st.audio(msg["audio_path"])

            # Render Agent Telemetry Trace
            act = msg.get("activity")
            if act and msg["role"] == "assistant":
                render_agent_trace(act, sources)

    # Chat Action Controls
    if st.session_state.messages:
        c_clr, c_exp = st.columns([1, 1])
        with c_clr:
            if st.button("🗑️ Clear Chat History", use_container_width=True, key="clear_chat_hist"):
                st.session_state.messages = []
                st.rerun()
        with c_exp:
            transcript_md = "# Mindly Agentic AI Conversation Log\n\n" + "\n\n".join(
                [f"### {'👤 User' if m['role'] == 'user' else '🌿 Mindly Agent'}\n{m['content']}" for m in st.session_state.messages]
            )
            st.download_button(
                "📥 Export Session Markdown",
                data=transcript_md,
                file_name=f"mindly_session_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
                key="export_chat_hist",
            )

    # Chat Input Box
    user_input = st.chat_input("💬 Ask a health question, share symptoms, or test agent routing...")

    active_query = None
    if st.session_state.pending_prompt:
        active_query = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    elif user_input:
        active_query = user_input.strip()

    if active_query or st.session_state.uploaded_image:
        user_text = active_query or ""
        img_data = st.session_state.uploaded_image
        st.session_state.uploaded_image = None

        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_text,
            "image": img_data,
        })
        with st.chat_message("user"):
            if img_data:
                st.image(img_data, width=260)
            st.markdown(user_text)

        # Run through Agentic LangGraph StateGraph
        with st.chat_message("assistant"):
            with st.spinner("Agent analyzing, planning tools, and evaluating..."):
                input_mode = "image" if img_data else "text"
                result = send_message(
                    user_text=user_text,
                    thread_id=st.session_state.thread_id,
                    input_type=input_mode,
                    image_data=img_data,
                    use_simulation=False,
                )

                resp_text = result.get("response", "")
                st.markdown(resp_text)

                retrieved_srcs = result.get("sources", [])
                if retrieved_srcs:
                    pgs = sorted(list(set(c.get("page", 1) for c in retrieved_srcs)))
                    pgs_str = ", ".join(f"Page {p}" for p in pgs)
                    st.info(f"📚 **Clinical Knowledge Source**: *Gale Encyclopedia of Medicine* ({pgs_str})")

                # Speech synthesis
                audio_file_path = None
                if st.session_state.enable_tts:
                    try:
                        audio_file_path = synthesize_speech(resp_text[:350])
                        if audio_file_path and os.path.exists(audio_file_path):
                            st.audio(audio_file_path)
                    except Exception:
                        pass

                # Agent Trace Telemetry
                render_agent_trace(result, retrieved_srcs)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resp_text,
                    "activity": result,
                    "sources": retrieved_srcs,
                    "audio_path": audio_file_path,
                })


# ==============================================================================
# TAB 2: MULTI-AGENT ARCHITECTURE (LANGGRAPH VISUALIZER)
# ==============================================================================
with tab_graph:
    st.markdown("### 🧠 LangGraph Multi-Agent Architecture")
    st.caption("Visual StateGraph execution lifecycle and node responsibility breakdown.")

    st.markdown(
        """```mermaid
graph TD
    START((START)) --> SAFETY[assess_safety<br>Deterministic Risk Pre-Screen]
    SAFETY -->|HIGH RISK| CRISIS[crisis_escalation_node<br>Lifelines & Crisis Protocol]
    CRISIS --> SUPERVISOR[supervisor_escalation<br>Human-in-the-Loop Alert]
    SUPERVISOR --> END_NODE((END))

    SAFETY -->|NORMAL / SAFE| UNDERSTAND[understand_node<br>Input Normalization]
    UNDERSTAND --> INTENT[intent_node<br>12 Clinical/Wellness Categories]
    INTENT --> EMOTION[emotion_node<br>Affect & Sentiment Analysis]
    EMOTION --> PLANNER[planner_node<br>Hierarchical Priority Router 0-9]

    PLANNER -->|Medical Inquiries| RAG[tools/rag<br>Gale Encyclopedia RAG]
    PLANNER -->|Acute Red-Flags| URGENT[tools/rag<br>Urgent Clinical Protocol]
    PLANNER -->|Symptom Image| VISION[tools/vision<br>Multimodal Vision Analyzer]
    PLANNER -->|Calming Requests| GROUND[tools/grounding<br>Box Breathing / Reflection]
    PLANNER -->|Stress / Workload| COPING[tools/coping<br>Adaptive Coping Generator]
    PLANNER -->|Factual / Math / Scope| DIRECT[tools/direct_response<br>Direct & Scope Guidance]

    RAG --> EVALUATOR[evaluator_node<br>Self-Reflection & Hallucination Check]
    VISION --> EVALUATOR
    GROUND --> EVALUATOR
    COPING --> EVALUATOR
    DIRECT --> EVALUATOR

    EVALUATOR -->|RETRY: replan count < 2| PLANNER
    EVALUATOR -->|PASS or Max Retries| FINAL[generate_final_response]
    FINAL --> END_NODE
```"""
    )

    st.markdown("---")
    st.markdown("##### 🧩 Architectural Node Responsibilities")

    c_n1, c_n2 = st.columns(2)
    with c_n1:
        with st.container(border=True):
            st.markdown("**🛡️ Priority 0: Deterministic Safety Interceptor**")
            st.markdown(
                "Screens user input for acute high-risk indicators (self-harm, crisis). "
                "High risk immediately **bypasses normal routing**, delivers validated crisis lifelines (988, 741741), "
                "and records an auditable escalation record."
            )

        with st.container(border=True):
            st.markdown("**🎯 Cognitive Understanding (Intent & Emotion)**")
            st.markdown(
                "Categorizes input across 12 standardized clinical and conversational intents "
                "(HEALTH_INFORMATION, URGENT_HEALTH, EMOTIONAL_SUPPORT, GROUNDING_REQUEST, OUT_OF_SCOPE, etc.) "
                "and detects dominant emotional tone."
            )

    with c_n2:
        with st.container(border=True):
            st.markdown("**⚡ Autonomous Planner Node (Hierarchical 0-9)**")
            st.markdown(
                "Dynamically selects the optimal tool based on clinical urgency, safety screening, "
                "and input type. Enforces domain scope boundaries and prevents unnecessary tool execution."
            )

        with st.container(border=True):
            st.markdown("**🔍 Self-Reflective Evaluator Node**")
            st.markdown(
                "Validates tool outputs for clinical boundaries (non-diagnostic guarantee), output completeness (>20 chars), "
                "and grounding. Implements a bounded reflection loop (`MAX_RETRIES = 2`) preventing infinite execution cycles."
            )


# ==============================================================================
# TAB 3: CLINICAL KNOWLEDGE RAG STUDIO
# ==============================================================================
with tab_rag:
    st.markdown("### 📚 Clinical Knowledge RAG Studio")
    st.caption("Inspect and query the local Gale Encyclopedia of Medicine Vector Store.")

    rag_query = st.text_input("Enter a medical symptom, disease, or clinical query:", placeholder="e.g. hypertension, migraine, iron deficiency, gastritis, insomnia")

    if st.button("🔍 Search Gale Encyclopedia Vector Store", use_container_width=True) and rag_query.strip():
        with st.spinner("Performing semantic vector similarity search..."):
            chunks = retrieve_relevant_chunks(rag_query.strip(), top_k=3)
            if chunks:
                pages = sorted(list(set(c.get("page", 1) for c in chunks)))
                pages_str = ", ".join(f"Page {p}" for p in pages)
                st.success(f"✓ Retrieved {len(chunks)} relevant clinical chunks across {pages_str}")

                for idx, c in enumerate(chunks, 1):
                    with st.container(border=True):
                        st.markdown(f"**📄 Excerpt {idx} — Page {c.get('page', 1)}**")
                        st.markdown(f"> {c.get('text', '')}")
                        st.caption(f"Length: {len(c.get('text', ''))} characters | Source: Gale Encyclopedia of Medicine")

                st.caption("*(Educational Disclaimer: Derived from the Gale Encyclopedia of Medicine for educational awareness. Does not replace professional medical evaluation.)*")
            else:
                st.warning("No matching excerpts found for that query in the vector store.")


# ==============================================================================
# TAB 4: SAFETY & GUARDRAILS MONITOR
# ==============================================================================
with tab_safety:
    st.markdown("### 🛡️ Safety & Supervisor Guardrails")
    st.caption("Real-time safety guard testing sandbox and supervisor escalation audit trail.")

    st.markdown("##### 🔬 Live Safety Screening Sandbox")
    test_prompt = st.text_input("Test any query against the safety classifiers:", placeholder="e.g. I have severe chest pain and cannot breathe")

    if test_prompt.strip():
        safety_eval = assess_safety(test_prompt.strip())
        urgent_eval = screen_urgent_medical(test_prompt.strip())
        scope_eval = is_out_of_scope(test_prompt.strip())
        math_eval = evaluate_math_expression(test_prompt.strip())

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            risk = safety_eval["risk_level"]
            col_color = "red" if risk == "HIGH" else "green"
            st.metric("Risk Level", f"{risk}", delta="SAFE" if risk == "LOW" else "GUARD TRIGGERED", delta_color="normal" if risk == "LOW" else "inverse")
        with col_s2:
            st.metric("Urgent Medical", "YES" if urgent_eval["is_urgent"] else "NO")
        with col_s3:
            st.metric("Out of Scope", "YES" if scope_eval else "NO")
        with col_s4:
            st.metric("Math Expression", f"{math_eval['ans']}" if math_eval else "NO")

        if safety_eval["risk_level"] == "HIGH":
            st.error("🚨 HIGH RISK DETECTED: Normal agentic routing bypassed. Crisis response & supervisor alert triggered.")
        elif urgent_eval["is_urgent"]:
            st.warning("⚠️ URGENT CLINICAL SYMPTOMS: Immediate assessment advisory recommended.")

    st.markdown("---")
    st.markdown("##### 🚨 Supervisor Escalation Audit Log")
    escalations = db.get_escalations(limit=20)
    if escalations:
        for esc in escalations:
            with st.container(border=True):
                c_e1, c_e2 = st.columns([3, 1])
                with c_e1:
                    st.markdown(f"**Timestamp:** `{esc.get('timestamp')}`")
                    st.markdown(f"**Masked Prompt:** *\"{esc.get('user_prompt')}\"*")
                    st.markdown(f"**Trigger Reason:** {esc.get('reason')}")
                with c_e2:
                    st.markdown(f"**Status:** `{esc.get('status')}`")
                    st.markdown(f"**Supervisor:** `{esc.get('supervisor_name')}`")
    else:
        st.info("No supervisor escalations recorded in this session.")


# ==============================================================================
# TAB 5: MODEL DIAGNOSTICS & SYSTEM SPECS
# ==============================================================================
with tab_diagnostics:
    st.markdown("### ⚡ Model Diagnostics & System Specs")
    st.caption("Inspect live AI backend orchestration and local execution health.")

    c_d1, c_d2 = st.columns(2)
    with c_d1:
        with st.container(border=True):
            st.markdown("**🤖 Model Orchestrator**")
            st.markdown(f"• **Ollama Server:** `http://localhost:11434` ({'ONLINE' if health['is_running'] else 'OFFLINE'})")
            st.markdown(f"• **Configured Ollama Model:** `{settings.ollama_model}`")
            st.markdown(f"• **Installed Models:** `{', '.join(health.get('available_models', [])) or 'None'}`")
            st.markdown(f"• **Vision Model:** `{settings.vision_model}`")
            st.markdown(f"• **Ollama Temperature:** `{settings.ollama_temperature}`")
            st.markdown(f"• **Max Evaluator Retries:** `{settings.max_eval_retries}`")

    with c_d2:
        with st.container(border=True):
            st.markdown("**📦 Vector Store & Database**")
            st.markdown(f"• **Vector Store Count:** `{vector_store.count()} indexed chunks`")
            st.markdown(f"• **Knowledge Directory:** `{settings.knowledge_dir}`")
            st.markdown(f"• **Gale PDF Filename:** `{settings.gale_pdf_filename}`")
            st.markdown(f"• **SQLite Database:** `{settings.db_path}`")
            st.markdown(f"• **Thread Checkpointer:** `langgraph.checkpoint.memory.MemorySaver`")

    st.markdown("---")
    st.markdown("##### 📋 System Technical Specifications")
    st.markdown(
        """
        - **Project**: Mindly — Agentic AI Medical & Wellness Copilot
        - **Architecture**: LangGraph StateGraph, Deterministic Guardrails, Self-Evaluating Reflection Loops
        - **Clinical Core**: Gale Encyclopedia of Medicine local Vector Database
        - **Multi-Backend LLMs**: Local Ollama (`llama3.2:1b`, `llava`), Groq Cloud API, OpenAI API, and Smart Dynamic Local Fallback
        - **Clinical Safety Guarantee**: Non-diagnostic, deterministic safety screening, medical domain scope boundary enforcement, and human supervisor escalation
        """
    )
