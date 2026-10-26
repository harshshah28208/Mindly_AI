# 🌿 How to Run Mindly Locally — Step-by-Step Guide

This guide walks you through setting up and running **Mindly** on your local machine (Windows, macOS, or Linux).

---

## 📋 1. Prerequisites

- **Python 3.10+** (Python 3.11 or 3.12 recommended).
- **Git** (to clone/manage the project).
- *(Optional)* **Ollama** if you want to run completely offline local open-source LLMs (`llama3.2:1b`, `llava`).
- *(Optional)* **Groq or OpenAI API Key** if you prefer ultra-fast cloud LLM inference.

> [!NOTE]
> **Zero-Server Dynamic Mode**: Mindly includes a built-in **Smart Dynamic Local Engine**. Even if Ollama is not installed and you don't have an API key, Mindly runs dynamically out-of-the-box without returning fixed or canned answers!

---

## 🚀 2. Quickstart (Under 2 Minutes)

### Step 1: Open Terminal in Project Directory
Open PowerShell or Command Prompt (Windows) or Terminal (macOS/Linux) in the project folder:
```powershell
cd "c:\Users\DELL\OneDrive\Desktop\Ai Project"
```

### Step 2: Activate the Virtual Environment
If `.venv` already exists:
- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

*(If you need to create a fresh virtual environment from scratch)*:
```bash
python -m venv .venv
# Then activate as shown above
pip install -r requirements.txt
```

### Step 3: Initialize the Knowledge Base (One-Time Setup)
Extract and index the *Gale Encyclopedia of Medicine* PDF into the local persistent vector store:
```bash
python ingest.py
```
*Expected output*: `Status : SUCCESS (indexed 40+ chunks into local vector store)`.

### Step 4: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Mindly will open automatically in your default browser at:
👉 **`http://localhost:8501`**

---

## 🤖 3. Choosing Your LLM & Intelligence Provider

Mindly supports **three flexible backends** configured in `.env` or adjusted live inside the app under the **⚙️ Settings** tab:

### Option A: Smart Dynamic Local Mode (Zero Setup Needed)
- **No external server required** and **no API keys needed**.
- Mindly runs an intelligent semantic reasoning engine locally that parses your conversation, extracts emotions, and generates authentic, context-tailored responses.

### Option B: Cloud AI (Ultra-Fast Groq or OpenAI)
If you want to use state-of-the-art open or commercial LLMs:
1. Open the **⚙️ Settings** tab in the web app.
2. Select **Groq Cloud (Free Llama 3.3)** or **OpenAI**.
3. Paste your API key (e.g., from [console.groq.com](https://console.groq.com) or [platform.openai.com](https://platform.openai.com)).
4. Or set it in your `.env` file:
   ```env
   GROQ_API_KEY=gsk_your_key_here
   ```

### Option C: Local Ollama (100% Private Offline Inference)
If you want to run open models on your local GPU/CPU:
1. Download Ollama from [ollama.com](https://ollama.com).
2. Start Ollama and download the primary text model:
   ```bash
   ollama run llama3.2:1b
   ```
3. (Optional for multimodal image analysis) Pull the vision model:
   ```bash
   ollama pull llava
   ```
4. Mindly will automatically detect Ollama running at `http://localhost:11434`.

---

## 🎯 4. Available Features in the UI

1. **🏠 Home (Chat)**:
   - Dynamic conversational wellness companion.
   - 🎤 **Voice Input**: Record voice notes directly with your microphone using `st.audio_input` or upload audio files.
   - 📷 **Photo Analysis**: Attach photos/diagrams for dynamic visual perception analysis.
   - 🛡️ **Medical Scope Protection**: Automatically detects out-of-scope non-health queries (e.g., coding, stock trading, car repair) and politely guides users back to health and wellness.
   - 🚨 **Safety Screening**: Instant deterministic detection of acute self-harm or medical emergencies with crisis hotlines (988, Tele-MANAS, Crisis Text Line) and supervisor alerts.
   - 📥 **Export Chat**: Download conversation history as a formatted Markdown document.

2. **📝 Journal**:
   - Write private daily wellness reflections.
   - 📊 **Mood & Reflection Analytics**: Real-time breakdown of entry frequency, top moods, and emotional wellbeing metrics.

3. **🎯 Goals**:
   - Create study targets and daily habits.
   - 📈 **Visual Progress Bar**: Track completed vs active goals with one-click completion.

4. **📚 Learn**:
   - Search the *Gale Encyclopedia of Medicine* with live semantic excerpt retrieval and citations.

5. **💭 Memory**:
   - Consent-aware memory vault where Mindly securely remembers your preferred name and personal habits.

6. **⚙️ Settings**:
   - Switch LLM providers, configure API keys, toggle Developer/Telemetry mode, and inspect system diagnostics.
