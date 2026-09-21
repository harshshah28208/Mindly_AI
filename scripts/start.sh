#!/usr/bin/env bash
set -e

echo "=========================================="
echo "   🌿 Starting Mindly AI Service   "
echo "=========================================="

# 1. Start Ollama daemon in the background
echo "[1/3] Starting background Ollama daemon..."
ollama serve &
OLLAMA_PID=$!

# 2. Poll until Ollama API responds
echo "[2/3] Waiting for Ollama engine to initialize..."
MAX_RETRIES=30
RETRY_COUNT=0
until curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Warning: Ollama took longer than 30s to initialize, continuing startup..."
        break
    fi
    sleep 1
done

if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama daemon is active and responding on port 11434."
fi

# 3. Pull required primary model
TARGET_MODEL=${OLLAMA_MODEL:-"llama3.2:1b"}
echo "[3/3] Verifying / pulling primary model: '$TARGET_MODEL'..."
ollama pull "$TARGET_MODEL" || echo "Note: Model pull exited. If model is pre-cached, startup continues."

# Optional Vision model pull
if [ -n "$VISION_MODEL" ] && [ "$VISION_MODEL" != "none" ]; then
    echo "Checking vision model: '$VISION_MODEL'..."
    ollama pull "$VISION_MODEL" || echo "Vision model pull skipped or deferred. Dynamic fallback enabled."
fi

# 4. Launch Streamlit Web Application
PORT=${PORT:-${STREAMLIT_SERVER_PORT:-7860}}
echo "=========================================="
echo "   🌿 Launching Mindly UI on port $PORT"
echo "=========================================="
exec streamlit run app.py \
    --server.port="$PORT" \
    --server.address="0.0.0.0" \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
