# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    OLLAMA_BASE_URL="http://127.0.0.1:11434" \
    OLLAMA_MODEL="llama3.2:1b" \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS="0.0.0.0"

# Install OS utilities, audio synthesis dependencies, and curl for Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    espeak-ng \
    libespeak-ng-dev \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama engine
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set permissions for startup script
RUN chmod +x scripts/start.sh

# Expose ports for both HuggingFace Spaces (7860) and standard Streamlit (8501)
EXPOSE 7860
EXPOSE 8501

CMD ["/bin/bash", "scripts/start.sh"]
