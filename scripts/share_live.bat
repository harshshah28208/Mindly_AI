@echo off
echo ===============================================================
echo     Mindly AI - Start Local Server & Share Live Public Tunnel
echo ===============================================================
echo.
echo 1. Ensuring Ollama is running...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] Ollama is already running.
) else (
    echo Starting Ollama serve in background...
    start "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    timeout /t 3 >NUL
)

echo.
echo 2. Starting Streamlit application on port 8501...
start "" .venv\Scripts\streamlit.exe run app.py --server.port 8501 --server.headless true
timeout /t 3 >NUL

echo.
echo 3. Launching Cloudflare Public HTTPS Tunnel...
echo Share the generated https://*.trycloudflare.com link with your teachers and teammates!
echo.
.\cloudflared.exe tunnel --url http://localhost:8501
pause
