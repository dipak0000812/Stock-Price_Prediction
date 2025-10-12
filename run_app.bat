@echo off
echo ========================================
echo   Stock Price Predictor - Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if exist "venv\" (
    echo [+] Activating virtual environment...
    call venv\Scripts\activate
) else (
    echo [!] No virtual environment found
    echo [+] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo [+] Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo [+] Starting Streamlit app...
echo.
echo ========================================
echo   App will open in your browser
echo   Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run app.py

pause