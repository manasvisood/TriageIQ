@echo off
echo ============================================================
echo  FRONTLINE - AI Customer Message Triage System
echo  Setup Script
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo      Done.

REM Check for .env
if not exist ".env" (
    echo.
    echo [2/3] Creating .env file...
    copy .env.example .env
    echo.
    echo *** ACTION REQUIRED ***
    echo Open the .env file and replace "your_groq_api_key_here" with your real key.
    echo Get a free key at: https://console.groq.com
    echo.
    notepad .env
) else (
    echo [2/3] .env already exists. Skipping.
)

echo [3/3] Running tests to verify setup...
python -m pytest tests/ -q --no-header
if errorlevel 1 (
    echo [WARNING] Some tests failed. Check output above.
) else (
    echo      All tests passed.
)

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  To run the triage pipeline:
echo      python main.py
echo.
echo  To evaluate results:
echo      python evaluate.py
echo.
echo  To run tests:
echo      python -m pytest tests/ -v
echo ============================================================
pause
