@echo off
echo ============================================================
echo  FRONTLINE - 3-Minute Live Demo
echo ============================================================
echo.
echo Step 1: Running full triage pipeline on 40 customer messages...
echo          (This calls the Groq LLM API - takes ~2-3 minutes)
echo.
python main.py
echo.
echo ============================================================
echo Step 2: Re-running evaluation on saved results...
echo ============================================================
echo.
python evaluate.py
echo.
echo ============================================================
echo  Demo complete.
echo  Results saved to: results\triage_results.json
echo ============================================================
pause
