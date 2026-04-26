@echo off
echo Starting Adaptive Document Processing — Pro Tier (Gemini 2.5 Pro)
echo Port: 9002
echo.

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo Starting server on http://0.0.0.0:9002
python main.py
