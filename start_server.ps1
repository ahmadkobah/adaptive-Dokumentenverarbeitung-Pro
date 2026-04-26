# Start Adaptive Document Processing — Pro Tier (Gemini 2.5 Pro)
Write-Host "Starting Pro Tier AI Service on port 9002" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt --quiet

Write-Host ""
Write-Host "Starting server on http://0.0.0.0:9002" -ForegroundColor Green
python main.py
