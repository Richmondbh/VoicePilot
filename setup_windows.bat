@echo off
REM One-time setup for VoicePilot on Windows 
cd /d %~dp0
python -m venv .venv
call .venv\Scripts\activate
pip install -r backend\requirements.txt
if not exist backend\.env copy backend\.env.example backend\.env
python scripts\preprocess.py
python scripts\train_classifier.py
cd frontend
call npm install
cd ..
echo.
echo Setup finished.
pause
