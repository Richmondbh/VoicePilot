@echo off
REM Starts the backend (port 8000) and the frontend (port 5173) in two windows.
cd /d %~dp0
start "VoicePilot backend" cmd /k "call .venv\Scripts\activate && cd backend && uvicorn app.main:app --reload"
start "VoicePilot frontend" cmd /k "cd frontend && npm run dev"
timeout /t 5 >nul
start http://localhost:5173
