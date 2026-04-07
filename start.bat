@echo off
title NBA Predictor Launcher
echo ================================================
echo   NBA Predictor - Starting services...
echo ================================================
echo.

echo [1/2] Starting FastAPI backend (port 8000)...
start "NBA Predictor - Backend" cmd /k "cd /d "%~dp0" && call .venv\Scripts\activate.bat && uvicorn api.server:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > nul

echo [2/2] Starting Vite dashboard (port 5173)...
start "NBA Predictor - Dashboard" cmd /k "cd /d "%~dp0\dashboard" && npm run dev"

echo.
echo ================================================
echo   Dashboard: http://localhost:5173
echo   API Docs:  http://localhost:8000/docs
echo ================================================
echo.
echo Both services are starting in separate windows.
echo Close those windows to stop the servers.
echo.
pause
