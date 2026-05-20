@echo off
title TREES-LLM

echo ========================================
echo   TREES-LLM Startup
echo ========================================
echo.

set ZHIPUAI_API_KEY=32871b74afe147af83edfe74281edaaf.EyDpmMOAPjS85vJI

cd /d "%~dp0"

netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo [WARN] Port 8000 in use, killing...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    timeout /t 2 /nobreak >nul
)

echo [1/2] Starting backend...
start "TREES-LLM Backend" cmd /k "python src/api/main.py"
timeout /t 5 /nobreak >nul

echo [2/2] Opening browser...
start http://localhost:8000/web

echo.
echo ========================================
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:8000/web
echo   Health:   http://localhost:8000/health
echo ========================================
echo.
echo Press any key to exit (backend keeps running)...
pause >nul
