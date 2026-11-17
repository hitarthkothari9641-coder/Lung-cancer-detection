@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  One-click launcher for the Lung Nodule Screening app.
REM
REM  Starts the FastAPI backend on http://127.0.0.1:8000 and the
REM  Next.js frontend on http://localhost:3000, then opens the
REM  browser. Closing this window stops both servers.
REM ============================================================

cd /d "%~dp0"
title Lung Nodule Screening - Launcher

echo.
echo ============================================================
echo   Lung Nodule Screening - starting the application
echo ============================================================
echo.

REM ---------- 1. Locate Python ----------
set "PY_CMD="
where py >nul 2>nul && set "PY_CMD=py -3"
if not defined PY_CMD (
    where python >nul 2>nul && set "PY_CMD=python"
)
if not defined PY_CMD (
    echo [ERROR] Python 3.9+ was not found on PATH.
    echo         Install it from https://www.python.org/downloads/ and tick
    echo         "Add Python to PATH" during setup, then run this file again.
    echo.
    pause
    exit /b 1
)

REM ---------- 2. Locate Node.js ----------
where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js / npm was not found on PATH.
    echo         Install the LTS build from https://nodejs.org/ and run this file again.
    echo.
    pause
    exit /b 1
)

REM ---------- 3. Python environment ----------
if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating virtual environment in .venv ...
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/5] Using existing virtual environment in .venv
)
set "PY=.venv\Scripts\python.exe"

echo [2/5] Checking backend dependencies ...
"%PY%" -c "import fastapi, uvicorn, torch" >nul 2>nul
if errorlevel 1 (
    echo        Installing backend dependencies ^(first run may take a while^) ...
    "%PY%" -m pip install --upgrade pip --quiet
    "%PY%" -m pip install -r backend\requirements.txt --quiet
    if errorlevel 1 (
        echo [ERROR] Backend dependency installation failed.
        pause
        exit /b 1
    )
)

REM ---------- 4. Frontend dependencies ----------
echo [3/5] Checking frontend dependencies ...
if not exist "frontend\node_modules\next" (
    echo        Installing frontend packages ^(first run may take a while^) ...
    pushd frontend
    call npm install --no-audit --no-fund
    if errorlevel 1 (
        popd
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
    popd
)

REM ---------- 5. Sample cohort data, if missing ----------
if not exist "data\sample_patients.json" (
    echo [4/5] Generating the sample patient cohort ...
    "%PY%" scripts\make_sample_data.py
) else (
    echo [4/5] Sample cohort data present
)

REM ---------- 6. Start the services ----------
echo [5/5] Starting backend and frontend ...
echo.
start "Lung Nodule Screening - Backend" cmd /k ""%PY%" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
timeout /t 4 /nobreak >nul
start "Lung Nodule Screening - Frontend" cmd /k "cd /d "%CD%\frontend" && npm run dev"
timeout /t 8 /nobreak >nul
start "" "http://localhost:3000"

echo.
echo ============================================================
echo   Application is starting
echo ============================================================
echo   Frontend : http://localhost:3000
echo   API docs : http://127.0.0.1:8000/docs
echo.
echo   Two new windows were opened. Keep them running while you
echo   use the app; closing those windows shuts the servers down.
echo.
echo   Tip: run train.bat once to train the model before analyzing
echo        real .mhd volumes. Without weights the app uses the
echo        built-in demo cohort.
echo.
pause
endlocal
exit /b 0
