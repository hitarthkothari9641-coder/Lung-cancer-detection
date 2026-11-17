@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  One-click training for the Lung Nodule Screening project.
REM
REM  Step 1: preprocess.py  - resample CT volumes into ./cache
REM  Step 2: train.py       - 5-fold training + metrics + model
REM
REM  Uses .venv if present, otherwise creates it automatically.
REM  Extra arguments are forwarded to train.py, e.g.
REM      train.bat --epochs 2
REM      train.bat --fold 0 --dry-run
REM ============================================================

cd /d "%~dp0"
title Lung Nodule Screening - Training

echo.
echo ============================================================
echo   Lung Nodule Screening - one-click training
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

REM ---------- 2. Create / reuse the virtual environment ----------
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

REM ---------- 3. Install dependencies ----------
echo [2/5] Installing training dependencies ^(first run may take a while^) ...
"%PY%" -m pip install --upgrade pip --quiet
"%PY%" -m pip install -r training\requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo         Check your internet connection and run train.bat again.
    pause
    exit /b 1
)
echo        Dependencies ready.

REM ---------- 4. Step 1: preprocess ----------
echo.
echo [3/5] Step 1 of 2 - preprocessing CT volumes into .\cache ...
"%PY%" training\preprocess.py
if errorlevel 1 (
    echo.
    echo [ERROR] Preprocessing failed. See the message above.
    pause
    exit /b 1
)

REM ---------- 5. Step 2: train ----------
echo.
echo [4/5] Step 2 of 2 - training the model ...
"%PY%" training\train.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] Training failed. See the message above.
    pause
    exit /b 1
)

echo.
echo [5/5] Training finished successfully.
echo       Model weights : artifacts\model.pt
echo       TorchScript   : artifacts\model_ts.pt
echo       Metrics       : artifacts\metrics.json
echo       Plots         : artifacts\plots\
echo.
echo You can now run start.bat to launch the application.
echo.
pause
endlocal
exit /b 0
