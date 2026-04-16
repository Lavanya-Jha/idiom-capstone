@echo off
REM ============================================================
REM  Idiom Capstone — One-Click Setup Script (Windows)
REM  Run this from inside the idiom-capstone folder.
REM ============================================================

echo.
echo ============================================================
echo   Idiom Capstone Project — Automated Setup
echo ============================================================
echo.

REM --- Check Python is available ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not on PATH.
    echo Please install Python 3.10 or 3.11 from https://python.org
    pause
    exit /b 1
)

REM --- Create virtual environment ---
echo [1/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo       Done.
) else (
    echo       venv already exists, skipping.
)

REM --- Activate venv ---
call venv\Scripts\activate.bat

REM --- Install PyTorch CPU ---
echo.
echo [2/5] Installing PyTorch (CPU version)...
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo WARNING: PyTorch install had issues. Trying without version pins...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

REM --- Install CLIP from GitHub ---
echo.
echo [3/5] Installing OpenAI CLIP...
pip install git+https://github.com/openai/CLIP.git
if errorlevel 1 (
    echo ERROR: CLIP install failed. Make sure Git is installed.
    echo Download Git from: https://git-scm.com/downloads
    pause
    exit /b 1
)

REM --- Install other dependencies ---
echo.
echo [4/5] Installing remaining dependencies...
pip install Pillow tqdm numpy ftfy regex flask streamlit

REM --- Download ConceptNet ---
echo.
echo [5/5] Setting up ConceptNet NumberBatch embeddings...
if exist "conceptnet\numberbatch_en.pkl" (
    echo       ConceptNet already set up, skipping.
) else (
    python setup_conceptnet.py
)

REM --- Done ---
echo.
echo ============================================================
echo   Setup complete!
echo.
echo   To run the demo:
echo     1. Open a terminal in this folder
echo     2. Run:  venv\Scripts\activate
echo     3. Run:  streamlit run streamlit_app.py
echo     (Browser will open automatically)
echo ============================================================
echo.
pause
