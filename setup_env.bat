@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM setup_env_fixed.bat
REM Always uses THIS project's .venv python (no PATH confusion)
REM ==========================================================

echo.
echo ================================
echo   SkillSight Environment Setup
echo ================================
echo.

REM ---- Move to the script directory (project root) ----
cd /d "%~dp0"
set "PROJECT_ROOT=%cd%"
echo [INFO] Project root: !PROJECT_ROOT!

REM ---- Check Python in PATH (only to create venv) ----
python --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python is not found in PATH.
  echo         Install Python 3.10+ and check "Add python.exe to PATH".
  exit /b 1
)
echo [OK] System Python found.

REM ---- Create venv if missing ----
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment in .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
  )
  echo [OK] Virtual environment created.
) else (
  echo [OK] Virtual environment already exists.
)

REM ---- Pin to THIS venv's python/pip (no activate needed) ----
set "VENV_PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "VENV_PIP=%PROJECT_ROOT%\.venv\Scripts\pip.exe"

if not exist "%VENV_PY%" (
  echo [ERROR] Expected venv python not found: "%VENV_PY%"
  exit /b 1
)

echo [INFO] Using venv python: "%VENV_PY%"
"%VENV_PY%" -c "import sys; print('[INFO] sys.executable =', sys.executable)"

REM ---- Upgrade pip ----
echo [INFO] Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip >nul 2>&1
echo [OK] pip upgraded (if needed).

REM ==========================================================
REM requirements.txt
REM If missing, create it automatically with the needed packages
REM ==========================================================
if not exist "requirements.txt" (
  echo [INFO] requirements.txt not found - creating a minimal one...
  (
    echo numpy
    echo matplotlib
    echo torch
    echo transformers
    echo accelerate
    echo huggingface_hub
  ) > requirements.txt
  echo [OK] Created requirements.txt
) else (
  echo [OK] requirements.txt found.
)

REM ---- Install requirements (log to file) ----
echo [INFO] Installing requirements ...
echo [INFO] Log will be saved to install_log.txt
"%VENV_PY%" -m pip install -r requirements.txt > install_log.txt 2>&1
set "INSTALL_EXIT=!ERRORLEVEL!"

if not "!INSTALL_EXIT!"=="0" (
  echo [ERROR] Installation failed. See install_log.txt for details.
  echo.
  echo --- Last 25 lines of log ---
  powershell -NoProfile -Command "Get-Content -Path 'install_log.txt' -Tail 25"
  exit /b 1
)

echo [OK] All requirements installed successfully.

REM ---- Snapshot installed packages ----
echo.
echo [INFO] Installed packages snapshot:
"%VENV_PY%" -m pip list

REM ==========================================================
REM OPTIONAL: Download models from Hugging Face into exact local folders
REM Run: setup_env_fixed.bat --download-models
REM ==========================================================
set DO_DOWNLOAD=0
for %%A in (%*) do if /I "%%~A"=="--download-models" set DO_DOWNLOAD=1

REM --- HF repo IDs ---
set "HF_REPO_ONEPASS_CW=Roni1999/seed43_ep3_cw"
set "HF_REPO_ONEPASS_BASELINE=Roni1999/seed43_ep3_baseline"
set "HF_REPO_PAIRWISE=Roni1999/pairwise_seed42_epoch3"

REM --- Target folders (KEEPING your current structure) ---
set "TARGET_ONEPASS_CW=!PROJECT_ROOT!\trained onepass\seed43_ep3_cw\final"
set "TARGET_ONEPASS_BASELINE=!PROJECT_ROOT!\trained onepass\seed43_ep3_baseline\final"
set "TARGET_PAIRWISE=!PROJECT_ROOT!\trained pairwise\pairwise_seed42_epoch3\final"

if "!DO_DOWNLOAD!"=="1" (
  echo.
  echo [INFO] Downloading models from Hugging Face ...
  echo [INFO] Targets:
  echo   - OnePass CW       : "!TARGET_ONEPASS_CW!"
  echo   - OnePass Baseline : "!TARGET_ONEPASS_BASELINE!"
  echo   - Pairwise         : "!TARGET_PAIRWISE!"
  echo.

  "%VENV_PY%" -m pip show huggingface_hub >nul 2>&1
  if errorlevel 1 (
    echo [INFO] Installing huggingface_hub ...
    "%VENV_PY%" -m pip install -U huggingface_hub >nul 2>&1
    if errorlevel 1 (
      echo [ERROR] Failed to install huggingface_hub.
      exit /b 1
    )
  )

  call :download_model "!HF_REPO_ONEPASS_CW!" "!TARGET_ONEPASS_CW!"
  if errorlevel 1 exit /b 1

  call :download_model "!HF_REPO_ONEPASS_BASELINE!" "!TARGET_ONEPASS_BASELINE!"
  if errorlevel 1 exit /b 1

  call :download_model "!HF_REPO_PAIRWISE!" "!TARGET_PAIRWISE!"
  if errorlevel 1 exit /b 1

  echo [OK] Model download step finished.
) else (
  echo.
  echo [INFO] Skipping model download. To download run: setup_env_fixed.bat --download-models
  echo [INFO] If you do download, models will be saved under:
  echo   - "!TARGET_ONEPASS_CW!"
  echo   - "!TARGET_ONEPASS_BASELINE!"
  echo   - "!TARGET_PAIRWISE!"
)

echo.
echo ================================
echo   DONE - Environment is ready
echo ================================
echo.
exit /b 0


REM =================== SUBROUTINES ===================

:download_model
set "REPO_ID=%~1"
set "TARGET_DIR=%~2"

REM ---- Skip if we already have a complete local model folder ----
set HAS_CONFIG=0
set HAS_WEIGHTS=0

if exist "!TARGET_DIR!\config.json" set HAS_CONFIG=1
if exist "!TARGET_DIR!\model.safetensors" set HAS_WEIGHTS=1
if exist "!TARGET_DIR!\pytorch_model.bin" set HAS_WEIGHTS=1

if "!HAS_CONFIG!"=="1" if "!HAS_WEIGHTS!"=="1" (
  echo [OK] Found existing model in "!TARGET_DIR!" - skipping download. Repo: !REPO_ID!
  exit /b 0
)

echo [INFO] Downloading !REPO_ID! -> "!TARGET_DIR!"
mkdir "!TARGET_DIR!" 2>nul

"%VENV_PY%" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=r'!REPO_ID!', repo_type='model', local_dir=r'!TARGET_DIR!', local_dir_use_symlinks=False); print('done')"
if errorlevel 1 (
  echo [ERROR] Download failed for !REPO_ID!
  exit /b 1
)

echo [OK] Downloaded !REPO_ID! into "!TARGET_DIR!"
exit /b 0
