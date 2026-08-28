@echo off
REM ============================================================
REM  GeoCore Analytics Studio - Windows launcher
REM  Double-click this file to start the app.
REM ============================================================
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python not found on PATH.
  echo Install Python 3.10+ from python.org and tick "Add Python to PATH".
  pause
  exit /b 1
)

python -c "import PyQt6" >nul 2>&1
if errorlevel 1 (
  echo First run: installing dependencies. This takes a few minutes...
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Dependency install failed. See messages above.
    pause
    exit /b 1
  )
)

echo Starting GeoCore Analytics Studio...
python -m app.main
if errorlevel 1 (
  echo.
  echo [ERROR] GeoCore exited with an error. Scroll up for the traceback.
  pause
)
