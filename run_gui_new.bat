@echo off
REM Double-click for the web demo. Use --desktop for the original GUI.
cd /d "%~dp0"
if not exist ".venv-cadfds\Scripts\python.exe" (
  echo [Error] .venv-cadfds not found. See the environment section in AGENTS.md to set it up first.
  pause
  exit /b 1
)
if /i "%~1"=="--desktop" (
  ".venv-cadfds\Scripts\python.exe" gui_new_v3.py
) else if "%~1"=="" (
  ".venv-cadfds\Scripts\python.exe" web_demo.py
) else if /i "%~1"=="--web" (
  ".venv-cadfds\Scripts\python.exe" web_demo.py
) else (
  ".venv-cadfds\Scripts\python.exe" gui_new_v3.py %*
)
if errorlevel 1 pause
