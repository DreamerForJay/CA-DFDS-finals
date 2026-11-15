@echo off
REM CA-DFDS new GUI v3 launcher (6-method comparison, YOLO-SAM auto mask, video). Double-click this file.
cd /d "%~dp0"
if not exist ".venv-cadfds\Scripts\python.exe" (
  echo [Error] .venv-cadfds not found. See the environment section in AGENTS.md to set it up first.
  pause
  exit /b 1
)
".venv-cadfds\Scripts\python.exe" gui_new_v3.py %*
if errorlevel 1 pause
