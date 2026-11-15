@echo off
REM CA-DFDS legacy GUI launcher (senior labmate method: paper weights, single method).
cd /d "%~dp0"
if not exist ".venv-cadfds\Scripts\python.exe" (
  echo [Error] .venv-cadfds not found. See the environment section in AGENTS.md to set it up first.
  pause
  exit /b 1
)
".venv-cadfds\Scripts\python.exe" gui_old_senior.py %*
if errorlevel 1 pause
