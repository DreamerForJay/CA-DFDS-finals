@echo off
REM CA-DFDS GUI demo launcher. Double-click this file or run run_gui.bat in a terminal.
cd /d "%~dp0"
if not exist ".venv-cadfds\Scripts\python.exe" (
  echo [Error] .venv-cadfds not found. See the environment section in AGENTS.md to set it up first.
  pause
  exit /b 1
)
".venv-cadfds\Scripts\python.exe" gui_demo.py %*
if errorlevel 1 pause
