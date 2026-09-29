@echo off
cd /d "%~dp0"
python analyze_dfire.py
if errorlevel 1 (
    echo Analysis failed. See the message above.
    pause
    exit /b 1
)
start "" "%~dp0report\report.html"
