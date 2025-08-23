@echo off
echo ====================================================
echo           AI Trading System Startup
echo ====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or later from https://python.org
    pause
    exit /b 1
)

echo Starting AI Trading System...
echo.

REM Run the startup script
python start.py

echo.
echo System has stopped.
pause