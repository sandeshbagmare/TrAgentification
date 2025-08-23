@echo off
echo ============================================================
echo           AI Trading System - Windows Environment Fix
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or later from https://python.org
    pause
    exit /b 1
)

echo Current Python version:
python --version
echo.

echo ============================================================
echo                    Fixing Package Issues
echo ============================================================
echo.

echo Step 1: Running numpy fix script...
python fix_numpy.py

echo.
echo Step 2: Installing remaining packages...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Step 3: Installing core trading system packages...
python -m pip install --no-cache-dir -r requirements.txt

echo.
echo Step 4: Verifying installation...
python -c "import numpy, pandas, sklearn; print('✅ Core packages installed successfully')"

if errorlevel 1 (
    echo.
    echo ❌ Package verification failed. Trying alternative installation...
    echo.
    echo Installing packages one by one...
    python -m pip install numpy==1.24.3 --force-reinstall --no-cache-dir
    python -m pip install pandas==1.5.3 --no-cache-dir
    python -m pip install scikit-learn==1.3.0 --no-cache-dir
    python -m pip install matplotlib==3.7.1 --no-cache-dir
    python -m pip install streamlit --no-cache-dir
    python -m pip install yfinance --no-cache-dir
    python -m pip install lightgbm --no-cache-dir
    python -m pip install xgboost --no-cache-dir
    python -m pip install python-dotenv --no-cache-dir
    python -m pip install aiohttp --no-cache-dir
    python -m pip install plotly --no-cache-dir
)

echo.
echo ============================================================
echo                    Environment Ready!
echo ============================================================
echo.
echo ✅ Package installation completed
echo 📝 Next steps:
echo    1. Update your .env file with API keys
echo    2. Run: python start.py
echo    3. Open dashboard at: http://localhost:8000
echo.
echo For AI features, add either:
echo    - GEMINI_API_KEY (recommended)
echo    - AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT
echo.

pause