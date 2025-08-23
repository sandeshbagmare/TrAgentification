@echo off
echo ============================================
echo Advanced Windows Numpy Fix for Trading System
echo ============================================
echo.

echo Step 1: Clearing pip cache...
python -m pip cache purge >nul 2>&1

echo Step 2: Upgrading pip...
python -m pip install --upgrade pip

echo Step 3: Force removing numpy (ignoring errors)...
python -m pip uninstall numpy -y >nul 2>&1
python -m pip uninstall numpy -y >nul 2>&1

echo Step 4: Clearing any corrupted numpy installations...
python -c "import shutil, os, site; [shutil.rmtree(os.path.join(sp, 'numpy'), ignore_errors=True) for sp in site.getsitepackages() if os.path.exists(os.path.join(sp, 'numpy'))]" >nul 2>&1

echo Step 5: Installing Windows-compatible numpy...
python -m pip install numpy==1.24.3 --no-cache-dir --force-reinstall --no-deps

echo Step 6: Verifying numpy installation...
python -c "import numpy; print(f'Numpy version: {numpy.__version__}')"

if %ERRORLEVEL% EQU 0 (
    echo ✅ Numpy installed successfully!
    echo.
    echo Step 7: Installing project requirements...
    python -m pip install -r requirements.txt --no-cache-dir
    echo.
    echo ✅ All requirements installed!
    echo.
    echo 🚀 Your trading system is ready to run!
    echo Use: python start.py
) else (
    echo ❌ Numpy installation failed.
    echo.
    echo Trying alternative fix...
    python fix_numpy_advanced.py
)

echo.
echo Press any key to exit...
pause >nul