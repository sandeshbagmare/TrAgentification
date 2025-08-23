# Windows Troubleshooting Guide - AI Trading System

## Numpy Installation Issues

### Problem: "Invalid requirement: 'numpy==1.25.1\\'"
This is a common Windows issue with corrupted numpy installations or path parsing problems.

### Quick Fix (Recommended)
Run the automated fix script:
```bash
# Option 1: Run the batch file
fix_environment_advanced.bat

# Option 2: Run the Python script
python fix_numpy_advanced.py
```

### Manual Fix Steps

#### Step 1: Clean Environment
```bash
# Clear pip cache
python -m pip cache purge

# Upgrade pip
python -m pip install --upgrade pip

# Force uninstall numpy (ignore errors)
python -m pip uninstall numpy -y
```

#### Step 2: Remove Corrupted Files
```bash
# Remove numpy directories manually
python -c "import shutil, os, site; [shutil.rmtree(os.path.join(sp, 'numpy'), ignore_errors=True) for sp in site.getsitepackages() if os.path.exists(os.path.join(sp, 'numpy'))]"
```

#### Step 3: Install Windows-Compatible Numpy
```bash
# Install specific Windows-compatible version
python -m pip install numpy==1.24.3 --no-cache-dir --force-reinstall --no-deps

# Verify installation
python -c "import numpy; print(f'Numpy version: {numpy.__version__}')"
```

#### Step 4: Install Project Requirements
```bash
# Install remaining requirements
python -m pip install -r requirements.txt --no-cache-dir
```

### Alternative Solutions

#### Solution 1: Fresh Virtual Environment
```bash
# Create new virtual environment
python -m venv fresh_trading_env

# Activate it
fresh_trading_env\Scripts\activate

# Install numpy first
pip install numpy==1.24.3

# Install project requirements
pip install -r requirements.txt
```

#### Solution 2: Use Conda (If Available)
```bash
# Create conda environment
conda create -n trading python=3.11

# Activate environment
conda activate trading

# Install numpy via conda
conda install numpy=1.24.3

# Install remaining via pip
pip install -r requirements.txt
```

#### Solution 3: Pre-compiled Wheels
```bash
# Download pre-compiled wheel from PyPI
# Visit: https://pypi.org/project/numpy/1.24.3/#files
# Download: numpy-1.24.3-cp311-cp311-win_amd64.whl

# Install directly
pip install numpy-1.24.3-cp311-cp311-win_amd64.whl
```

### Common Windows Issues & Solutions

#### Issue 1: "Cannot uninstall numpy None"
**Solution:** Use the advanced fix script which manually removes numpy files.

#### Issue 2: "Microsoft Visual C++ 14.0 is required"
**Solution:** 
1. Install Microsoft C++ Build Tools
2. Or use pre-compiled wheels: `pip install --only-binary=all numpy==1.24.3`

#### Issue 3: "Permission denied"
**Solution:** 
1. Run Command Prompt as Administrator
2. Or use: `pip install --user numpy==1.24.3`

#### Issue 4: "Failed building wheel for numpy"
**Solution:**
1. Use pre-compiled wheels: `pip install --only-binary=all numpy`
2. Or upgrade setuptools: `pip install --upgrade setuptools wheel`

### Testing Your Installation

After fixing numpy, test the installation:

```python
# Test script
import numpy as np
import pandas as pd
import sklearn
import lightgbm
import xgboost

print("✅ All core packages imported successfully!")
print(f"Numpy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
```

### Running the Trading System

Once numpy is fixed, start the trading system:

```bash
# Start the complete system
python start.py

# Or start individual components
python dashboard/app.py  # Dashboard only
python agents/automated_trainer.py  # Training only
```

### Docker Alternative (If Issues Persist)

If Windows package conflicts continue, use Docker:

```bash
# Build Docker image
docker build -t ai-trading-system .

# Run the system
docker run -d -p 8501:8501 ai-trading-system
```

### Contact & Support

If issues persist:
1. Check the exact Python version: `python --version`
2. Check pip version: `pip --version`
3. Try the fresh virtual environment approach
4. Consider using Anaconda/Miniconda for package management

### Version Compatibility Matrix

| Python | Numpy | Pandas | Scikit-learn | Status |
|--------|-------|--------|--------------|--------|
| 3.11.x | 1.24.3 | 1.5.3 | 1.3.0 | ✅ Recommended |
| 3.10.x | 1.24.3 | 1.5.3 | 1.2.2 | ✅ Supported |
| 3.9.x  | 1.23.5 | 1.5.2 | 1.2.0 | ⚠️ Limited |

---
**Last Updated:** August 2024
**Tested On:** Windows 11 23H2, Windows 10 22H2