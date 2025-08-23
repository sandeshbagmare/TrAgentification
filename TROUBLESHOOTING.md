# 🛠️ Windows Installation Troubleshooting Guide

## 🚨 Common Issues and Solutions

### Issue 1: Numpy Installation Error
```
ERROR: Invalid requirement: 'numpy==1.25.1\\': Expected end or semicolon
```

**Solution:**
1. **Run the fix script:**
   ```batch
   fix_environment.bat
   ```
   OR
   ```bash
   python fix_numpy.py
   ```

2. **Manual fix:**
   ```bash
   # Step 1: Completely remove numpy
   pip uninstall numpy -y
   pip cache purge
   
   # Step 2: Install specific version
   pip install numpy==1.24.3 --no-cache-dir --force-reinstall
   
   # Step 3: Install other packages
   pip install pandas==1.5.3 scikit-learn==1.3.0
   ```

### Issue 2: TensorFlow Installation Problems

**Solution:**
TensorFlow is now optional. The system will work without it using LightGBM and XGBoost only.

To install TensorFlow (optional):
```bash
pip install tensorflow-cpu  # For CPU-only version (recommended for trading)
```

### Issue 3: Package Conflicts

**Solution:**
1. **Create a fresh virtual environment:**
   ```bash
   python -m venv trading_env
   trading_env\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Alternative - Use conda:**
   ```bash
   conda create -n trading python=3.9
   conda activate trading
   conda install numpy pandas scikit-learn matplotlib
   pip install -r requirements.txt
   ```

### Issue 4: Permission Errors

**Solution:**
1. **Run Command Prompt as Administrator**
2. **Or install for user only:**
   ```bash
   pip install --user numpy pandas scikit-learn
   ```

### Issue 5: "Microsoft Visual C++ 14.0 is required"

**Solution:**
1. **Install Microsoft C++ Build Tools:**
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "C++ build tools" workload

2. **Or use pre-compiled wheels:**
   ```bash
   pip install --only-binary=all numpy pandas scikit-learn
   ```

## 🔧 Step-by-Step Fix Process

### Option 1: Automated Fix (Recommended)
```batch
# Double-click this file:
fix_environment.bat
```

### Option 2: Manual Fix
```bash
# 1. Clean environment
pip uninstall numpy pandas scikit-learn matplotlib -y
pip cache purge

# 2. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 3. Install core packages one by one
pip install numpy==1.24.3 --no-cache-dir
pip install pandas==1.5.3 --no-cache-dir  
pip install scikit-learn==1.3.0 --no-cache-dir
pip install matplotlib==3.7.1 --no-cache-dir

# 4. Install trading-specific packages
pip install yfinance streamlit plotly lightgbm xgboost
pip install python-dotenv aiohttp requests

# 5. Install AI packages (optional)
pip install google-generativeai openai
```

### Option 3: Conda Environment (Most Reliable)
```bash
# Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

# Create environment
conda create -n trading python=3.9 numpy pandas scikit-learn matplotlib
conda activate trading

# Install remaining packages
pip install yfinance streamlit plotly lightgbm xgboost
pip install python-dotenv aiohttp requests google-generativeai
```

## ✅ Verification

After installation, verify everything works:
```python
python -c "import numpy, pandas, sklearn, lightgbm, xgboost; print('✅ All packages installed successfully')"
```

## 🚀 Quick Start After Fix

1. **Configure your API keys** in `.env` file:
   ```env
   UPSTOX_API_KEY=your_upstox_api_key
   UPSTOX_API_SECRET=your_upstox_secret
   GEMINI_API_KEY=your_gemini_key  # Optional
   ```

2. **Start the system:**
   ```bash
   python start.py
   ```

3. **Access dashboard:** http://localhost:8000

## 📞 Still Having Issues?

1. **Check Python version:** `python --version` (Should be 3.8+)
2. **Check pip version:** `pip --version` (Should be latest)
3. **Try in virtual environment** (most reliable solution)
4. **Use conda instead of pip** (alternative package manager)

## 🎯 Alternative: Docker Setup (Skip Installation Issues)

If you continue having package issues, use Docker:

```bash
# Install Docker Desktop for Windows
# Then run:
docker-compose up --build
```

This bypasses all Windows package installation issues!