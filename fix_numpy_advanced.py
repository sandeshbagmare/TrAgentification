"""
Advanced Numpy Fix for Windows
Comprehensive solution for corrupted numpy installations
"""
import subprocess
import sys
import os
import shutil
import site
import pkg_resources
from pathlib import Path

def run_command(cmd, ignore_errors=False):
    """Run a command and return success status"""
    try:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✓ Success: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        if not ignore_errors:
            print(f"✗ Error: {e.stderr}")
        return False

def find_and_remove_numpy_files():
    """Find and remove all numpy files from the system"""
    print("🔍 Searching for numpy installations...")
    
    # Get all site-packages directories
    site_packages = site.getsitepackages()
    if hasattr(site, 'getusersitepackages'):
        site_packages.append(site.getusersitepackages())
    
    numpy_paths = []
    
    for site_pkg in site_packages:
        if os.path.exists(site_pkg):
            # Look for numpy directories
            numpy_dir = os.path.join(site_pkg, 'numpy')
            if os.path.exists(numpy_dir):
                numpy_paths.append(numpy_dir)
            
            # Look for numpy dist-info
            for item in os.listdir(site_pkg):
                if item.startswith('numpy') and ('dist-info' in item or 'egg-info' in item):
                    numpy_paths.append(os.path.join(site_pkg, item))
    
    # Remove found numpy installations
    for path in numpy_paths:
        try:
            print(f"🗑️ Removing: {path}")
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
            print(f"✓ Removed: {path}")
        except Exception as e:
            print(f"⚠️ Could not remove {path}: {e}")

def clear_pip_cache():
    """Clear pip cache"""
    print("🧹 Clearing pip cache...")
    run_command(f"{sys.executable} -m pip cache purge", ignore_errors=True)

def upgrade_pip():
    """Upgrade pip to latest version"""
    print("⬆️ Upgrading pip...")
    return run_command(f"{sys.executable} -m pip install --upgrade pip")

def fix_numpy_installation():
    """Main function to fix numpy installation"""
    print("🔧 Starting Advanced Numpy Fix for Windows...")
    print("=" * 50)
    
    # Step 1: Clear pip cache
    clear_pip_cache()
    
    # Step 2: Upgrade pip
    upgrade_pip()
    
    # Step 3: Force uninstall numpy (ignore errors)
    print("🗑️ Force uninstalling numpy...")
    run_command(f"{sys.executable} -m pip uninstall numpy -y", ignore_errors=True)
    
    # Step 4: Remove numpy files manually
    find_and_remove_numpy_files()
    
    # Step 5: Install numpy with specific Windows-compatible version
    print("📦 Installing Windows-compatible numpy...")
    numpy_versions = [
        "1.24.3",  # Known good version for Windows
        "1.24.4",  # Alternative
        "1.23.5"   # Fallback
    ]
    
    for version in numpy_versions:
        print(f"🔄 Trying numpy=={version}...")
        success = run_command(
            f"{sys.executable} -m pip install numpy=={version} --no-cache-dir --force-reinstall --no-deps"
        )
        if success:
            print(f"✅ Successfully installed numpy=={version}")
            break
        else:
            print(f"❌ Failed to install numpy=={version}, trying next version...")
    
    # Step 6: Install numpy dependencies
    print("📦 Installing numpy dependencies...")
    dependencies = [
        "setuptools",
        "wheel",
        "Cython"
    ]
    
    for dep in dependencies:
        run_command(f"{sys.executable} -m pip install {dep} --upgrade", ignore_errors=True)
    
    # Step 7: Verify installation
    print("🔍 Verifying numpy installation...")
    try:
        import numpy as np
        print(f"✅ Numpy successfully installed! Version: {np.__version__}")
        print(f"✅ Numpy location: {np.__file__}")
        return True
    except ImportError as e:
        print(f"❌ Numpy import failed: {e}")
        return False

def install_project_requirements():
    """Install project requirements after fixing numpy"""
    print("📦 Installing project requirements...")
    
    # Install requirements with Windows-compatible versions
    requirements = [
        "pandas==1.5.3",
        "scikit-learn==1.3.0", 
        "lightgbm>=3.3.5",
        "xgboost>=1.5.0",
        "flask>=2.0.0",
        "requests>=2.26.0",
        "websockets>=10.0",
        "python-dotenv>=0.19.0",
        "aiohttp>=3.8.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "yfinance>=0.2.18",
        "statsmodels>=0.13.0",
        "transformers>=4.18.0",
        "scipy==1.11.1",
        "google-generativeai>=0.3.0",
        "openai>=1.0.0",
        "azure-identity>=1.12.0",
        "matplotlib==3.7.1",
        "streamlit>=1.28.0",
        "prometheus-client>=0.14.0",
        "plotly>=5.10.0",
        "altair>=5.0.0",
        "seaborn>=0.12.0"
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        success = run_command(f"{sys.executable} -m pip install {req} --no-cache-dir")
        if success:
            print(f"✅ {req} installed successfully")
        else:
            print(f"⚠️ {req} installation failed, continuing...")

if __name__ == "__main__":
    print("🚀 Advanced Numpy Fix for Windows Trading System")
    print("This script will fix numpy installation issues comprehensively")
    print("=" * 60)
    
    # Fix numpy installation
    if fix_numpy_installation():
        print("\n🎉 Numpy fix completed successfully!")
        
        # Ask user if they want to install project requirements
        response = input("\n📦 Do you want to install project requirements now? (y/n): ")
        if response.lower() in ['y', 'yes']:
            install_project_requirements()
            print("\n✅ Project setup completed!")
        else:
            print("\n💡 You can install requirements later using: pip install -r requirements.txt")
    else:
        print("\n❌ Numpy fix failed. Please try the manual steps below:")
        print("1. Create a new virtual environment: python -m venv fresh_env")
        print("2. Activate it: fresh_env\\Scripts\\activate")
        print("3. Install numpy: pip install numpy==1.24.3")
        print("4. Install other requirements: pip install -r requirements.txt")
    
    print("\n" + "=" * 60)
    print("Fix completed. Press Enter to exit...")
    input()