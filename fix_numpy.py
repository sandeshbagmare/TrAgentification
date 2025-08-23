#!/usr/bin/env python3
"""
Numpy Fix Script for AI Trading System
Resolves numpy installation issues on Windows
"""
import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_numpy_installation():
    """Fix numpy installation issues"""
    logger.info("🔧 Fixing numpy installation issues...")
    
    try:
        # Step 1: Uninstall all numpy packages completely
        logger.info("📦 Step 1: Removing all numpy installations...")
        
        # Try different uninstall commands
        uninstall_commands = [
            [sys.executable, '-m', 'pip', 'uninstall', 'numpy', '-y'],
            [sys.executable, '-m', 'pip', 'uninstall', 'numpy', '--force', '-y'],
            [sys.executable, '-c', 'import numpy; print("Numpy found, removing..."); import shutil; import numpy; shutil.rmtree(numpy.__path__[0])']
        ]
        
        for cmd in uninstall_commands:
            try:
                subprocess.run(cmd, check=False, capture_output=True)
                logger.info(f"✅ Executed: {' '.join(cmd)}")
            except Exception as e:
                logger.warning(f"⚠️ Command failed: {e}")
        
        # Step 2: Clear pip cache
        logger.info("🧹 Step 2: Clearing pip cache...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'cache', 'purge'], check=False)
        except Exception as e:
            logger.warning(f"⚠️ Cache clear failed: {e}")
        
        # Step 3: Upgrade pip
        logger.info("⬆️ Step 3: Upgrading pip...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        
        # Step 4: Install numpy with specific version
        logger.info("📦 Step 4: Installing numpy...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 
            'numpy==1.24.3',  # Stable version that works well
            '--no-cache-dir',
            '--force-reinstall'
        ], check=True)
        
        # Step 5: Verify installation
        logger.info("✅ Step 5: Verifying numpy installation...")
        result = subprocess.run([
            sys.executable, '-c', 'import numpy; print(f"Numpy {numpy.__version__} installed successfully")'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ {result.stdout.strip()}")
            return True
        else:
            logger.error(f"❌ Verification failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Installation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def install_compatible_packages():
    """Install packages with compatible versions"""
    logger.info("📦 Installing compatible package versions...")
    
    # Compatible package versions for Windows
    compatible_packages = [
        'numpy==1.24.3',
        'pandas==1.5.3',
        'scikit-learn==1.3.0',
        'matplotlib==3.7.1',
        'scipy==1.11.1'
    ]
    
    for package in compatible_packages:
        try:
            logger.info(f"Installing {package}...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                package, '--no-cache-dir'
            ], check=True)
            logger.info(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Failed to install {package}: {e}")

def main():
    """Main function"""
    logger.info("🚀 Starting AI Trading System Package Fix")
    logger.info("=" * 50)
    
    # Fix numpy first
    if fix_numpy_installation():
        logger.info("✅ Numpy fix completed successfully")
        
        # Install other compatible packages
        install_compatible_packages()
        
        logger.info("=" * 50)
        logger.info("🎉 Package fix completed!")
        logger.info("You can now run: python start.py")
        
    else:
        logger.error("❌ Numpy fix failed")
        logger.info("Try running as administrator or use conda instead of pip")
        logger.info("Alternative: conda install numpy pandas scikit-learn")

if __name__ == "__main__":
    main()