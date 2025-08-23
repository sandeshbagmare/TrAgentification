#!/usr/bin/env python3
"""
AI Trading System Startup Script
Comprehensive startup script that validates environment, installs dependencies, runs tests, and starts the application
"""
import os
import sys
import subprocess
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import importlib.util

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSystemStarter:
    """Main class for starting the AI Trading System"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / '.env'
        self.requirements_file = self.project_root / 'requirements.txt'
        self.docker_compose_file = self.project_root / 'docker-compose.yml'
        
        # Required environment variables
        self.required_env_vars = [
            'UPSTOX_API_KEY',
            'UPSTOX_API_SECRET'
        ]
        
        # Optional AI API keys (at least one should be provided)
        self.optional_ai_keys = [
            'GEMINI_API_KEY',
            'AZURE_OPENAI_API_KEY'
        ]
        
    def validate_environment(self) -> bool:
        """Validate that the environment is properly configured"""
        logger.info("🔍 Validating environment...")
        
        # Check if .env file exists
        if not self.env_file.exists():
            logger.error("❌ .env file not found. Please create one with your API keys.")
            return False
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv(self.env_file)
        except ImportError:
            logger.error("❌ python-dotenv not installed. Installing...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'python-dotenv'])
            from dotenv import load_dotenv
            load_dotenv(self.env_file)
        
        # Check required environment variables
        missing_vars = []
        for var in self.required_env_vars:
            if not os.getenv(var) or os.getenv(var) == f'your_{var.lower()}_here':
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
            logger.error("Please update your .env file with the correct values.")
            return False
        
        # Check for at least one AI API key
        ai_keys_available = []
        for key in self.optional_ai_keys:
            value = os.getenv(key)
            if value and value != f'your_{key.lower()}_here' and len(value) > 10:
                ai_keys_available.append(key)
        
        if not ai_keys_available:
            logger.warning("⚠️ No AI API keys configured. The system will work with technical analysis only.")
            logger.warning("Consider adding GEMINI_API_KEY or AZURE_OPENAI_API_KEY for enhanced AI features.")
        else:
            logger.info(f"✅ AI integrations available: {', '.join(ai_keys_available)}")
        
        logger.info("✅ Environment validation completed.")
        return True
    
    def install_dependencies(self) -> bool:
        """Install required Python dependencies"""
        logger.info("📦 Installing dependencies...")
        
        if not self.requirements_file.exists():
            logger.error("❌ requirements.txt not found.")
            return False
        
        try:
            # Upgrade pip first
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ])
            
            # Install requirements
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', str(self.requirements_file)
            ])
            
            logger.info("✅ Dependencies installed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            logger.error("Please check your internet connection and try again.")
            return False
    
    def check_imports(self) -> bool:
        """Check if all required modules can be imported"""
        logger.info("🔍 Checking imports...")
        
        required_modules = [
            'pandas', 'numpy', 'streamlit', 'plotly', 'requests',
            'sklearn', 'lightgbm', 'xgboost', 'tensorflow'
        ]
        
        optional_modules = [
            'google.generativeai', 'openai', 'azure.identity'
        ]
        
        failed_imports = []
        
        # Check required modules
        for module in required_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                failed_imports.append(module)
        
        if failed_imports:
            logger.error(f"❌ Failed to import required modules: {', '.join(failed_imports)}")
            return False
        
        # Check optional modules
        optional_failed = []
        for module in optional_modules:
            try:
                importlib.import_module(module)
            except ImportError:
                optional_failed.append(module)
        
        if optional_failed:
            logger.warning(f"⚠️ Optional modules not available: {', '.join(optional_failed)}")
            logger.warning("Some AI features may not be available.")
        
        logger.info("✅ Import check completed.")
        return True
    
    def run_tests(self) -> bool:
        """Run the test suite"""
        logger.info("🧪 Running tests...")
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("✅ All tests passed.")
                return True
            else:
                logger.warning("⚠️ Some tests failed, but continuing startup...")
                logger.warning(f"Test output: {result.stdout}")
                logger.warning(f"Test errors: {result.stderr}")
                return True  # Don't fail startup for test failures
        except FileNotFoundError:
            logger.warning("⚠️ pytest not found. Skipping tests.")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Error running tests: {e}")
            return True
    
    def start_trading_system(self, skip_training: bool = False) -> bool:
        """Start the main trading system with optional automated training"""
        logger.info("🚀 Starting AI Trading System...")
        
        try:
            # Import and start the main trading system
            sys.path.append(str(self.project_root))
            from main import initialize_trading_system, run_trading_system
            
            # Run automated training if not skipped
            if not skip_training:
                logger.info("🤖 Running automated training first...")
                
                # Import and run automated trainer
                from agents.automated_trainer import run_automated_training
                
                # Run training in the background
                import asyncio
                
                async def run_training():
                    training_result = await run_automated_training(365)  # 1 year of data
                    if training_result.get('success'):
                        logger.info("✅ Automated training completed successfully")
                        logger.info(f"📊 Trained on {training_result.get('symbols_processed', 0)} symbols")
                        logger.info(f"🤖 {training_result.get('total_samples', 0)} samples processed")
                    else:
                        logger.warning(f"⚠️ Training failed: {training_result.get('error', 'Unknown error')}")
                        logger.warning("System will continue with existing models or technical analysis only")
                
                # Run training
                try:
                    asyncio.run(run_training())
                except Exception as e:
                    logger.warning(f"⚠️ Training failed: {str(e)}")
                    logger.warning("Continuing without automated training...")
            
            # Initialize components
            components = initialize_trading_system()
            logger.info("✅ Trading system components initialized.")
            
            # Initialize P&L tracking
            from agents.pnl_tracker import pnl_tracker
            logger.info("✅ P&L tracking system initialized.")
            
            # Start the trading system in background
            import threading
            trading_thread = threading.Thread(target=run_trading_system, daemon=True)
            trading_thread.start()
            
            logger.info("✅ Trading system started successfully.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start trading system: {e}")
            return False
    
    def start_dashboard(self) -> bool:
        """Start the Streamlit dashboard"""
        logger.info("📊 Starting dashboard...")
        
        try:
            dashboard_path = self.project_root / 'dashboard' / 'app.py'
            if not dashboard_path.exists():
                logger.error("❌ Dashboard app.py not found.")
                return False
            
            # Start Streamlit dashboard
            subprocess.Popen([
                sys.executable, '-m', 'streamlit', 'run', 
                str(dashboard_path), '--server.port', '8000'
            ])
            
            logger.info("✅ Dashboard started on http://localhost:8000")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return False
    
    def start_with_docker(self) -> bool:
        """Start the system using Docker Compose"""
        logger.info("🐳 Starting with Docker...")
        
        if not self.docker_compose_file.exists():
            logger.error("❌ docker-compose.yml not found.")
            return False
        
        try:
            # Build and start with docker-compose
            subprocess.check_call(['docker-compose', 'build'], cwd=self.project_root)
            subprocess.Popen(['docker-compose', 'up'], cwd=self.project_root)
            
            logger.info("✅ Docker containers started successfully.")
            logger.info("📊 Dashboard will be available at http://localhost:8000")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Docker startup failed: {e}")
            return False
        except FileNotFoundError:
            logger.error("❌ Docker or docker-compose not found.")
            logger.error("Please install Docker and docker-compose or use native startup.")
            return False
    
    def display_startup_info(self):
        """Display startup information and instructions"""
        logger.info("=" * 60)
        logger.info("🤖 AI Trading System")
        logger.info("=" * 60)
        logger.info("")
        logger.info("📊 Dashboard: http://localhost:8000")
        logger.info("📝 Logs: logs/")
        logger.info("🔧 Configuration: .env")
        logger.info("")
        logger.info("🚨 IMPORTANT NOTES:")
        logger.info("- This system trades real money. Use with caution!")
        logger.info("- Start with paper trading or small amounts")
        logger.info("- Monitor the dashboard regularly")
        logger.info("- Check logs for any errors")
        logger.info("")
        logger.info("📱 Available Features:")
        
        # Check which AI integrations are available
        ai_features = []
        if os.getenv('GEMINI_API_KEY') and len(os.getenv('GEMINI_API_KEY', '')) > 10:
            ai_features.append("🤖 Gemini AI Analysis")
        if os.getenv('AZURE_OPENAI_API_KEY') and len(os.getenv('AZURE_OPENAI_API_KEY', '')) > 10:
            ai_features.append("🤖 Azure OpenAI Analysis")
        
        if ai_features:
            for feature in ai_features:
                logger.info(f"  {feature}")
        else:
            logger.info("  📈 Technical Analysis Only")
        
        logger.info("  📊 Risk Management")
        logger.info("  🎯 Strategy Orchestration")
        logger.info("  📈 Real-time Dashboard")
        logger.info("  📋 Order Execution")
        logger.info("")
        logger.info("🛑 To stop the system: Press Ctrl+C")
        logger.info("=" * 60)
    
    def run(self, use_docker: bool = False, skip_training: bool = False):
        """Main entry point to start the trading system"""
        try:
            # Display banner
            self.display_startup_info()
            
            if use_docker:
                # Docker startup
                if not self.start_with_docker():
                    logger.error("❌ Docker startup failed. Try native startup.")
                    return False
            else:
                # Native startup
                if not self.validate_environment():
                    return False
                
                if not self.install_dependencies():
                    return False
                
                if not self.check_imports():
                    return False
                
                if not self.run_tests():
                    logger.warning("⚠️ Test issues detected, but continuing...")
                
                if not self.start_trading_system(skip_training=skip_training):
                    return False
                
                if not self.start_dashboard():
                    return False
            
            # Keep the script running
            logger.info("✅ System started successfully!")
            logger.info("🔄 System is running. Press Ctrl+C to stop.")
            
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                logger.info("🛑 Shutdown signal received.")
                logger.info("👋 AI Trading System stopped.")
                
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}")
            return False

def main():
    """Main function with command line argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Trading System Startup Script')
    parser.add_argument('--docker', action='store_true', help='Start using Docker')
    parser.add_argument('--test-only', action='store_true', help='Run tests only')
    parser.add_argument('--dashboard-only', action='store_true', help='Start dashboard only')
    parser.add_argument('--skip-training', action='store_true', help='Skip automated training')
    parser.add_argument('--train-only', action='store_true', help='Run training only')
    
    args = parser.parse_args()
    
    starter = TradingSystemStarter()
    
    if args.test_only:
        starter.run_tests()
    elif args.dashboard_only:
        starter.validate_environment()
        starter.start_dashboard()
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            logger.info("🛑 Dashboard stopped.")
    elif args.train_only:
        # Run training only
        import asyncio
        
        async def run_training_only():
            from agents.automated_trainer import run_automated_training
            result = await run_automated_training(365)
            if result.get('success'):
                logger.info("✅ Training completed successfully!")
            else:
                logger.error(f"❌ Training failed: {result.get('error')}")
        
        asyncio.run(run_training_only())
    else:
        starter.run(use_docker=args.docker, skip_training=args.skip_training)

if __name__ == "__main__":
    main()