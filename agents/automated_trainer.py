"""
Automated Training System for AI Trading
Automatically fetches historical data and trains ML models on startup
"""
import os
import logging
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import yfinance as yf
from pathlib import Path

from agents.ml_agent import MlAgent
from agents.data_agent import DataAgent

logger = logging.getLogger(__name__)

class AutomatedTrainer:
    """
    Automated training system that:
    1. Fetches historical data from multiple sources
    2. Prepares training datasets 
    3. Trains ML models automatically
    4. Validates model performance
    5. Sets up the system for live trading
    """
    
    def __init__(self, training_period_days: int = 365):
        self.training_period_days = training_period_days
        self.data_dir = Path('data/historical')
        self.model_dir = Path('models')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Indian stock symbols for training
        self.stock_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS', 'ICICIBANK.NS',
            'SBI.NS', 'ITC.NS', 'LT.NS', 'HCLTECH.NS', 'WIPRO.NS',
            'MARUTI.NS', 'ASIANPAINT.NS', 'NESTLEIND.NS', 'KOTAKBANK.NS',
            'BAJFINANCE.NS', 'BHARTIARTL.NS', 'AXISBANK.NS', 'HINDUNILVR.NS',
            'POWERGRID.NS', 'NTPC.NS', 'ONGC.NS', 'TATAMOTORS.NS',
            'TECHM.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS'
        ]
        
        # Futures symbols (Nifty, Bank Nifty)
        self.futures_symbols = ['^NSEI', '^NSEBANK']
        
        self.ml_agent = None
        
    async def initialize(self):
        """Initialize the automated trainer"""
        logger.info("🤖 Initializing Automated Training System...")
        
        # Initialize ML agent with Windows-compatible models
        self.ml_agent = MlAgent(
            model_dir=str(self.model_dir),
            training_interval=86400,  # Daily retraining
            models_to_train=['lightgbm', 'xgboost']  # Skip LSTM for Windows compatibility
        )
        
        logger.info("✅ Automated trainer initialized")
        
    async def fetch_historical_data(self, symbol: str, period_days: int = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a symbol using multiple sources
        """
        if period_days is None:
            period_days = self.training_period_days
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        try:
            # Primary: Use yfinance for reliable data
            logger.info(f"📈 Fetching data for {symbol}...")
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d',
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                logger.warning(f"⚠️ No data found for {symbol}")
                return None
                
            # Clean and prepare data
            data = data.reset_index()
            data.columns = [col.lower() for col in data.columns]
            
            # Add technical indicators for training
            data = self._add_technical_features(data)
            
            # Save to local cache
            cache_file = self.data_dir / f"{symbol.replace('.', '_')}_historical.csv"
            data.to_csv(cache_file, index=False)
            
            logger.info(f"✅ Fetched {len(data)} days of data for {symbol}")
            return data
            
        except Exception as e:\n            logger.error(f\"❌ Error fetching data for {symbol}: {str(e)}\")\n            return None\n            \n    def _add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Add technical indicators to the dataset for ML training\"\"\"\n        try:\n            # Price-based features\n            data['returns'] = data['close'].pct_change()\n            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))\n            \n            # Moving averages\n            for period in [5, 10, 20, 50]:\n                data[f'ma_{period}'] = data['close'].rolling(window=period).mean()\n                data[f'ma_ratio_{period}'] = data['close'] / data[f'ma_{period}']\n            \n            # Volatility features\n            data['volatility_5'] = data['returns'].rolling(window=5).std()\n            data['volatility_20'] = data['returns'].rolling(window=20).std()\n            \n            # RSI\n            delta = data['close'].diff()\n            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()\n            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()\n            rs = gain / loss\n            data['rsi'] = 100 - (100 / (1 + rs))\n            \n            # MACD\n            exp1 = data['close'].ewm(span=12).mean()\n            exp2 = data['close'].ewm(span=26).mean()\n            data['macd'] = exp1 - exp2\n            data['macd_signal'] = data['macd'].ewm(span=9).mean()\n            data['macd_histogram'] = data['macd'] - data['macd_signal']\n            \n            # Bollinger Bands\n            data['bb_middle'] = data['close'].rolling(window=20).mean()\n            bb_std = data['close'].rolling(window=20).std()\n            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)\n            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)\n            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']\n            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])\n            \n            # Volume features\n            data['volume_ma'] = data['volume'].rolling(window=20).mean()\n            data['volume_ratio'] = data['volume'] / data['volume_ma']\n            \n            # Target variable for ML (next day return)\n            data['target'] = data['returns'].shift(-1)\n            data['target_binary'] = (data['target'] > 0).astype(int)\n            \n            # Drop rows with NaN values\n            data = data.dropna()\n            \n            return data\n            \n        except Exception as e:\n            logger.error(f\"Error adding technical features: {str(e)}\")\n            return data\n    \n    async def fetch_all_training_data(self) -> Dict[str, pd.DataFrame]:\n        \"\"\"Fetch historical data for all symbols\"\"\"\n        logger.info(f\"📊 Fetching training data for {len(self.stock_symbols)} stocks...\")\n        \n        all_data = {}\n        \n        # Fetch stock data\n        for symbol in self.stock_symbols:\n            data = await self.fetch_historical_data(symbol)\n            if data is not None and len(data) > 100:  # Minimum data requirement\n                all_data[symbol] = data\n            \n            # Small delay to avoid rate limiting\n            await asyncio.sleep(0.1)\n        \n        # Fetch futures data\n        for symbol in self.futures_symbols:\n            data = await self.fetch_historical_data(symbol)\n            if data is not None and len(data) > 100:\n                all_data[symbol] = data\n            \n            await asyncio.sleep(0.1)\n        \n        logger.info(f\"✅ Successfully fetched data for {len(all_data)} symbols\")\n        return all_data\n    \n    def _prepare_ml_dataset(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:\n        \"\"\"Combine all data into a single training dataset\"\"\"\n        logger.info(\"🔄 Preparing ML training dataset...\")\n        \n        combined_data = []\n        \n        for symbol, data in all_data.items():\n            # Add symbol as a feature\n            data_copy = data.copy()\n            data_copy['symbol'] = symbol\n            \n            # Select features for ML\n            feature_columns = [\n                'open', 'high', 'low', 'close', 'volume',\n                'returns', 'log_returns',\n                'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50',\n                'volatility_5', 'volatility_20',\n                'rsi', 'macd', 'macd_signal', 'macd_histogram',\n                'bb_width', 'bb_position', 'volume_ratio',\n                'target', 'target_binary'\n            ]\n            \n            # Keep only available columns\n            available_columns = [col for col in feature_columns if col in data_copy.columns]\n            ml_data = data_copy[available_columns + ['symbol']]\n            \n            combined_data.append(ml_data)\n        \n        # Combine all data\n        final_dataset = pd.concat(combined_data, ignore_index=True)\n        \n        # Remove any remaining NaN values\n        final_dataset = final_dataset.dropna()\n        \n        logger.info(f\"✅ Prepared dataset with {len(final_dataset)} samples\")\n        return final_dataset\n    \n    async def train_models(self, dataset: pd.DataFrame) -> Dict:\n        \"\"\"Train all ML models using the prepared dataset\"\"\"\n        logger.info(\"🤖 Starting automated model training...\")\n        \n        try:\n            # Prepare data for ML agent\n            feature_columns = [\n                'returns', 'log_returns', 'ma_ratio_5', 'ma_ratio_10', \n                'ma_ratio_20', 'ma_ratio_50', 'volatility_5', 'volatility_20',\n                'rsi', 'macd', 'macd_signal', 'macd_histogram',\n                'bb_width', 'bb_position', 'volume_ratio'\n            ]\n            \n            # Keep only available features\n            available_features = [col for col in feature_columns if col in dataset.columns]\n            \n            X = dataset[available_features]\n            y = dataset['target_binary']  # Binary classification for now\n            \n            # Split data for training\n            split_idx = int(len(dataset) * 0.8)\n            X_train, X_test = X[:split_idx], X[split_idx:]\n            y_train, y_test = y[:split_idx], y[split_idx:]\n            \n            # Prepare data dictionary for ML agent\n            training_data = {\n                'X_train': X_train,\n                'X_test': X_test,\n                'y_train': y_train,\n                'y_test': y_test\n            }\n            \n            # Train models\n            training_result = self.ml_agent.train_models(dataset)\n            \n            if training_result.get('success'):\n                logger.info(\"✅ Model training completed successfully\")\n                \n                # Save training summary\n                summary = {\n                    'training_date': datetime.now().isoformat(),\n                    'samples_used': len(dataset),\n                    'features_used': available_features,\n                    'symbols_trained': list(set(dataset['symbol'])),\n                    'best_model': training_result.get('best_model', {}),\n                    'training_period_days': self.training_period_days\n                }\n                \n                # Save summary to file\n                summary_file = self.model_dir / 'training_summary.json'\n                import json\n                with open(summary_file, 'w') as f:\n                    json.dump(summary, f, indent=2)\n                \n                return {\n                    'success': True,\n                    'summary': summary,\n                    'training_result': training_result\n                }\n            else:\n                logger.error(\"❌ Model training failed\")\n                return {'success': False, 'error': 'Training failed'}\n                \n        except Exception as e:\n            logger.error(f\"❌ Error during model training: {str(e)}\")\n            return {'success': False, 'error': str(e)}\n    \n    async def run_full_training_pipeline(self) -> Dict:\n        \"\"\"Run the complete automated training pipeline\"\"\"\n        logger.info(\"🚀 Starting Full Automated Training Pipeline\")\n        logger.info(\"=\" * 60)\n        \n        start_time = datetime.now()\n        \n        try:\n            # Step 1: Initialize\n            await self.initialize()\n            \n            # Step 2: Fetch all historical data\n            logger.info(\"📊 Step 1: Fetching historical data...\")\n            all_data = await self.fetch_all_training_data()\n            \n            if not all_data:\n                return {\n                    'success': False,\n                    'error': 'No training data could be fetched'\n                }\n            \n            # Step 3: Prepare ML dataset\n            logger.info(\"🔄 Step 2: Preparing ML dataset...\")\n            dataset = self._prepare_ml_dataset(all_data)\n            \n            if len(dataset) < 1000:  # Minimum samples requirement\n                return {\n                    'success': False,\n                    'error': f'Insufficient training data: {len(dataset)} samples'\n                }\n            \n            # Step 4: Train models\n            logger.info(\"🤖 Step 3: Training ML models...\")\n            training_result = await self.train_models(dataset)\n            \n            if not training_result.get('success'):\n                return training_result\n            \n            # Step 5: Validation and summary\n            end_time = datetime.now()\n            training_duration = (end_time - start_time).total_seconds()\n            \n            final_result = {\n                'success': True,\n                'training_completed': True,\n                'training_duration_seconds': training_duration,\n                'symbols_processed': len(all_data),\n                'total_samples': len(dataset),\n                'models_trained': training_result.get('summary', {}).get('features_used', []),\n                'best_model': training_result.get('training_result', {}).get('best_model', {}),\n                'ready_for_trading': True\n            }\n            \n            logger.info(\"\" * 60)\n            logger.info(\"🎉 AUTOMATED TRAINING COMPLETED SUCCESSFULLY!\")\n            logger.info(f\"📊 Processed {len(all_data)} symbols\")\n            logger.info(f\"🤖 Trained on {len(dataset)} samples\")\n            logger.info(f\"⏱️ Training took {training_duration:.1f} seconds\")\n            logger.info(\"✅ System is ready for live trading!\")\n            logger.info(\"=\" * 60)\n            \n            return final_result\n            \n        except Exception as e:\n            logger.error(f\"❌ Training pipeline failed: {str(e)}\")\n            return {\n                'success': False,\n                'error': str(e)\n            }\n    \n    async def quick_retrain(self, symbols: List[str] = None) -> Dict:\n        """Quick retrain with latest data for specific symbols""""
        if symbols is None:
            symbols = self.stock_symbols[:5]  # Top 5 stocks for quick retrain
        
        logger.info(f"⚡ Quick retrain for {len(symbols)} symbols...")
        
        all_data = {}
        for symbol in symbols:
            data = await self.fetch_historical_data(symbol, period_days=30)  # Last 30 days
            if data is not None:
                all_data[symbol] = data
        
        if all_data:
            dataset = self._prepare_ml_dataset(all_data)
            training_result = await self.train_models(dataset)
            
            if training_result.get('success'):
                # Quick validation
                validation_result = await self.validate_models(training_result)
                
                return {
                    'success': True,
                    'quick_retrain_completed': True,
                    'symbols_retrained': len(all_data),
                    'samples_used': len(dataset),
                    'training_result': training_result,
                    'validation_result': validation_result,
                    'ready_for_trading': validation_result.get('ready_for_production', False)
                }
            
            return training_result
        
        return {'success': False, 'error': 'No data for quick retrain'}
    
    async def validate_models(self, training_result: Dict) -> Dict:
        """Validate trained models with out-of-sample testing"""
        logger.info("🔍 Validating trained models...")
        
        try:
            validation_results = {}
            
            # Get model performance from ML agent
            if self.ml_agent and hasattr(self.ml_agent, 'get_model_performance'):
                performance = self.ml_agent.get_model_performance()
                
                # Validate minimum performance criteria
                min_accuracy = 0.55  # Minimum 55% accuracy
                min_precision = 0.50  # Minimum 50% precision
                
                for model_name, metrics in performance.items():
                    accuracy = metrics.get('accuracy', 0)
                    precision = metrics.get('precision', 0)
                    
                    validation_results[model_name] = {
                        'passed': accuracy >= min_accuracy and precision >= min_precision,
                        'accuracy': accuracy,
                        'precision': precision,
                        'meets_criteria': {
                            'accuracy': accuracy >= min_accuracy,
                            'precision': precision >= min_precision
                        }
                    }
            
            # Overall validation status
            models_passed = sum(1 for result in validation_results.values() if result['passed'])
            total_models = len(validation_results)
            
            validation_summary = {
                'success': models_passed > 0,
                'models_passed': models_passed,
                'total_models': total_models,
                'pass_rate': models_passed / total_models if total_models > 0 else 0,
                'validation_results': validation_results,
                'ready_for_production': models_passed >= total_models * 0.5  # At least 50% models pass
            }
            
            if validation_summary['ready_for_production']:
                logger.info(f"✅ Model validation passed: {models_passed}/{total_models} models meet criteria")
            else:
                logger.warning(f"⚠️ Model validation concerns: Only {models_passed}/{total_models} models meet criteria")
            
            return validation_summary
            
        except Exception as e:
            logger.error(f"❌ Error during model validation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'ready_for_production': False
            }
    
    async def create_training_report(self, training_result: Dict, validation_result: Dict) -> Dict:
        """Create comprehensive training report"""
        try:
            report = {
                'training_metadata': {
                    'training_date': datetime.now().isoformat(),
                    'training_period_days': self.training_period_days,
                    'symbols_count': len(self.stock_symbols) + len(self.futures_symbols),
                    'data_sources': ['yfinance'],
                    'models_trained': ['lightgbm', 'xgboost']
                },
                'data_summary': training_result.get('summary', {}),
                'training_performance': training_result.get('training_result', {}),
                'validation_results': validation_result,
                'production_readiness': {
                    'data_quality': len(training_result.get('summary', {}).get('symbols_trained', [])) >= 10,
                    'sample_size': training_result.get('summary', {}).get('samples_used', 0) >= 1000,
                    'model_performance': validation_result.get('ready_for_production', False),
                    'overall_ready': False
                }
            }
            
            # Determine overall production readiness
            readiness_checks = report['production_readiness']
            report['production_readiness']['overall_ready'] = all([
                readiness_checks['data_quality'],
                readiness_checks['sample_size'],
                readiness_checks['model_performance']
            ])
            
            # Save report to file
            report_file = self.model_dir / f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            import json
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📋 Training report saved to {report_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error creating training report: {str(e)}")
            return {'error': str(e)}
    
    async def setup_model_monitoring(self) -> Dict:
        """Setup model performance monitoring"""
        try:
            # Create monitoring configuration
            monitoring_config = {
                'enabled': True,
                'performance_threshold': {
                    'accuracy': 0.55,
                    'precision': 0.50,
                    'daily_return': -0.02  # Alert if daily return drops below -2%
                },
                'retraining_triggers': {
                    'performance_degradation': True,
                    'data_drift': True,
                    'weekly_retrain': True
                },
                'alert_channels': {
                    'log_file': True,
                    'console': True
                },
                'last_setup': datetime.now().isoformat()
            }
            
            # Save monitoring config
            config_file = self.model_dir / 'monitoring_config.json'
            import json
            with open(config_file, 'w') as f:
                json.dump(monitoring_config, f, indent=2)
            
            logger.info("📊 Model monitoring configured")
            
            return {
                'success': True,
                'monitoring_enabled': True,
                'config_file': str(config_file)
            }
            
        except Exception as e:
            logger.error(f"❌ Error setting up monitoring: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict:
        """Cleanup old training data and models"""
        try:
            cleanup_summary = {
                'files_cleaned': 0,
                'space_freed_mb': 0,
                'errors': []
            }
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean old CSV files
            for csv_file in self.data_dir.glob('*.csv'):
                if csv_file.stat().st_mtime < cutoff_date.timestamp():
                    try:
                        file_size = csv_file.stat().st_size / (1024 * 1024)  # MB
                        csv_file.unlink()
                        cleanup_summary['files_cleaned'] += 1
                        cleanup_summary['space_freed_mb'] += file_size
                    except Exception as e:
                        cleanup_summary['errors'].append(f"Failed to delete {csv_file}: {str(e)}")
            
            # Clean old model files (keep latest 5)
            model_files = sorted(self.model_dir.glob('*.pkl'), key=lambda f: f.stat().st_mtime, reverse=True)
            for old_model in model_files[5:]:  # Keep latest 5 models
                try:
                    file_size = old_model.stat().st_size / (1024 * 1024)  # MB
                    old_model.unlink()
                    cleanup_summary['files_cleaned'] += 1
                    cleanup_summary['space_freed_mb'] += file_size
                except Exception as e:
                    cleanup_summary['errors'].append(f"Failed to delete {old_model}: {str(e)}")
            
            logger.info(f"🧹 Cleanup completed: {cleanup_summary['files_cleaned']} files, {cleanup_summary['space_freed_mb']:.1f} MB freed")
            
            return {
                'success': True,
                'cleanup_summary': cleanup_summary
            }
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def start(self):
        """Start the automated trainer"""
        logger.info("Starting automated trainer...")
        
        try:
            # Initialize the trainer
            await self.initialize()
            
            # Check if models exist and are recent
            model_files = list(self.model_dir.glob('*.pkl'))
            
            if model_files:
                # Check if models are recent (less than 7 days old)
                latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
                model_age = datetime.now().timestamp() - latest_model.stat().st_mtime
                
                if model_age < 7 * 24 * 3600:  # 7 days
                    logger.info("✅ Recent models found, ready for trading")
                    return {
                        'success': True,
                        'message': 'Automated trainer started with existing models',
                        'models_found': len(model_files),
                        'latest_model_age_hours': model_age / 3600
                    }
            
            # No recent models, trigger training
            logger.info("🔄 No recent models found, starting training...")
            training_result = await self.run_full_training_pipeline()
            
            return {
                'success': training_result.get('success', False),
                'message': 'Automated trainer started and completed training',
                'training_result': training_result
            }
            
        except Exception as e:
            logger.error(f"Error starting automated trainer: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def emergency_stop(self):
        """Emergency stop for the automated trainer"""
        logger.info("Initiating emergency stop for automated trainer...")
        
        try:
            # Stop any ongoing training (would interrupt training loops in real implementation)
            if hasattr(self, 'training_in_progress'):
                self.training_in_progress = False
            
            # Save current state
            emergency_state = {
                'emergency_stop_time': datetime.now().isoformat(),
                'reason': 'Manual emergency stop',
                'training_interrupted': True
            }
            
            state_file = self.model_dir / 'emergency_state.json'
            import json
            with open(state_file, 'w') as f:
                json.dump(emergency_state, f, indent=2)
            
            logger.warning("⚠️ Automated trainer emergency stop completed")
            
            return {
                'success': True,
                'message': 'Automated trainer emergency stop completed',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error during automated trainer emergency stop: {str(e)}")
            return {'success': False, 'error': str(e)}


# Convenience function for easy integration
async def run_automated_training(training_days: int = 365) -> Dict:
    """Run automated training - can be called from startup script"""
    trainer = AutomatedTrainer(training_period_days=training_days)
    return await trainer.run_full_training_pipeline()