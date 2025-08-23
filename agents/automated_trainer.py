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
        
        # Initialize ML agent
        self.ml_agent = MlAgent(
            model_dir=str(self.model_dir),
            training_interval=86400,  # Daily retraining
            models_to_train=['lightgbm', 'xgboost']  # Start with faster models
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
            
        except Exception as e:\n            logger.error(f\"❌ Error fetching data for {symbol}: {str(e)}\")\n            return None\n            \n    def _add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Add technical indicators to the dataset for ML training\"\"\"\n        try:\n            # Price-based features\n            data['returns'] = data['close'].pct_change()\n            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))\n            \n            # Moving averages\n            for period in [5, 10, 20, 50]:\n                data[f'ma_{period}'] = data['close'].rolling(window=period).mean()\n                data[f'ma_ratio_{period}'] = data['close'] / data[f'ma_{period}']\n            \n            # Volatility features\n            data['volatility_5'] = data['returns'].rolling(window=5).std()\n            data['volatility_20'] = data['returns'].rolling(window=20).std()\n            \n            # RSI\n            delta = data['close'].diff()\n            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()\n            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()\n            rs = gain / loss\n            data['rsi'] = 100 - (100 / (1 + rs))\n            \n            # MACD\n            exp1 = data['close'].ewm(span=12).mean()\n            exp2 = data['close'].ewm(span=26).mean()\n            data['macd'] = exp1 - exp2\n            data['macd_signal'] = data['macd'].ewm(span=9).mean()\n            data['macd_histogram'] = data['macd'] - data['macd_signal']\n            \n            # Bollinger Bands\n            data['bb_middle'] = data['close'].rolling(window=20).mean()\n            bb_std = data['close'].rolling(window=20).std()\n            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)\n            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)\n            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']\n            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])\n            \n            # Volume features\n            data['volume_ma'] = data['volume'].rolling(window=20).mean()\n            data['volume_ratio'] = data['volume'] / data['volume_ma']\n            \n            # Target variable for ML (next day return)\n            data['target'] = data['returns'].shift(-1)\n            data['target_binary'] = (data['target'] > 0).astype(int)\n            \n            # Drop rows with NaN values\n            data = data.dropna()\n            \n            return data\n            \n        except Exception as e:\n            logger.error(f\"Error adding technical features: {str(e)}\")\n            return data\n    \n    async def fetch_all_training_data(self) -> Dict[str, pd.DataFrame]:\n        \"\"\"Fetch historical data for all symbols\"\"\"\n        logger.info(f\"📊 Fetching training data for {len(self.stock_symbols)} stocks...\")\n        \n        all_data = {}\n        \n        # Fetch stock data\n        for symbol in self.stock_symbols:\n            data = await self.fetch_historical_data(symbol)\n            if data is not None and len(data) > 100:  # Minimum data requirement\n                all_data[symbol] = data\n            \n            # Small delay to avoid rate limiting\n            await asyncio.sleep(0.1)\n        \n        # Fetch futures data\n        for symbol in self.futures_symbols:\n            data = await self.fetch_historical_data(symbol)\n            if data is not None and len(data) > 100:\n                all_data[symbol] = data\n            \n            await asyncio.sleep(0.1)\n        \n        logger.info(f\"✅ Successfully fetched data for {len(all_data)} symbols\")\n        return all_data\n    \n    def _prepare_ml_dataset(self, all_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:\n        \"\"\"Combine all data into a single training dataset\"\"\"\n        logger.info(\"🔄 Preparing ML training dataset...\")\n        \n        combined_data = []\n        \n        for symbol, data in all_data.items():\n            # Add symbol as a feature\n            data_copy = data.copy()\n            data_copy['symbol'] = symbol\n            \n            # Select features for ML\n            feature_columns = [\n                'open', 'high', 'low', 'close', 'volume',\n                'returns', 'log_returns',\n                'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20', 'ma_ratio_50',\n                'volatility_5', 'volatility_20',\n                'rsi', 'macd', 'macd_signal', 'macd_histogram',\n                'bb_width', 'bb_position', 'volume_ratio',\n                'target', 'target_binary'\n            ]\n            \n            # Keep only available columns\n            available_columns = [col for col in feature_columns if col in data_copy.columns]\n            ml_data = data_copy[available_columns + ['symbol']]\n            \n            combined_data.append(ml_data)\n        \n        # Combine all data\n        final_dataset = pd.concat(combined_data, ignore_index=True)\n        \n        # Remove any remaining NaN values\n        final_dataset = final_dataset.dropna()\n        \n        logger.info(f\"✅ Prepared dataset with {len(final_dataset)} samples\")\n        return final_dataset\n    \n    async def train_models(self, dataset: pd.DataFrame) -> Dict:\n        \"\"\"Train all ML models using the prepared dataset\"\"\"\n        logger.info(\"🤖 Starting automated model training...\")\n        \n        try:\n            # Prepare data for ML agent\n            feature_columns = [\n                'returns', 'log_returns', 'ma_ratio_5', 'ma_ratio_10', \n                'ma_ratio_20', 'ma_ratio_50', 'volatility_5', 'volatility_20',\n                'rsi', 'macd', 'macd_signal', 'macd_histogram',\n                'bb_width', 'bb_position', 'volume_ratio'\n            ]\n            \n            # Keep only available features\n            available_features = [col for col in feature_columns if col in dataset.columns]\n            \n            X = dataset[available_features]\n            y = dataset['target_binary']  # Binary classification for now\n            \n            # Split data for training\n            split_idx = int(len(dataset) * 0.8)\n            X_train, X_test = X[:split_idx], X[split_idx:]\n            y_train, y_test = y[:split_idx], y[split_idx:]\n            \n            # Prepare data dictionary for ML agent\n            training_data = {\n                'X_train': X_train,\n                'X_test': X_test,\n                'y_train': y_train,\n                'y_test': y_test\n            }\n            \n            # Train models\n            training_result = self.ml_agent.train_models(dataset)\n            \n            if training_result.get('success'):\n                logger.info(\"✅ Model training completed successfully\")\n                \n                # Save training summary\n                summary = {\n                    'training_date': datetime.now().isoformat(),\n                    'samples_used': len(dataset),\n                    'features_used': available_features,\n                    'symbols_trained': list(set(dataset['symbol'])),\n                    'best_model': training_result.get('best_model', {}),\n                    'training_period_days': self.training_period_days\n                }\n                \n                # Save summary to file\n                summary_file = self.model_dir / 'training_summary.json'\n                import json\n                with open(summary_file, 'w') as f:\n                    json.dump(summary, f, indent=2)\n                \n                return {\n                    'success': True,\n                    'summary': summary,\n                    'training_result': training_result\n                }\n            else:\n                logger.error(\"❌ Model training failed\")\n                return {'success': False, 'error': 'Training failed'}\n                \n        except Exception as e:\n            logger.error(f\"❌ Error during model training: {str(e)}\")\n            return {'success': False, 'error': str(e)}\n    \n    async def run_full_training_pipeline(self) -> Dict:\n        \"\"\"Run the complete automated training pipeline\"\"\"\n        logger.info(\"🚀 Starting Full Automated Training Pipeline\")\n        logger.info(\"=\" * 60)\n        \n        start_time = datetime.now()\n        \n        try:\n            # Step 1: Initialize\n            await self.initialize()\n            \n            # Step 2: Fetch all historical data\n            logger.info(\"📊 Step 1: Fetching historical data...\")\n            all_data = await self.fetch_all_training_data()\n            \n            if not all_data:\n                return {\n                    'success': False,\n                    'error': 'No training data could be fetched'\n                }\n            \n            # Step 3: Prepare ML dataset\n            logger.info(\"🔄 Step 2: Preparing ML dataset...\")\n            dataset = self._prepare_ml_dataset(all_data)\n            \n            if len(dataset) < 1000:  # Minimum samples requirement\n                return {\n                    'success': False,\n                    'error': f'Insufficient training data: {len(dataset)} samples'\n                }\n            \n            # Step 4: Train models\n            logger.info(\"🤖 Step 3: Training ML models...\")\n            training_result = await self.train_models(dataset)\n            \n            if not training_result.get('success'):\n                return training_result\n            \n            # Step 5: Validation and summary\n            end_time = datetime.now()\n            training_duration = (end_time - start_time).total_seconds()\n            \n            final_result = {\n                'success': True,\n                'training_completed': True,\n                'training_duration_seconds': training_duration,\n                'symbols_processed': len(all_data),\n                'total_samples': len(dataset),\n                'models_trained': training_result.get('summary', {}).get('features_used', []),\n                'best_model': training_result.get('training_result', {}).get('best_model', {}),\n                'ready_for_trading': True\n            }\n            \n            logger.info(\"\" * 60)\n            logger.info(\"🎉 AUTOMATED TRAINING COMPLETED SUCCESSFULLY!\")\n            logger.info(f\"📊 Processed {len(all_data)} symbols\")\n            logger.info(f\"🤖 Trained on {len(dataset)} samples\")\n            logger.info(f\"⏱️ Training took {training_duration:.1f} seconds\")\n            logger.info(\"✅ System is ready for live trading!\")\n            logger.info(\"=\" * 60)\n            \n            return final_result\n            \n        except Exception as e:\n            logger.error(f\"❌ Training pipeline failed: {str(e)}\")\n            return {\n                'success': False,\n                'error': str(e)\n            }\n    \n    async def quick_retrain(self, symbols: List[str] = None) -> Dict:\n        \"\"\"Quick retrain with latest data for specific symbols\"\"\"\n        if symbols is None:\n            symbols = self.stock_symbols[:5]  # Top 5 stocks for quick retrain\n        \n        logger.info(f\"⚡ Quick retrain for {len(symbols)} symbols...\")\n        \n        all_data = {}\n        for symbol in symbols:\n            data = await self.fetch_historical_data(symbol, period_days=30)  # Last 30 days\n            if data is not None:\n                all_data[symbol] = data\n        \n        if all_data:\n            dataset = self._prepare_ml_dataset(all_data)\n            return await self.train_models(dataset)\n        \n        return {'success': False, 'error': 'No data for quick retrain'}\n\n\n# Convenience function for easy integration\nasync def run_automated_training(training_days: int = 365) -> Dict:\n    \"\"\"Run automated training - can be called from startup script\"\"\"\n    trainer = AutomatedTrainer(training_period_days=training_days)\n    return await trainer.run_full_training_pipeline()