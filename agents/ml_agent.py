"""
Automated ML Agent for AI Trading System
Responsible for training, deploying, and using machine learning models for trading signals
"""
import logging
import os
import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import xgboost as xgb

# Try to import TensorFlow, make it optional for Windows compatibility
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM models will be disabled.")

logger = logging.getLogger(__name__)

class MLModel(ABC):
    """Abstract base class for ML models"""
    
    @abstractmethod
    def prepare_data(self, raw_data: pd.DataFrame) -> Dict:
        """Prepare and preprocess data for model training/prediction"""
        pass
    
    @abstractmethod
    def train(self, data: Dict) -> Dict:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, data: Dict) -> Dict:
        """Make predictions using the trained model"""
        pass
    
    @abstractmethod
    def evaluate(self, data: Dict) -> Dict:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the trained model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load a trained model from disk"""
        pass

class LightGBMModel(MLModel):
    """LightGBM model implementation"""
    
    def __init__(self, params: Dict = None):
        self.params = params or {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        self.model = None
        self.trained = False
    
    def prepare_data(self, raw_data: pd.DataFrame) -> Dict:
        """Prepare and preprocess data for LightGBM model"""
        try:
            # Feature engineering for trading signals
            features_df = raw_data.copy()
            
            # Technical indicators as features
            features_df['returns'] = features_df['close'].pct_change()
            features_df['sma_5'] = features_df['close'].rolling(window=5).mean()
            features_df['sma_20'] = features_df['close'].rolling(window=20).mean()
            features_df['sma_50'] = features_df['close'].rolling(window=50).mean()
            
            # RSI
            delta = features_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = features_df['close'].ewm(span=12).mean()
            ema_26 = features_df['close'].ewm(span=26).mean()
            features_df['macd'] = ema_12 - ema_26
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            bb_sma = features_df['close'].rolling(window=20).mean()
            bb_std = features_df['close'].rolling(window=20).std()
            features_df['bb_upper'] = bb_sma + (bb_std * 2)
            features_df['bb_lower'] = bb_sma - (bb_std * 2)
            features_df['bb_position'] = (features_df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Volume indicators
            features_df['volume_sma'] = features_df['volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            
            # Price position within daily range
            features_df['price_position'] = (features_df['close'] - features_df['low']) / (features_df['high'] - features_df['low'])
            
            # Volatility
            features_df['volatility'] = features_df['returns'].rolling(window=20).std()
            
            # Generate target variable (next day return > 0.2%)
            features_df['target'] = (features_df['returns'].shift(-1) > 0.002).astype(int)
            
            # Select features for training
            feature_columns = [
                'returns', 'rsi', 'macd', 'macd_signal', 'bb_position',
                'volume_ratio', 'price_position', 'volatility',
                'sma_5', 'sma_20', 'sma_50'
            ]
            
            # Normalize features
            for col in feature_columns:
                if col in features_df.columns:
                    features_df[col] = (features_df[col] - features_df[col].mean()) / features_df[col].std()
            
            # Remove NaN values
            features_df = features_df.dropna()
            
            # Prepare training data
            X = features_df[feature_columns].values
            y = features_df['target'].values
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X': X,
                'y': y,
                'feature_names': feature_columns,
                'data_shape': X.shape
            }
            
        except Exception as e:
            logger.error(f"Error in LightGBM data preparation: {e}")
            return {}
    
    def train(self, data: Dict) -> Dict:
        """Train the LightGBM model"""
        try:
            # Create LightGBM dataset
            train_data = lgb.Dataset(data['X_train'], label=data['y_train'])
            
            # Train model
            self.model = lgb.train(self.params, train_data)
            self.trained = True
            
            # Evaluate model
            evaluation = self.evaluate(data)
            
            return {
                'success': True,
                'evaluation': evaluation
            }
        except Exception as e:
            logger.error(f"Error training LightGBM model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, data: Dict) -> Dict:
        """Make predictions using the trained LightGBM model"""
        if not self.trained:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            # Make predictions
            predictions = self.model.predict(data['X'])
            
            return {
                'success': True,
                'predictions': predictions
            }
        except Exception as e:
            logger.error(f"Error making LightGBM predictions: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def evaluate(self, data: Dict) -> Dict:
        """Evaluate LightGBM model performance"""
        if not self.trained:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            # Make predictions
            predictions = self.model.predict(data['X_test'])
            
            # Convert probabilities to binary predictions
            binary_predictions = (predictions > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(data['y_test'], binary_predictions)
            precision = precision_score(data['y_test'], binary_predictions)
            recall = recall_score(data['y_test'], binary_predictions)
            f1 = f1_score(data['y_test'], binary_predictions)
            
            return {
                'success': True,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        except Exception as e:
            logger.error(f"Error evaluating LightGBM model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def save(self, path: str):
        """Save the trained LightGBM model to disk"""
        if not self.trained:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            self.model.save_model(path)
            return {'success': True}
        except Exception as e:
            logger.error(f"Error saving LightGBM model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def load(self, path: str):
        """Load a trained LightGBM model from disk"""
        try:
            self.model = lgb.Booster(model_file=path)
            self.trained = True
            return {'success': True}
        except Exception as e:
            logger.error(f"Error loading LightGBM model: {str(e)}")
            return {'success': False, 'error': str(e)}

class XGBoostModel(MLModel):
    """XGBoost model implementation"""
    
    def __init__(self, params: Dict = None):
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.7
        }
        self.model = None
        self.trained = False
    
    def prepare_data(self, raw_data: pd.DataFrame) -> Dict:
        """Prepare and preprocess data for XGBoost model"""
        try:
            # Feature engineering similar to LightGBM but with XGBoost-specific optimizations
            features_df = raw_data.copy()
            
            # Technical indicators
            features_df['returns'] = features_df['close'].pct_change()
            features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                features_df[f'sma_{window}'] = features_df['close'].rolling(window=window).mean()
                features_df[f'price_sma_{window}_ratio'] = features_df['close'] / features_df[f'sma_{window}']
            
            # RSI with multiple periods
            for period in [14, 21]:
                delta = features_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                features_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD variants
            ema_12 = features_df['close'].ewm(span=12).mean()
            ema_26 = features_df['close'].ewm(span=26).mean()
            features_df['macd'] = ema_12 - ema_26
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
            
            # Stochastic Oscillator
            low_14 = features_df['low'].rolling(window=14).min()
            high_14 = features_df['high'].rolling(window=14).max()
            features_df['stoch_k'] = 100 * (features_df['close'] - low_14) / (high_14 - low_14)
            features_df['stoch_d'] = features_df['stoch_k'].rolling(window=3).mean()
            
            # Williams %R
            features_df['williams_r'] = -100 * (high_14 - features_df['close']) / (high_14 - low_14)
            
            # Average True Range (ATR)
            tr1 = features_df['high'] - features_df['low']
            tr2 = abs(features_df['high'] - features_df['close'].shift(1))
            tr3 = abs(features_df['low'] - features_df['close'].shift(1))
            features_df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            features_df['atr'] = features_df['tr'].rolling(window=14).mean()
            
            # Volume indicators
            features_df['volume_sma'] = features_df['volume'].rolling(window=20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            features_df['vwap'] = (features_df['close'] * features_df['volume']).rolling(window=20).sum() / features_df['volume'].rolling(window=20).sum()
            
            # Price action features
            features_df['high_low_pct'] = (features_df['high'] - features_df['low']) / features_df['close']
            features_df['open_close_pct'] = (features_df['close'] - features_df['open']) / features_df['open']
            
            # Generate target (multi-class: strong_buy, buy, hold, sell, strong_sell)
            future_returns = features_df['returns'].shift(-1)
            conditions = [
                future_returns > 0.005,  # > 0.5% - strong buy
                (future_returns > 0.002) & (future_returns <= 0.005),  # 0.2-0.5% - buy
                (future_returns >= -0.002) & (future_returns <= 0.002),  # -0.2% to 0.2% - hold
                (future_returns >= -0.005) & (future_returns < -0.002),  # -0.5% to -0.2% - sell
                future_returns < -0.005  # < -0.5% - strong sell
            ]
            choices = [4, 3, 2, 1, 0]  # Strong buy to strong sell
            features_df['target'] = np.select(conditions, choices, default=2)
            
            # Select features
            feature_columns = [
                'returns', 'log_returns', 'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_histogram',
                'stoch_k', 'stoch_d', 'williams_r', 'atr', 'volume_ratio', 'vwap',
                'high_low_pct', 'open_close_pct', 'price_sma_5_ratio', 'price_sma_20_ratio'
            ]
            
            # Feature scaling
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            
            # Remove NaN values
            features_df = features_df.dropna()
            
            X = features_df[feature_columns].values
            y = features_df['target'].values
            
            # Scale features
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X': X_scaled,
                'y': y,
                'feature_names': feature_columns,
                'scaler': scaler,
                'data_shape': X_scaled.shape
            }
            
        except Exception as e:
            logger.error(f"Error in XGBoost data preparation: {e}")
            return {}
    
    def train(self, data: Dict) -> Dict:
        """Train the XGBoost model"""
        try:
            # Create DMatrix
            dtrain = xgb.DMatrix(data['X_train'], label=data['y_train'])
            
            # Train model
            self.model = xgb.train(self.params, dtrain)
            self.trained = True
            
            # Evaluate model
            evaluation = self.evaluate(data)
            
            return {
                'success': True,
                'evaluation': evaluation
            }
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, data: Dict) -> Dict:
        """Make predictions using the trained XGBoost model"""
        if not self.trained:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            # Create DMatrix
            dtest = xgb.DMatrix(data['X'])
            
            # Make predictions
            predictions = self.model.predict(dtest)
            
            return {
                'success': True,
                'predictions': predictions
            }
        except Exception as e:
            logger.error(f"Error making XGBoost predictions: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def evaluate(self, data: Dict) -> Dict:
        """Evaluate XGBoost model performance"""
        if not self.trained:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            # Create DMatrix
            dtest = xgb.DMatrix(data['X_test'], label=data['y_test'])
            
            # Make predictions
            predictions = self.model.predict(dtest)
            
            # Convert probabilities to binary predictions
            binary_predictions = (predictions > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(data['y_test'], binary_predictions)
            precision = precision_score(data['y_test'], binary_predictions)
            recall = recall_score(data['y_test'], binary_predictions)
            f1 = f1_score(data['y_test'], binary_predictions)
            
            return {
                'success': True,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        except Exception as e:
            logger.error(f"Error evaluating XGBoost model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def save(self, path: str):
        """Save the trained XGBoost model to disk"""
        if not self.trained:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            self.model.save_model(path)
            return {'success': True}
        except Exception as e:
            logger.error(f"Error saving XGBoost model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def load(self, path: str):
        """Load a trained XGBoost model from disk"""
        try:
            self.model = xgb.Booster(model_file=path)
            self.model.load_model(path)
            self.trained = True
            return {'success': True}
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {str(e)}")
            return {'success': False, 'error': str(e)}

class LSTMModel(MLModel):
    """LSTM model implementation for time series prediction"""
    
    def __init__(self, params: Dict = None):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. LSTM models require TensorFlow.")
            
        self.params = params or {
            'sequence_length': 60,
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001,
            'units': 64,
            'dropout': 0.2
        }
        self.model = None
        self.trained = False
        self.scaler = None
    
    def prepare_data(self, raw_data: pd.DataFrame) -> Dict:
        """Prepare and preprocess data for LSTM model"""
        try:
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow not available for LSTM model")
            
            # Feature engineering for time series
            features_df = raw_data.copy()
            
            # Technical indicators
            features_df['returns'] = features_df['close'].pct_change()
            features_df['sma_10'] = features_df['close'].rolling(window=10).mean()
            features_df['sma_30'] = features_df['close'].rolling(window=30).mean()
            features_df['rsi'] = self._calculate_rsi(features_df['close'], 14)
            features_df['macd'] = self._calculate_macd(features_df['close'])
            
            # Normalize features
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            
            # Select features for LSTM
            feature_columns = ['close', 'volume', 'returns', 'sma_10', 'sma_30', 'rsi', 'macd']
            features_df = features_df[feature_columns].dropna()
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(features_df.values)
            
            # Create sequences for LSTM
            sequence_length = self.params['sequence_length']
            X, y = self._create_sequences(scaled_data, sequence_length)
            
            # Generate target (price direction)
            # 1 if next price > current price, 0 otherwise
            price_data = features_df['close'].values
            targets = []
            for i in range(len(price_data) - 1):
                if i >= sequence_length:
                    targets.append(1 if price_data[i + 1] > price_data[i] else 0)
            
            y = np.array(targets)
            
            # Ensure X and y have the same length
            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y[:min_length]
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X': X,
                'y': y,
                'feature_names': feature_columns,
                'sequence_length': sequence_length,
                'scaler': self.scaler
            }
            
        except Exception as e:
            logger.error(f"Error in LSTM data preparation: {e}")
            return {}
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def _create_sequences(self, data, sequence_length):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            # Use next time step's first feature (close price) as target
            y.append(data[i + sequence_length, 0])
        return np.array(X), np.array(y)
    
    def train(self, data: Dict) -> Dict:
        """Train the LSTM model"""
        try:
            # Build model if not already built
            if not self.model:
                self.build_model(data['X_train'].shape)
            
            # Train model
            history = self.model.fit(
                data['X_train'],
                data['y_train'],
                batch_size=self.params['batch_size'],
                epochs=self.params['epochs'],
                validation_split=0.1
            )
            self.trained = True
            
            # Evaluate model
            evaluation = self.evaluate(data)
            
            return {
                'success': True,
                'evaluation': evaluation
            }
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def build_model(self, input_shape):
        """Build the LSTM model architecture"""
        self.model = Sequential()
        self.model.add(LSTM(
            units=self.params['units'],
            return_sequences=True,
            input_shape=input_shape
        ))
        self.model.add(Dropout(self.params['dropout']))
        self.model.add(LSTM(units=self.params['units']))
        self.model.add(Dropout(self.params['dropout']))
        self.model.add(Dense(1, activation='sigmoid'))
        
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
    
    def predict(self, data: Dict) -> Dict:
        """Make predictions using the trained LSTM model"""
        if not self.trained:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            # Make predictions
            predictions = self.model.predict(data['X'])
            
            return {
                'success': True,
                'predictions': predictions
            }
        except Exception as e:
            logger.error(f"Error making LSTM predictions: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def evaluate(self, data: Dict) -> Dict:
        """Evaluate LSTM model performance"""
        if not self.trained:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            # Evaluate model
            loss, accuracy = self.model.evaluate(data['X_test'], data['y_test'], verbose=0)
            
            # Make predictions
            predictions = self.model.predict(data['X_test'])
            binary_predictions = (predictions > 0.5).astype(int)
            
            # Calculate metrics
            precision = precision_score(data['y_test'], binary_predictions)
            recall = recall_score(data['y_test'], binary_predictions)
            f1 = f1_score(data['y_test'], binary_predictions)
            
            return {
                'success': True,
                'loss': loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        except Exception as e:
            logger.error(f"Error evaluating LSTM model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def save(self, path: str):
        """Save the trained LSTM model to disk"""
        if not self.trained:
            return {'success': False, 'error': 'Model not trained'}
        
        try:
            # Save model architecture and weights
            self.model.save(f"{path}_model")
            
            # Save scaler if exists
            if self.scaler:
                with open(f"{path}_scaler.pkl", 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            return {'success': True}
        except Exception as e:
            logger.error(f"Error saving LSTM model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def load(self, path: str):
        """Load a trained LSTM model from disk"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(f"{path}_model")
            
            # Load scaler if exists
            try:
                with open(f"{path}_scaler.pkl", 'rb') as f:
                    self.scaler = pickle.load(f)
            except:
                self.scaler = None
            
            self.trained = True
            return {'success': True}
        except Exception as e:
            logger.error(f"Error loading LSTM model: {str(e)}")
            return {'success': False, 'error': str(e)}

class ModelSelector:
    """
    Model Selector
    Implements champion-challenger model selection based on performance criteria
    """
    
    def __init__(self, performance_thresholds: Dict):
        self.performance_thresholds = performance_thresholds
        self.model_history = {}  # Stores model performance history
        
    def select_best_model(self, model_results: List[Dict]) -> Dict:
        """
        Select the best model based on performance criteria
        Implements champion-challenger approach
        """
        # Filter models that meet minimum performance thresholds
        qualified_models = [
            result for result in model_results
            if self._meets_thresholds(result['evaluation'])
        ]
        
        if not qualified_models:
            return {
                'success': False,
                'error': 'No models meet minimum performance thresholds'
            }
        
        # Sort by performance (e.g., Sharpe ratio, F1 score, etc.)
        sorted_models = sorted(
            qualified_models,
            key=lambda x: x['evaluation']['f1_score'],
            reverse=True
        )
        
        # Get current champion if exists
        current_champion = self._get_current_champion()
        
        # If no current champion, select the best qualified model
        if not current_champion:
            return {
                'success': True,
                'selected_model': sorted_models[0],
                'reason': 'No current champion, selecting best qualified model'
            }
        
        # Compare current champion with best challenger
        best_challenger = sorted_models[0]
        
        # If best challenger significantly outperforms current champion, promote it
        if self._is_significantly_better(current_champion, best_challenger):
            return {
                'success': True,
                'selected_model': best_challenger,
                'reason': 'Best challenger significantly outperforms current champion'
            }
        
        # Otherwise, keep current champion
        return {
            'success': True,
            'selected_model': current_champion,
            'reason': 'Current champion performs better than challengers'
        }
    
    def _meets_thresholds(self, evaluation: Dict) -> bool:
        """Check if model meets minimum performance thresholds"""
        # Implementation for threshold checking
        return True  # Simplified for now
    
    def _get_current_champion(self):
        """Get current champion model from history"""
        # Implementation to retrieve current champion
        return None  # Simplified for now
    
    def _is_significantly_better(self, champion, challenger) -> bool:
        """Determine if challenger is significantly better than champion"""
        # Implementation for significance testing
        return True  # Simplified for now

class MlAgent:
    """
    Automated ML Agent
    Responsible for training, deploying, and managing machine learning models
    """
    
    def __init__(self, model_dir: str, training_interval: int = 86400, models_to_train: List[str] = ['lightgbm', 'xgboost']):
        self.model_dir = model_dir
        self.training_interval = training_interval  # in seconds
        self.models_to_train = models_to_train
        
        # Available models (conditionally include LSTM)
        self.available_models = {
            'lightgbm': LightGBMModel,
            'xgboost': XGBoostModel
        }
        
        # Add LSTM only if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self.available_models['lstm'] = LSTMModel
        else:
            logger.warning("LSTM models not available - TensorFlow not installed")
            # Remove LSTM from models to train if it was requested
            self.models_to_train = [m for m in self.models_to_train if m.lower() != 'lstm']
        
        self.active_models = {}
        self.model_selector = ModelSelector({
            'min_accuracy': 0.55,
            'min_f1_score': 0.55,
            'max_drawdown': 0.05
        })
        self.current_model = None
        self.model_update_timestamp = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the requested ML models"""
        for model_name in self.models_to_train:
            if model_name.lower() in self.available_models:
                self.active_models[model_name.lower()] = self.available_models[model_name.lower()]()
            else:
                logger.warning(f"Unsupported ML model: {model_name}")
    
    def train_models(self, data: pd.DataFrame) -> Dict:
        """Train all active ML models with the provided data"""
        if data.empty:
            return {'success': False, 'error': 'No data to train on'}
        
        model_results = []
        training_summary = {
            'total_models': len(self.active_models),
            'successful_models': 0,
            'failed_models': 0,
            'training_start_time': datetime.now().isoformat(),
            'model_performances': {}
        }
        
        logger.info(f"Starting training for {len(self.active_models)} models...")
        
        # Train each model
        for model_name, model in self.active_models.items():
            try:
                logger.info(f"Training {model_name} model...")
                
                # Prepare data for this model
                prepared_data = model.prepare_data(data)
                
                if not prepared_data:
                    logger.error(f"Data preparation failed for {model_name}")
                    training_summary['failed_models'] += 1
                    continue
                
                # Train model
                train_result = model.train(prepared_data)
                
                if train_result['success']:
                    # Save trained model
                    model_save_path = os.path.join(
                        self.model_dir, 
                        f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    
                    # Create model directory if it doesn't exist
                    os.makedirs(self.model_dir, exist_ok=True)
                    
                    save_result = model.save(model_save_path)
                    
                    if save_result.get('success', False):
                        # Store model performance
                        performance_metrics = train_result.get('evaluation', {})
                        
                        model_info = {
                            'model_name': model_name,
                            'model_path': model_save_path,
                            'performance': performance_metrics,
                            'training_time': datetime.now().isoformat(),
                            'data_shape': prepared_data.get('data_shape', 'unknown'),
                            'feature_count': len(prepared_data.get('feature_names', []))
                        }
                        
                        model_results.append({
                            **train_result,
                            **model_info
                        })
                        
                        training_summary['model_performances'][model_name] = performance_metrics
                        training_summary['successful_models'] += 1
                        
                        logger.info(f"Successfully trained and saved {model_name} model")
                        logger.info(f"Performance: {performance_metrics}")
                    else:
                        logger.error(f"Failed to save {model_name} model")
                        training_summary['failed_models'] += 1
                else:
                    logger.error(f"Training failed for {model_name}: {train_result.get('error', 'Unknown error')}")
                    training_summary['failed_models'] += 1
                    
            except Exception as e:
                logger.error(f"Error training {model_name} model: {str(e)}")
                training_summary['failed_models'] += 1
                continue
        
        training_summary['training_end_time'] = datetime.now().isoformat()
        
        # Select best model if any were successful
        if model_results:
            selection_result = self.model_selector.select_best_model(model_results)
            
            if selection_result['success']:
                self.current_model = selection_result['selected_model']
                self.model_update_timestamp = datetime.now()
                
                logger.info(f"Selected best model: {self.current_model['model_name']}")
                logger.info(f"Selection reason: {selection_result['reason']}")
                
                return {
                    'success': True,
                    'best_model': selection_result['selected_model'],
                    'selection_reason': selection_result['reason'],
                    'training_summary': training_summary,
                    'all_models': model_results
                }
            else:
                logger.error(f"Model selection failed: {selection_result.get('error', 'Unknown error')}")
        
        return {
            'success': False, 
            'error': 'No models were successfully trained',
            'training_summary': training_summary
        }
    
    def generate_prediction(self, data: pd.DataFrame) -> Dict:
        """Generate prediction using the current best model"""
        if not self.current_model:
            return {'success': False, 'error': 'No model currently selected'}
        
        try:
            model_name = self.current_model['model_name']
            
            if model_name not in self.active_models:
                return {'success': False, 'error': f'Active model {model_name} not found'}
            
            model = self.active_models[model_name]
            
            # Prepare data for prediction
            prepared_data = model.prepare_data(data)
            
            if not prepared_data:
                return {'success': False, 'error': 'Data preparation failed for prediction'}
            
            # Make prediction
            prediction_result = model.predict(prepared_data)
            
            if prediction_result['success']:
                predictions = prediction_result['predictions']
                
                # Calculate confidence based on prediction probability
                if isinstance(predictions, np.ndarray) and len(predictions) > 0:
                    # For binary classification
                    if model_name == 'lightgbm':
                        confidence = float(predictions[-1])  # Last prediction
                        signal = 1 if confidence > 0.5 else 0
                        confidence = confidence if signal == 1 else (1 - confidence)
                    
                    # For multi-class classification (XGBoost)
                    elif model_name == 'xgboost':
                        confidence = float(np.max(predictions[-1]) if predictions.ndim > 1 else predictions[-1])
                        signal = int(np.argmax(predictions[-1]) if predictions.ndim > 1 else predictions[-1])
                    
                    # For LSTM
                    elif model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                        confidence = float(predictions[-1])
                        signal = 1 if confidence > 0.5 else 0
                    else:
                        confidence = 0.5
                        signal = 0
                else:
                    confidence = 0.5
                    signal = 0
                
                # Add metadata about the prediction
                prediction_metadata = {
                    'model_used': model_name,
                    'model_performance': self.current_model.get('performance', {}),
                    'prediction_time': datetime.now().isoformat(),
                    'data_points_used': len(data),
                    'feature_count': len(prepared_data.get('feature_names', []))
                }
                
                return {
                    'success': True,
                    'signal': signal,
                    'confidence': confidence,
                    'raw_prediction': predictions[-1] if len(predictions) > 0 else 0,
                    'metadata': prediction_metadata
                }
            else:
                return {
                    'success': False,
                    'error': f"Prediction failed: {prediction_result.get('error', 'Unknown error')}"
                }
                
        except Exception as e:
            logger.error(f"Error generating ML prediction: {str(e)}")
            return {
                'success': False,
                'error': f"Prediction error: {str(e)}"
            }
    
    def update_model(self, new_data: pd.DataFrame) -> Dict:
        """Update the ML models with new data and potentially switch to a better model"""
        # Check if it's time to update
        if self.model_update_timestamp and (datetime.now() - self.model_update_timestamp).total_seconds() < self.training_interval:
            return {'success': False, 'reason': 'Too soon to update model'}
        
        # Train models with new data
        training_result = self.train_models(new_data)
        
        if training_result['success']:
            # Update current model if a better one was selected
            if training_result['best_model']['model_name'] != self.current_model['model_name']:
                logger.info(f"Model updated: {self.current_model['model_name']} -> {training_result['best_model']['model_name']}")
            
            return {
                'success': True,
                'current_model': self.current_model['model_name'],
                'model_update_time': self.model_update_timestamp.isoformat()
            }
        
        return training_result
    
    def load_model(self, model_name: str, path: str) -> Dict:
        """Load a pre-trained model from disk"""
        if model_name not in self.active_models:
            return {'success': False, 'error': f"Model {model_name} not in active models"}
        
        try:
            result = self.active_models[model_name].load(path)
            if result['success']:
                # Set as current model
                self.current_model = {
                    'model_name': model_name,
                    'model_path': path,
                    'loaded_time': datetime.now().isoformat()
                }
            
            return result
        except Exception as e:
            logger.error(f"Error loading ML model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def start(self):
        """Start the ML agent"""
        logger.info("Starting ML agent...")
        # Implementation for starting the agent
        # This would typically involve:
        # - Loading existing models
        # - Starting model update loop
        # - Connecting to data source
        
    async def emergency_stop(self):
        """Emergency stop for the ML agent"""
        logger.info("Initiating emergency stop for ML agent...")
        # Implementation for emergency stop procedures
        pass