"""
Unit tests for the ML Agent
"""
import pytest
import asyncio
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from agents.ml_agent import MlAgent, LightGBMModel, XGBoostModel, LSTMModel

# Sample data for testing
@pytest.fixture
def sample_data():
    """Create sample data for testing ML models"""
    # Create synthetic data for binary classification
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y
    
    return data

# Sample models for testing
@pytest.fixture
def lightgbm_model():
    """Create a LightGBM model instance for testing"""
    return LightGBMModel()

@pytest.fixture
def xgboost_model():
    """Create an XGBoost model instance for testing"""
    return XGBoostModel()

@pytest.fixture
def lstm_model():
    """Create an LSTM model instance for testing"""
    return LSTMModel()

# ML agent for testing
@pytest.fixture
def ml_agent(tmpdir):
    """Create an ML agent instance for testing"""
    return MlAgent(model_dir=str(tmpdir))

class TestMLModels:
    """Test suite for individual ML models"""
    
    def test_lightgbm_model(self, lightgbm_model, sample_data):
        """Test LightGBM model training and prediction"""
        # Prepare data
        data = {
            'X_train': sample_data.iloc[:800, :-1],
            'y_train': sample_data.iloc[:800, -1],
            'X_test': sample_data.iloc[800:, :-1],
            'y_test': sample_data.iloc[800:, -1]
        }
        
        # Test model training
        train_result = lightgbm_model.train(data)
        assert train_result['success']
        assert 'model' in lightgbm_model.__dict__
        
        # Test model prediction
        prediction = lightgbm_model.predict(data)
        assert prediction['success']
        assert 'predictions' in prediction
        
        # Test model evaluation
        evaluation = lightgbm_model.evaluate(data)
        assert evaluation['success']
        assert 'accuracy' in evaluation
        assert 'precision' in evaluation
        assert 'recall' in evaluation
        assert 'f1_score' in evaluation
        
    def test_xgboost_model(self, xgboost_model, sample_data):
        """Test XGBoost model training and prediction"""
        # Prepare data
        data = {
            'X_train': sample_data.iloc[:800, :-1],
            'y_train': sample_data.iloc[:800, -1],
            'X_test': sample_data.iloc[800:, :-1],
            'y_test': sample_data.iloc[800:, -1]
        }
        
        # Test model training
        train_result = xgboost_model.train(data)
        assert train_result['success']
        assert 'model' in xgboost_model.__dict__
        
        # Test model prediction
        prediction = xgboost_model.predict(data)
        assert prediction['success']
        assert 'predictions' in prediction
        
        # Test model evaluation
        evaluation = xgboost_model.evaluate(data)
        assert evaluation['success']
        assert 'accuracy' in evaluation
        assert 'precision' in evaluation
        assert 'recall' in evaluation
        assert 'f1_score' in evaluation
        
    def test_lstm_model(self, lstm_model, sample_data):
        """Test LSTM model training and prediction"""
        # Prepare data
        data = {
            'X_train': sample_data.iloc[:800, :-1].values.reshape(-1, 1, 10),
            'y_train': sample_data.iloc[:800, -1].values,
            'X_test': sample_data.iloc[800:, :-1].values.reshape(-1, 1, 10),
            'y_test': sample_data.iloc[800:, -1].values
        }
        
        # Test model building
        lstm_model.build_model(data['X_train'].shape)
        assert 'model' in lstm_model.__dict__
        
        # Test model training
        train_result = lstm_model.train(data)
        assert train_result['success']
        
        # Test model prediction
        prediction = lstm_model.predict(data)
        assert prediction['success']
        assert 'predictions' in prediction
        
        # Test model evaluation
        evaluation = lstm_model.evaluate(data)
        assert evaluation['success']
        assert 'loss' in evaluation
        assert 'accuracy' in evaluation
        assert 'precision' in evaluation
        assert 'recall' in evaluation
        assert 'f1_score' in evaluation
        
    def test_model_saving_loading(self, lightgbm_model, xgboost_model, lstm_model, tmpdir, sample_data):
        """Test saving and loading of ML models"""
        # Prepare data
        data = {
            'X_train': sample_data.iloc[:800, :-1],
            'y_train': sample_data.iloc[:800, -1],
            'X_test': sample_data.iloc[800:, :-1],
            'y_test': sample_data.iloc[800:, -1]
        }
        
        # Test LightGBM model saving and loading
        lightgbm_model.train(data)
        save_path = str(tmpdir.join("test_lgb.model"))
        lightgbm_model.save(save_path)
        result = lightgbm_model.load(save_path)
        assert result['success']
        
        # Test XGBoost model saving and loading
        xgboost_model.train(data)
        save_path = str(tmpdir.join("test_xgb.model"))
        xgboost_model.save(save_path)
        result = xgboost_model.load(save_path)
        assert result['success']
        
        # Test LSTM model saving and loading
        # For LSTM, we need to reshape data for sequence input
        lstm_data = {
            'X_train': data['X_train'].values.reshape(-1, 1, 10),
            'y_train': data['y_train'],
            'X_test': data['X_test'].values.reshape(-1, 1, 10),
            'y_test': data['y_test']
        }
        
        lstm_model.build_model(lstm_data['X_train'].shape)
        lstm_model.train(lstm_data)
        save_path = str(tmpdir.join("test_lstm"))
        lstm_model.save(save_path)
        result = lstm_model.load(save_path)
        assert result['success']

class TestMLAgent:
    """Test suite for the ML Agent class"""
    
    def test_initialization(self, ml_agent):
        """Test basic initialization of ML Agent"""
        assert ml_agent.model_dir is not None
        assert ml_agent.training_interval == 86400
        assert len(ml_agent.active_models) > 0
        assert ml_agent.model_selector is not None
        
    def test_train_models(self, ml_agent, sample_data):
        """Test training of ML models"""
        # Prepare data for training
        # For LightGBM and XGBoost
        data = {
            'X_train': sample_data.iloc[:800, :-1],
            'y_train': sample_data.iloc[:800, -1],
            'X_test': sample_data.iloc[800:, :-1],
            'y_test': sample_data.iloc[800:, -1]
        }
        
        # Test model training
        result = ml_agent.train_models(sample_data)
        assert result['success']
        assert 'best_model' in result
        
        # Check that models were saved
        best_model_name = result['best_model']['model_name']
        best_model_path = result['best_model']['model_path']
        assert best_model_name in ml_agent.active_models
        assert os.path.exists(f"{best_model_path}_model") or os.path.exists(best_model_path + "_model")
        
    def test_generate_prediction(self, ml_agent, sample_data):
        """Test prediction generation"""
        # First train models
        result = ml_agent.train_models(sample_data)
        assert result['success']
        
        # Test prediction generation
        prediction = ml_agent.generate_prediction(sample_data.iloc[-10:, :-1])
        assert prediction['success']
        assert 'prediction' in prediction
        
    def test_update_model(self, ml_agent, sample_data):
        """Test model update functionality"""
        # First train models
        result = ml_agent.train_models(sample_data)
        assert result['success']
        
        # Store current model info
        current_model = ml_agent.current_model
        current_update_time = ml_agent.model_update_timestamp
        
        # Test model update too soon
        result = ml_agent.update_model(sample_data)
        assert not result['success']
        assert 'reason' in result
        
        # Test model update after waiting
        # For testing, we'll bypass the time check
        ml_agent.model_update_timestamp = None
        result = ml_agent.update_model(sample_data)
        assert result['success']
        
        # Check if model was updated
        if current_model['model_name'] != result['current_model']:
            assert result['current_model'] in ml_agent.active_models
            assert ml_agent.model_update_timestamp is not None
        
    @pytest.mark.asyncio
    async def test_start(self, ml_agent):
        """Test starting the ML agent"""
        result = await ml_agent.start()
        assert isinstance(result, dict)
        assert 'success' in result
        
    @pytest.mark.asyncio
    async def test_emergency_stop(self, ml_agent):
        """Test emergency stop functionality"""
        result = await ml_agent.emergency_stop()
        assert isinstance(result, dict)
        assert 'success' in result