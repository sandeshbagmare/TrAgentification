"""
Unit tests for the Technical Analysis Agent
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from agents.technical_agent import TechnicalAnalysisAgent, MovingAverage, RSI, MACD, BollingerBands, ATR

# Sample OHLC data for testing
@pytest.fixture
def sample_data():
    """Create sample OHLC data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    np.random.seed(42)
    
    # Create sample OHLC data with upward trend
    close = np.cumsum(np.random.normal(0, 1, 30)) + 100
    open = close - np.random.normal(0, 0.5, 30)
    high = close + np.random.normal(0, 1, 30)
    low = close - np.random.normal(0, 1, 30)
    volume = np.random.randint(1000, 5000, 30)
    
    data = pd.DataFrame({
        'date': dates,
        'open': open,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    data = data.set_index('date')
    return data

# Create TechnicalAnalysisAgent instance for testing
@pytest.fixture
def technical_agent():
    """Create a TechnicalAnalysisAgent instance for testing"""
    return TechnicalAnalysisAgent(indicators=['ma', 'rsi', 'macd', 'bollinger', 'atr'])

class TestTechnicalIndicators:
    """Test suite for technical analysis indicators"""
    
    def test_moving_average(self, sample_data):
        """Test Moving Average indicator calculation"""
        ma = MovingAverage(fast_period=10, slow_period=20)
        result = ma.calculate(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'ma_fast' in result.columns
        assert 'ma_slow' in result.columns
        assert not result['ma_fast'].isna().all()
        assert not result['ma_slow'].isna().all()
        
        # Test signal generation
        signal = ma.generate_signal(result)
        assert isinstance(signal, dict)
        assert 'signal' in signal
        assert 'message' in signal
        
    def test_rsi(self, sample_data):
        """Test RSI indicator calculation"""
        rsi = RSI(period=14)
        result = rsi.calculate(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'rsi' in result.columns
        assert not result['rsi'].isna().all()
        
        # Test signal generation
        signal = rsi.generate_signal(result)
        assert isinstance(signal, dict)
        assert 'signal' in signal
        assert 'message' in signal
        
    def test_macd(self, sample_data):
        """Test MACD indicator calculation"""
        macd = MACD()
        result = macd.calculate(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'ema_fast' in result.columns
        assert 'ema_slow' in result.columns
        assert 'macd_line' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_hist' in result.columns
        assert not result['macd_line'].isna().all()
        
        # Test signal generation
        signal = macd.generate_signal(result)
        assert isinstance(signal, dict)
        assert 'signal' in signal
        assert 'message' in signal
        
    def test_bollinger_bands(self, sample_data):
        """Test Bollinger Bands indicator calculation"""
        bollinger = BollingerBands()
        result = bollinger.calculate(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'ma' in result.columns
        assert 'std' in result.columns
        assert 'upper_band' in result.columns
        assert 'lower_band' in result.columns
        assert not result['upper_band'].isna().all()
        
        # Test signal generation
        signal = bollinger.generate_signal(result)
        assert isinstance(signal, dict)
        assert 'signal' in signal
        assert 'message' in signal
        
    def test_atr(self, sample_data):
        """Test ATR indicator calculation"""
        atr = ATR()
        result = atr.calculate(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'tr0' in result.columns
        assert 'tr1' in result.columns
        assert 'tr2' in result.columns
        assert 'tr' in result.columns
        assert 'atr' in result.columns
        assert not result['atr'].isna().all()
        
        # Test signal generation
        signal = atr.generate_signal(result)
        assert isinstance(signal, dict)
        assert 'signal' in signal
        assert 'message' in signal

class TestTechnicalAnalysisAgent:
    """Test suite for the TechnicalAnalysisAgent class"""
    
    def test_initialization(self, technical_agent):
        """Test basic initialization of TechnicalAnalysisAgent"""
        assert len(technical_agent.active_indicators) > 0
        assert 'ma' in technical_agent.active_indicators
        assert 'rsi' in technical_agent.active_indicators
        
    def test_analyze_data(self, technical_agent, sample_data):
        """Test analyze_data method with sample OHLC data"""
        signal_data = {
            'signals': {
                'technical': {
                    'data': sample_data
                }
            },
            'confidence': 0.75,
            'symbol': 'RELIANCE',
            'timestamp': '2023-06-01T10:00:00Z'
        }
        
        result = technical_agent.analyze_data(signal_data)
        
        assert isinstance(result, dict)
        assert 'signal' in result
        assert 'confidence' in result
        assert 'reason' in result
        assert 'rule_results' in result
        
    def test_combine_signals(self, technical_agent):
        """Test combine_signals method with different signal scenarios"""
        # Test with no signals
        signal_data = {}
        result = technical_agent.combine_signals(signal_data)
        assert result['signal'] == 0
        
        # Test with buy signals
        signal_data = {
            'signals': {
                'technical': {'signal': 1, 'confidence': 0.8, 'message': 'Strong technical signal'},
                'ml': {'signal': 1, 'confidence': 0.7, 'message': 'ML model prediction'}
            },
            'confidence': 0.75,
            'symbol': 'RELIANCE',
            'timestamp': '2023-06-01T10:00:00Z'
        }
        
        result = technical_agent.combine_signals(signal_data)
        assert result['signal'] == 1
        assert result['confidence'] > 0.7
        
        # Test with conflicting signals
        signal_data = {
            'signals': {
                'technical': {'signal': 1, 'confidence': 0.8, 'message': 'Strong technical signal'},
                'ml': {'signal': -1, 'confidence': 0.7, 'message': 'ML model prediction'}
            },
            'confidence': 0.75,
            'symbol': 'RELIANCE',
            'timestamp': '2023-06-01T10:00:00Z'
        }
        
        result = technical_agent.combine_signals(signal_data)
        assert result['signal'] == 0
        assert result['confidence'] < 0.75
        
        # Test with sell signals
        signal_data = {
            'signals': {
                'technical': {'signal': -1, 'confidence': 0.8, 'message': 'Strong technical signal'},
                'ml': {'signal': -1, 'confidence': 0.7, 'message': 'ML model prediction'}
            },
            'confidence': 0.75,
            'symbol': 'RELIANCE',
            'timestamp': '2023-06-01T10:00:00Z'
        }
        
        result = technical_agent.combine_signals(signal_data)
        assert result['signal'] == -1
        assert result['confidence'] > 0.7
        
    @pytest.mark.asyncio
    async def test_start(self, technical_agent):
        """Test starting the technical analysis agent"""
        result = await technical_agent.start()
        assert isinstance(result, dict)
        assert 'success' in result
        
    @pytest.mark.asyncio
    async def test_emergency_stop(self, technical_agent):
        """Test emergency stop functionality"""
        result = await technical_agent.emergency_stop()
        assert isinstance(result, dict)
        assert 'success' in result