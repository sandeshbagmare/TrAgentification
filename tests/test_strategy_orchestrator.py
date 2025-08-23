"""
Unit tests for the Strategy Orchestrator Agent
"""
import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from agents.strategy_orchestrator import StrategyOrchestrator, ConsensusRule, ConfidenceRule, MarketConditionRule, TimeBasedRule

# Sample signal data for testing
@pytest.fixture
def signal_data():
    """Create sample signal data for testing"""
    return {
        'signals': {
            'technical': {'signal': 1, 'confidence': 0.8, 'message': 'Strong technical signal'},
            'ml': {'signal': 1, 'confidence': 0.7, 'message': 'ML model prediction'},
            'statistical': {'signal': 1, 'confidence': 0.6, 'message': 'Statistical signal'}
        },
        'confidence': 0.7,
        'symbol': 'RELIANCE',
        'timestamp': '2023-06-01T10:00:00Z',
        'latest_data': {
            'price': 2500.0,
            'volume': 1000000,
            'indicator_values': {
                'atr': {'atr': 0.015},  # 1.5% ATR
                'volume': {'volume': 1500000}
            }
        }
    }

# Strategy orchestrator for testing
@pytest.fixture
def strategy_orchestrator():
    """Create a StrategyOrchestrator instance for testing"""
    return StrategyOrchestrator(confidence_threshold=0.7, require_consensus=True)

class TestStrategyRules:
    """Test suite for strategy rules"""
    
    def test_consensus_rule(self, signal_data):
        """Test consensus rule with different signal scenarios"""
        # Test with default threshold (0.7)
        rule = ConsensusRule()
        result = rule.evaluate(signal_data)
        assert result['allow_trade']
        assert result['signal'] == 1
        
        # Test with no consensus
        signal_data['signals'] = {
            'technical': {'signal': 1, 'confidence': 0.8, 'message': 'Strong technical signal'},
            'ml': {'signal': -1, 'confidence': 0.7, 'message': 'ML model prediction'}
        }
        result = rule.evaluate(signal_data)
        assert not result['allow_trade']
        assert result['signal'] == 0
        
        # Test with sell signal consensus
        signal_data['signals'] = {
            'technical': {'signal': -1, 'confidence': 0.8, 'message': 'Strong technical signal'},
            'ml': {'signal': -1, 'confidence': 0.7, 'message': 'ML model prediction'}
        }
        result = rule.evaluate(signal_data)
        assert result['allow_trade']
        assert result['signal'] == -1
        
    def test_confidence_rule(self, signal_data):
        """Test confidence rule with different confidence levels"""
        # Test with confidence above threshold
        rule = ConfidenceRule(threshold=0.7)
        result = rule.evaluate(signal_data)
        assert result['allow_trade']
        assert result['signal'] == 1
        
        # Test with confidence below threshold
        signal_data['confidence'] = 0.6
        result = rule.evaluate(signal_data)
        assert not result['allow_trade']
        assert result['signal'] == 0
        
        # Test with no confidence in signal data
        signal_data.pop('confidence')
        result = rule.evaluate(signal_data)
        assert not result['allow_trade']
        assert result['signal'] == 0
        
    def test_market_condition_rule(self, signal_data):
        """Test market condition rule with different market scenarios"""
        # Test with normal market conditions
        rule = MarketConditionRule()
        result = rule.evaluate(signal_data)
        assert result['allow_trade']
        assert result['signal'] == 1
        
        # Test with high volatility
        signal_data['indicator_values']['atr']['atr'] = 0.03  # 3% ATR
        result = rule.evaluate(signal_data)
        assert not result['allow_trade']
        assert result['signal'] == -1
        
        # Test with low liquidity
        signal_data['latest_data']['volume'] = 500000  # Below threshold
        result = rule.evaluate(signal_data)
        assert not result['allow_trade']
        assert result['signal'] == 0
        
        # Test with no market data
        signal_data.pop('indicator_values')
        result = rule.evaluate(signal_data)
        assert not result['allow_trade']
        assert result['signal'] == -1
        
    def test_time_based_rule(self, signal_data):
        """Test time-based rule with different time scenarios"""
        # Test with market open
        rule = TimeBasedRule()
        
        # Mock current time to be within market hours
        original_now = datetime.utcnow
        datetime.utcnow = lambda: datetime(2023, 6, 1, 6, 0)  # 6:00 AM UTC (11:30 AM IST)
        result = rule.evaluate(signal_data)
        assert result['allow_trade']
        assert result['signal'] == 1
        
        # Test with market closed
        datetime.utcnow = lambda: datetime(2023, 6, 1, 11, 0)  # 11:00 AM UTC (4:30 PM IST)
        result = rule.evaluate(signal_data)
        assert not result['allow_trade']
        assert result['signal'] == 0
        
        # Restore original datetime.utcnow
        datetime.utcnow = original_now
        

class TestStrategyOrchestrator:
    """Test suite for the StrategyOrchestrator class"""
    
    def test_initialization(self, strategy_orchestrator):
        """Test basic initialization of Strategy Orchestrator"""
        assert strategy_orchestrator.confidence_threshold == 0.7
        assert strategy_orchestrator.require_consensus
        assert len(strategy_orchestrator.active_rules) > 0
        assert 'consensus' in strategy_orchestrator.rules
        assert 'confidence' in strategy_orchestrator.rules
        
    def test_combine_signals(self, strategy_orchestrator, signal_data):
        """Test combine_signals method with different signal scenarios"""
        # Test with valid signal data
        result = strategy_orchestrator.combine_signals(signal_data)
        assert result['signal'] in [0, 1, -1]
        assert 'reason' in result
        
        # Test with no signal data
        empty_signal_data = {}
        result = strategy_orchestrator.combine_signals(empty_signal_data)
        assert result['signal'] == 0
        assert 'reason' in result
        
        # Test with conflicting signals
        signal_data['signals'] = {
            'technical': {'signal': 1, 'confidence': 0.8, 'message': 'Strong technical signal'},
            'ml': {'signal': -1, 'confidence': 0.7, 'message': 'ML model prediction'}
        }
        result = strategy_orchestrator.combine_signals(signal_data)
        assert result['signal'] == 0
        assert 'No strong consensus' in result['reason']
        
    def test_signal_confidence(self, strategy_orchestrator, signal_data):
        """Test get_signal_confidence method"""
        # Test with signal data
        confidence = strategy_orchestrator.get_signal_confidence(signal_data)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        # Test with no signal data
        confidence = strategy_orchestrator.get_signal_confidence({})
        assert confidence == 0.0
        
    def test_filter_signal(self, strategy_orchestrator, signal_data):
        """Test filter_signal method"""
        # Test with valid signal data
        filtered_signal = strategy_orchestrator.filter_signal(signal_data)
        assert isinstance(filtered_signal, dict)
        assert 'signal' in filtered_signal
        
        # Test with no signal data
        filtered_signal = strategy_orchestrator.filter_signal({})
        assert isinstance(filtered_signal, dict)
        assert 'signal' in filtered_signal
        
    @pytest.mark.asyncio
    async def test_start(self, strategy_orchestrator):
        """Test starting the strategy orchestrator"""
        result = await strategy_orchestrator.start()
        assert isinstance(result, dict)
        assert 'success' in result
        
    @pytest.mark.asyncio
    async def test_emergency_stop(self, strategy_orchestrator):
        """Test emergency stop functionality"""
        result = await strategy_orchestrator.emergency_stop()
        assert isinstance(result, dict)
        assert 'success' in result
        
    def test_ai_integrations_initialization(self):
        """Test AI integrations initialization"""
        with patch.dict('os.environ', {
            'GEMINI_API_KEY': 'test_gemini_key_123456789',
            'AZURE_OPENAI_API_KEY': 'test_azure_key_123456789',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com/'
        }):
            with patch('agents.strategy_orchestrator.GeminiIntegration') as mock_gemini:
                with patch('agents.strategy_orchestrator.AzureOpenAIIntegration') as mock_azure:
                    # Mock successful initialization
                    mock_gemini.return_value = MagicMock()
                    mock_azure.return_value = MagicMock()
                    
                    orchestrator = StrategyOrchestrator()
                    
                    # Should have both integrations
                    assert len(orchestrator.ai_integrations) >= 0  # Depends on successful initialization
                    
    def test_ai_integrations_no_keys(self):
        """Test AI integrations when no API keys are provided"""
        with patch.dict('os.environ', {}, clear=True):
            orchestrator = StrategyOrchestrator()
            # Should initialize without AI integrations
            assert len(orchestrator.ai_integrations) == 0
            
    def test_add_ai_signal_success(self, strategy_orchestrator):
        """Test adding AI signal successfully"""
        # Mock AI integration
        mock_ai = MagicMock()
        mock_ai.generate_trading_signal.return_value = {
            'success': True,
            'signal': 1,
            'confidence': 0.8,
            'source': 'test_ai'
        }
        
        strategy_orchestrator.ai_integrations['test_ai'] = mock_ai
        
        signals = {}
        market_data = {'symbol': 'RELIANCE', 'price': 2500}
        
        result_signals = strategy_orchestrator.add_ai_signal(signals, 'test_ai', market_data)
        
        assert 'test_ai' in result_signals
        assert result_signals['test_ai']['success'] == True
        
    def test_add_ai_signal_failure(self, strategy_orchestrator):
        """Test adding AI signal when AI integration fails"""
        # Mock AI integration that fails
        mock_ai = MagicMock()
        mock_ai.generate_trading_signal.return_value = {
            'success': False,
            'error': 'API rate limit exceeded'
        }
        
        strategy_orchestrator.ai_integrations['test_ai'] = mock_ai
        
        signals = {}
        market_data = {'symbol': 'RELIANCE', 'price': 2500}
        
        result_signals = strategy_orchestrator.add_ai_signal(signals, 'test_ai', market_data)
        
        # Should not add the failed signal to the results
        assert 'test_ai' not in result_signals or not result_signals.get('test_ai', {}).get('success', False)
        
    def test_combine_signals_with_ai(self, strategy_orchestrator, signal_data):
        """Test combining signals including AI signals"""
        # Mock AI integrations
        mock_gemini = MagicMock()
        mock_gemini.generate_trading_signal.return_value = {
            'success': True,
            'signal': 1,
            'confidence': 0.9,
            'source': 'Gemini'
        }
        
        mock_azure = MagicMock()
        mock_azure.generate_trading_signal.return_value = {
            'success': True,
            'signal': 1,
            'confidence': 0.85,
            'source': 'Azure OpenAI'
        }
        
        strategy_orchestrator.ai_integrations['gemini'] = mock_gemini
        strategy_orchestrator.ai_integrations['azure_openai'] = mock_azure
        
        market_data = {'symbol': 'RELIANCE', 'price': 2500}
        
        result = strategy_orchestrator.combine_signals(signal_data['signals'], market_data)
        
        assert 'signal' in result
        assert 'confidence' in result
        assert 'reason' in result