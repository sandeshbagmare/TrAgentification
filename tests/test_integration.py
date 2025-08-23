"""
Integration Tests for AI Trading System
Tests the complete system workflow from data ingestion to order execution
"""
import pytest
import asyncio
import os
import sys
import time
import pandas as pd
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_agent import DataAgent
from agents.technical_agent import TechnicalAnalysisAgent
from agents.ml_agent import MlAgent
from agents.strategy_orchestrator import StrategyOrchestrator
from agents.risk_manager import RiskManager
from agents.execution_engine import ExecutionEngine
from agents.monitoring_agent import MonitoringAgent

@pytest.mark.integration
class TestSystemIntegration:
    """Integration tests for the complete trading system"""
    
    @pytest.fixture
    async def mock_trading_system(self):
        """Create a mock trading system for testing"""
        # Mock credentials
        credentials = {
            'upstox_api_key': 'test_upstox_key',
            'upstox_api_secret': 'test_upstox_secret',
            'gemini_api_key': 'test_gemini_key',
            'azure_openai_api_key': 'test_azure_key',
            'azure_openai_endpoint': 'https://test.openai.azure.com/'
        }
        
        # Initialize agents
        data_agent = DataAgent(
            market_data_sources=['upstox'],
            historical_data_dir='test_data',
            tick_interval=1
        )
        
        technical_agent = TechnicalAnalysisAgent(
            indicators=['rsi', 'macd', 'ma']
        )
        
        ml_agent = MlAgent(
            model_dir='test_models',
            training_interval=3600,
            models_to_train=['lightgbm']
        )
        
        strategy_orchestrator = StrategyOrchestrator(
            confidence_threshold=0.7,
            require_consenus=True
        )
        
        risk_manager = RiskManager(
            max_risk_per_trade=0.005,
            daily_loss_cap=0.03,
            stop_loss_levels=['fixed']
        )
        
        execution_engine = ExecutionEngine(
            upstox_api_key=credentials['upstox_api_key'],
            upstox_api_secret=credentials['upstox_api_secret'],
            order_type='limit',
            slippage_tolerance=0.001
        )
        
        monitoring_agent = MonitoringAgent(
            dashboard_port=8001,  # Different port for testing
            log_dir='test_logs',
            alert_channels=['email']
        )\n        \n        return {\n            'data_agent': data_agent,\n            'technical_agent': technical_agent,\n            'ml_agent': ml_agent,\n            'strategy_orchestrator': strategy_orchestrator,\n            'risk_manager': risk_manager,\n            'execution_engine': execution_engine,\n            'monitoring_agent': monitoring_agent\n        }\n    \n    @pytest.mark.asyncio\n    async def test_complete_trading_workflow(self, mock_trading_system):\n        \"\"\"Test the complete workflow from data to order execution\"\"\"\n        agents = mock_trading_system\n        \n        # Step 1: Mock market data\n        mock_market_data = {\n            'symbol': 'RELIANCE',\n            'open': 2500.0,\n            'high': 2520.0,\n            'low': 2495.0,\n            'close': 2510.0,\n            'volume': 1000000,\n            'timestamp': '2023-06-01T10:00:00Z'\n        }\n        \n        # Step 2: Technical analysis\n        with patch.object(agents['data_agent'], 'get_market_data') as mock_data:\n            mock_data.return_value = mock_market_data\n            \n            # Get historical data for technical analysis\n            historical_data = pd.DataFrame([\n                {'open': 2480, 'high': 2500, 'low': 2475, 'close': 2495, 'volume': 800000},\n                {'open': 2495, 'high': 2515, 'low': 2490, 'close': 2505, 'volume': 900000},\n                {'open': 2505, 'high': 2525, 'low': 2500, 'close': 2510, 'volume': 1000000}\n            ])\n            \n            technical_analysis = agents['technical_agent'].analyze_data(\n                historical_data, 'RELIANCE'\n            )\n            \n            assert 'overall_signal' in technical_analysis\n            assert 'confidence' in technical_analysis\n        \n        # Step 3: Strategy orchestration\n        signals = {\n            'technical': {\n                'signal': technical_analysis['overall_signal'],\n                'confidence': technical_analysis['confidence']\n            }\n        }\n        \n        trading_decision = agents['strategy_orchestrator'].make_trading_decision(\n            signals, mock_market_data\n        )\n        \n        assert 'signal' in trading_decision\n        assert 'confidence' in trading_decision\n        \n        # Step 4: Risk management validation\n        if trading_decision['signal'] != 0:\n            order_details = {\n                'symbol': 'RELIANCE',\n                'quantity': 100,\n                'price': 2510.0,\n                'side': 'buy' if trading_decision['signal'] == 1 else 'sell',\n                'order_type': 'limit'\n            }\n            \n            risk_validation = await agents['risk_manager'].validate_order(order_details)\n            \n            # Risk manager should either approve or reject the order\n            assert 'approved' in risk_validation\n            assert isinstance(risk_validation['approved'], bool)\n            \n            # Step 5: Order execution (if approved)\n            if risk_validation['approved']:\n                with patch.object(agents['execution_engine'], '_make_api_request') as mock_api:\n                    mock_api.return_value = {\n                        'status': 'success',\n                        'data': {\n                            'order_id': 'TEST_ORDER_123',\n                            'status': 'success'\n                        }\n                    }\n                    \n                    execution_result = await agents['execution_engine'].place_order(order_details)\n                    \n                    assert execution_result['success'] == True\n                    assert 'order_id' in execution_result\n    \n    @pytest.mark.asyncio\n    async def test_ai_integration_workflow(self, mock_trading_system):\n        \"\"\"Test workflow with AI integrations\"\"\"\n        agents = mock_trading_system\n        \n        # Mock AI integrations\n        with patch('agents.strategy_orchestrator.GeminiIntegration') as mock_gemini:\n            with patch('agents.strategy_orchestrator.AzureOpenAIIntegration') as mock_azure:\n                # Setup mock AI responses\n                mock_gemini_instance = MagicMock()\n                mock_gemini_instance.generate_trading_signal.return_value = {\n                    'success': True,\n                    'signal': 1,\n                    'confidence': 0.85,\n                    'source': 'Gemini'\n                }\n                mock_gemini.return_value = mock_gemini_instance\n                \n                mock_azure_instance = MagicMock()\n                mock_azure_instance.generate_trading_signal.return_value = {\n                    'success': True,\n                    'signal': 1,\n                    'confidence': 0.80,\n                    'source': 'Azure OpenAI'\n                }\n                mock_azure.return_value = mock_azure_instance\n                \n                # Add AI integrations to strategy orchestrator\n                agents['strategy_orchestrator'].ai_integrations = {\n                    'gemini': mock_gemini_instance,\n                    'azure_openai': mock_azure_instance\n                }\n                \n                # Test AI signal generation\n                market_data = {\n                    'symbol': 'RELIANCE',\n                    'price': 2510.0,\n                    'volume': 1000000\n                }\n                \n                signals = {}\n                signals = agents['strategy_orchestrator'].add_ai_signal(\n                    signals, 'gemini', market_data\n                )\n                signals = agents['strategy_orchestrator'].add_ai_signal(\n                    signals, 'azure_openai', market_data\n                )\n                \n                # Should have AI signals\n                assert 'gemini' in signals\n                assert 'azure_openai' in signals\n                assert signals['gemini']['success'] == True\n                assert signals['azure_openai']['success'] == True\n    \n    @pytest.mark.asyncio\n    async def test_error_handling_and_recovery(self, mock_trading_system):\n        \"\"\"Test system behavior under error conditions\"\"\"\n        agents = mock_trading_system\n        \n        # Test data agent failure\n        with patch.object(agents['data_agent'], 'get_market_data') as mock_data:\n            mock_data.side_effect = Exception(\"API connection failed\")\n            \n            # System should handle data failure gracefully\n            result = await agents['data_agent'].get_market_data('RELIANCE')\n            assert result == {}  # Should return empty dict on failure\n        \n        # Test execution engine failure\n        with patch.object(agents['execution_engine'], '_make_api_request') as mock_api:\n            mock_api.side_effect = Exception(\"Order placement failed\")\n            \n            order_details = {\n                'symbol': 'RELIANCE',\n                'quantity': 100,\n                'price': 2510.0,\n                'side': 'buy',\n                'order_type': 'limit'\n            }\n            \n            result = await agents['execution_engine'].place_order(order_details)\n            assert result['success'] == False\n            assert 'error' in result\n        \n        # Test risk manager circuit breaker\n        large_order = {\n            'symbol': 'RELIANCE',\n            'quantity': 10000,  # Very large quantity\n            'price': 2510.0,\n            'side': 'buy',\n            'order_type': 'limit'\n        }\n        \n        risk_result = await agents['risk_manager'].validate_order(large_order)\n        # Should reject due to position size\n        assert risk_result['approved'] == False\n    \n    @pytest.mark.asyncio\n    async def test_performance_and_timing(self, mock_trading_system):\n        \"\"\"Test system performance and timing requirements\"\"\"\n        agents = mock_trading_system\n        \n        # Test data processing speed\n        start_time = time.time()\n        \n        # Simulate multiple market data requests\n        with patch.object(agents['data_agent'], 'get_market_data') as mock_data:\n            mock_data.return_value = {\n                'symbol': 'RELIANCE',\n                'price': 2510.0,\n                'volume': 1000000\n            }\n            \n            # Process multiple symbols\n            symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']\n            for symbol in symbols:\n                await agents['data_agent'].get_market_data(symbol)\n        \n        processing_time = time.time() - start_time\n        \n        # Should process 5 symbols in under 1 second\n        assert processing_time < 1.0\n        \n        # Test signal generation speed\n        start_time = time.time()\n        \n        signals = {\n            'technical': {'signal': 1, 'confidence': 0.8},\n            'ml': {'signal': 1, 'confidence': 0.7}\n        }\n        \n        decision = agents['strategy_orchestrator'].make_trading_decision(\n            signals, {'symbol': 'RELIANCE', 'price': 2510.0}\n        )\n        \n        decision_time = time.time() - start_time\n        \n        # Signal processing should be very fast\n        assert decision_time < 0.1\n        assert 'signal' in decision\n\n\n@pytest.mark.integration\n@pytest.mark.slow\nclass TestSystemLoad:\n    \"\"\"Load testing for the trading system\"\"\"\n    \n    @pytest.mark.asyncio\n    async def test_concurrent_order_processing(self):\n        \"\"\"Test handling multiple concurrent orders\"\"\"\n        execution_engine = ExecutionEngine(\n            upstox_api_key='test_key',\n            upstox_api_secret='test_secret',\n            order_type='limit',\n            slippage_tolerance=0.001\n        )\n        \n        # Mock the API calls\n        with patch.object(execution_engine, '_make_api_request') as mock_api:\n            mock_api.return_value = {\n                'status': 'success',\n                'data': {\n                    'order_id': 'TEST_ORDER',\n                    'status': 'success'\n                }\n            }\n            \n            # Create multiple orders\n            orders = []\n            for i in range(10):\n                orders.append({\n                    'symbol': f'STOCK_{i}',\n                    'quantity': 100,\n                    'price': 1000 + i,\n                    'side': 'buy',\n                    'order_type': 'limit'\n                })\n            \n            # Process orders concurrently\n            tasks = [execution_engine.place_order(order) for order in orders]\n            results = await asyncio.gather(*tasks)\n            \n            # All orders should be processed successfully\n            assert len(results) == 10\n            for result in results:\n                assert result['success'] == True\n    \n    @pytest.mark.asyncio\n    async def test_high_frequency_data_processing(self):\n        \"\"\"Test processing high-frequency market data\"\"\"\n        data_agent = DataAgent(\n            market_data_sources=['upstox'],\n            tick_interval=1\n        )\n        \n        # Mock rapid data updates\n        with patch.object(data_agent, 'get_market_data') as mock_data:\n            # Simulate 100 data points in rapid succession\n            data_points = []\n            for i in range(100):\n                mock_data.return_value = {\n                    'symbol': 'RELIANCE',\n                    'price': 2500 + i * 0.1,\n                    'volume': 1000 + i,\n                    'timestamp': f'2023-06-01T10:{i:02d}:00Z'\n                }\n                \n                data_point = await data_agent.get_market_data('RELIANCE')\n                data_points.append(data_point)\n            \n            # Should process all data points\n            assert len(data_points) == 100\n            \n            # Data should be properly sequenced\n            for i, point in enumerate(data_points):\n                expected_price = 2500 + i * 0.1\n                assert point['price'] == expected_price\n\n\nif __name__ == \"__main__\":\n    # Run integration tests\n    pytest.main([__file__, \"-v\", \"-m\", \"integration\"])