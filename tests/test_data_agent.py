"""
Unit tests for the Data Ingestion Agent
"""
import pytest
import asyncio
import pandas as pd
from agents.data_agent import DataAgent

# Sample configuration for testing
test_config = {
    'market_data_sources': ['upstox', 'finnhub', 'alpha_vantage'],
    'historical_data_dir': 'data/historical',
    'tick_interval': 1
}

# Sample credentials for testing
credentials = {
    'upstox_api_key': 'test_upstox_api_key',
    'upstox_api_secret': 'test_upstox_api_secret',
    'finnhub_api_key': 'test_finnhub_api_key',
    'alpha_vantage_api_key': 'test_alpha_vantage_api_key'
}

@pytest.fixture
async def data_agent():
    """Create and initialize a DataAgent instance for testing"""
    agent = DataAgent(**test_config)
    # Mock the initialize_data_sources method to avoid actual API calls
    agent.initialize_data_sources = lambda x: True
    # Mock the data collection loop to avoid infinite loop in tests
    agent._data_collection_loop = lambda: None
    return agent

class TestDataAgent:
    """Test suite for the DataAgent class"""

    def test_initialization(self, data_agent):
        """Test basic initialization of DataAgent"""
        assert data_agent.market_data_sources == test_config['market_data_sources']
        assert data_agent.historical_data_dir == test_config['historical_data_dir']
        assert data_agent.tick_interval == test_config['tick_interval']
        assert not data_agent.connected
        assert isinstance(data_agent.data_sources, dict)
        assert isinstance(data_agent.data_store, dict)

    def test_initialize_data_sources(self, data_agent):
        """Test data source initialization"""
        # Mock the data sources to avoid actual API calls
        data_agent.data_sources = {'upstox': 'mock_upstox', 'finnhub': 'mock_finnhub'}
        assert len(data_agent.data_sources) == 2

    @pytest.mark.asyncio
    async def test_get_market_data(self, data_agent):
        """Test get_market_data method"""
        # Test with no data sources
        data_agent.connected = False
        result = await data_agent.get_market_data('RELIANCE')
        assert result == {}

        # Test with mock data sources
        data_agent.connected = True
        data_agent.data_sources = {
            'upstox': type('MockDataSource', (), {'get_tick_data': lambda x, y: {'price': 2500.0}}),
            'finnhub': type('MockDataSource', (), {'get_tick_data': lambda x, y: {'price': 2501.0}})
        }
        
        result = await data_agent.get_market_data('RELIANCE')
        assert 'price' in result

    @pytest.mark.asyncio
    async def test_cross_validate_data(self, data_agent):
        """Test cross-validation of data from multiple sources"""
        # Test with no data
        result = data_agent._cross_validate_data({}, 'RELIANCE')
        assert result is None

        # Test with single source
        data = {'upstox': {'price': 2500.0}}
        result = data_agent._cross_validate_data(data, 'RELIANCE')
        assert result == data

        # Test with multiple sources with small discrepancies
        data = {
            'upstox': {'price': 2500.0},
            'finnhub': {'price': 2500.5},
            'alpha_vantage': {'price': 2500.2}
        }
        result = data_agent._cross_validate_data(data, 'RELIANCE')
        assert result is not None
        assert result['price'] == 2500.2  # Average of the prices

        # Test with conflicting data
        data = {
            'upstox': {'price': 2500.0},
            'finnhub': {'price': 3000.0}  # Large discrepancy
        }
        result = data_agent._cross_validate_data(data, 'RELIANCE')
        assert result is None  # Should return None for large discrepancies

    @pytest.mark.asyncio
    async def test_handle_data_validation_failure(self, data_agent):
        """Test handling of data validation failures"""
        # Test basic functionality
        result = await data_agent._handle_data_validation_failure('RELIANCE')
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_start(self, data_agent):
        """Test starting the data agent"""
        # Mock the methods that would make external calls
        data_agent.initialize_data_sources = lambda x: None
        data_agent._data_collection_loop = lambda: None
        
        result = await data_agent.start()
        assert isinstance(result, dict)
        assert 'success' in result

    @pytest.mark.asyncio
    async def test_emergency_stop(self, data_agent):
        """Test emergency stop functionality"""
        result = await data_agent.emergency_stop()
        assert isinstance(result, dict)
        assert 'success' in result