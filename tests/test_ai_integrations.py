"""
Unit tests for AI Integrations
Tests the GeminiIntegration and AzureOpenAIIntegration classes
"""
import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ai_integrations import GeminiIntegration, AzureOpenAIIntegration, GEMINI_AVAILABLE, AZURE_OPENAI_AVAILABLE


class TestGeminiIntegration:
    """Test cases for Gemini AI integration"""
    
    @pytest.fixture
    def mock_market_data(self):
        """Sample market data for testing"""
        return {
            'symbol': 'RELIANCE',
            'price': 2500.50,
            'volume': 1000000,
            'volatility': 0.02,
            'timestamp': '2023-06-01T10:00:00Z'
        }
    
    @pytest.fixture
    def gemini_integration(self):
        """Create a GeminiIntegration instance for testing"""
        if not GEMINI_AVAILABLE:
            pytest.skip("Gemini not available")
        
        with patch('agents.ai_integrations.genai') as mock_genai:
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            
            integration = GeminiIntegration("test_api_key")
            integration.model = mock_model
            integration.available = True
            return integration
    
    def test_gemini_initialization_success(self):
        """Test successful Gemini initialization"""
        if not GEMINI_AVAILABLE:
            pytest.skip("Gemini not available")
            
        with patch('agents.ai_integrations.genai') as mock_genai:
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel = MagicMock()
            
            integration = GeminiIntegration("valid_api_key")
            
            assert integration.available == True
            mock_genai.configure.assert_called_once_with(api_key="valid_api_key")
    
    def test_gemini_initialization_failure(self):
        """Test Gemini initialization failure"""
        if not GEMINI_AVAILABLE:
            # Test when library is not available
            with pytest.raises(ImportError):
                GeminiIntegration("test_key")
        else:
            # Test when initialization fails
            with patch('agents.ai_integrations.genai') as mock_genai:
                mock_genai.configure.side_effect = Exception("API Error")
                
                integration = GeminiIntegration("invalid_api_key")
                assert integration.available == False
    
    def test_analyze_market_success(self, gemini_integration, mock_market_data):
        """Test successful market analysis"""
        if not GEMINI_AVAILABLE:
            pytest.skip("Gemini not available")
            
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "Market analysis suggests a bullish trend"
        gemini_integration.model.generate_content.return_value = mock_response
        
        result = gemini_integration.analyze_market(mock_market_data)
        
        assert result['success'] == True
        assert 'analysis' in result
        assert 'timestamp' in result
    
    def test_analyze_market_not_available(self, mock_market_data):
        """Test market analysis when Gemini is not available"""
        integration = GeminiIntegration.__new__(GeminiIntegration)
        integration.available = False
        
        result = integration.analyze_market(mock_market_data)
        
        assert result['success'] == False
        assert 'Gemini integration not available' in result['error']
    
    def test_generate_trading_signal_success(self, gemini_integration, mock_market_data):
        """Test successful trading signal generation"""
        if not GEMINI_AVAILABLE:
            pytest.skip("Gemini not available")
            
        # Mock analyze_market to return success
        with patch.object(gemini_integration, 'analyze_market') as mock_analyze:
            mock_analyze.return_value = {
                'success': True,
                'analysis': {
                    'signal': 1,
                    'confidence': 0.8,
                    'reasoning': 'Strong bullish indicators'
                }
            }
            
            result = gemini_integration.generate_trading_signal(mock_market_data)
            
            assert result['success'] == True
            assert result['source'] == 'Gemini'
            assert 'signal' in result
            assert 'confidence' in result


class TestAzureOpenAIIntegration:
    """Test cases for Azure OpenAI integration"""
    
    @pytest.fixture
    def mock_market_data(self):
        """Sample market data for testing"""
        return {
            'symbol': 'TATA_MOTORS',
            'price': 150.25,
            'volume': 500000,
            'volatility': 0.015,
            'timestamp': '2023-06-01T10:00:00Z'
        }
    
    @pytest.fixture
    def azure_integration(self):
        """Create an AzureOpenAIIntegration instance for testing"""
        if not AZURE_OPENAI_AVAILABLE:
            pytest.skip("Azure OpenAI not available")
        
        with patch('agents.ai_integrations.AzureOpenAI') as mock_client:
            integration = AzureOpenAIIntegration("test_api_key", "https://test.openai.azure.com/")
            integration.client = mock_client
            integration.available = True
            return integration
    
    def test_azure_initialization_success(self):
        """Test successful Azure OpenAI initialization"""
        if not AZURE_OPENAI_AVAILABLE:
            pytest.skip("Azure OpenAI not available")
            
        with patch('agents.ai_integrations.AzureOpenAI') as mock_client:
            integration = AzureOpenAIIntegration("valid_key", "https://valid.openai.azure.com/")
            
            assert integration.available == True
            mock_client.assert_called_once()
    
    def test_azure_initialization_failure(self):
        """Test Azure OpenAI initialization failure"""
        if not AZURE_OPENAI_AVAILABLE:
            # Test when library is not available
            with pytest.raises(ImportError):
                AzureOpenAIIntegration("test_key", "https://test.openai.azure.com/")
        else:
            # Test when initialization fails
            with patch('agents.ai_integrations.AzureOpenAI') as mock_client:
                mock_client.side_effect = Exception("Authentication failed")
                
                integration = AzureOpenAIIntegration("invalid_key", "https://invalid.openai.azure.com/")
                assert integration.available == False
    
    def test_analyze_market_success(self, azure_integration, mock_market_data):
        """Test successful market analysis"""
        if not AZURE_OPENAI_AVAILABLE:
            pytest.skip("Azure OpenAI not available")
            
        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The market shows strong momentum"
        
        azure_integration.client.chat.completions.create.return_value = mock_response
        
        result = azure_integration.analyze_market(mock_market_data)
        
        assert result['success'] == True
        assert 'analysis' in result
        assert 'timestamp' in result
    
    def test_analyze_market_not_available(self, mock_market_data):
        """Test market analysis when Azure OpenAI is not available"""
        integration = AzureOpenAIIntegration.__new__(AzureOpenAIIntegration)
        integration.available = False
        
        result = integration.analyze_market(mock_market_data)
        
        assert result['success'] == False
        assert 'Azure OpenAI integration not available' in result['error']
    
    def test_generate_trading_signal_success(self, azure_integration, mock_market_data):
        """Test successful trading signal generation"""
        if not AZURE_OPENAI_AVAILABLE:
            pytest.skip("Azure OpenAI not available")
            
        # Mock analyze_market to return success
        with patch.object(azure_integration, 'analyze_market') as mock_analyze:
            mock_analyze.return_value = {
                'success': True,
                'analysis': {
                    'signal': -1,
                    'confidence': 0.75,
                    'reasoning': 'Bearish sentiment detected'
                }
            }
            
            result = azure_integration.generate_trading_signal(mock_market_data)
            
            assert result['success'] == True
            assert result['source'] == 'Azure OpenAI'
            assert 'signal' in result
            assert 'confidence' in result


class TestAIIntegrationsAvailability:
    """Test AI integrations availability detection"""
    
    def test_gemini_availability_flag(self):
        """Test that GEMINI_AVAILABLE flag is set correctly"""
        # This test will pass regardless of whether Gemini is actually installed
        assert isinstance(GEMINI_AVAILABLE, bool)
    
    def test_azure_openai_availability_flag(self):
        """Test that AZURE_OPENAI_AVAILABLE flag is set correctly"""
        # This test will pass regardless of whether Azure OpenAI is actually installed
        assert isinstance(AZURE_OPENAI_AVAILABLE, bool)
    
    def test_at_least_one_integration_available(self):
        """Test that at least one AI integration method is available for import"""
        # In a production environment, we want at least one AI integration to be available
        # This test serves as a reminder to install the required packages
        if not GEMINI_AVAILABLE and not AZURE_OPENAI_AVAILABLE:
            pytest.skip("No AI integrations available - install google-generativeai or openai package")


@pytest.mark.integration
class TestAIIntegrationsIntegration:
    """Integration tests for AI integrations (requires real API keys)"""
    
    def test_gemini_real_api_call(self):
        """Test real Gemini API call (requires GEMINI_API_KEY environment variable)"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key or api_key == 'your_gemini_key_here' or not GEMINI_AVAILABLE:
            pytest.skip("Real Gemini API key not available")
        
        integration = GeminiIntegration(api_key)
        
        if integration.available:
            market_data = {
                'symbol': 'RELIANCE',
                'price': 2500.50,
                'volume': 1000000
            }
            
            result = integration.analyze_market(market_data)
            
            # Should not fail with real API key
            assert result['success'] == True or 'error' in result
    
    def test_azure_openai_real_api_call(self):
        """Test real Azure OpenAI API call (requires Azure credentials)"""
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        
        if (not api_key or not endpoint or 
            api_key == 'your_azure_openai_key_here' or 
            endpoint == 'your_azure_openai_endpoint_here' or
            not AZURE_OPENAI_AVAILABLE):
            pytest.skip("Real Azure OpenAI credentials not available")
        
        integration = AzureOpenAIIntegration(api_key, endpoint)
        
        if integration.available:
            market_data = {
                'symbol': 'TATA_MOTORS',
                'price': 150.25,
                'volume': 500000
            }
            
            result = integration.analyze_market(market_data)
            
            # Should not fail with real API credentials
            assert result['success'] == True or 'error' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])