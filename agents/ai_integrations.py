"""
AI Integrations for AI Trading System
Responsible for integrating with external AI services like Gemini and Azure OpenAI
"""
import os
import logging
import requests
from typing import Dict, List, Optional, Union
from datetime import datetime

# Import AI libraries with error handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Generative AI not available. Install with: pip install google-generativeai")

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    logging.warning("Azure OpenAI not available. Install with: pip install openai azure-identity")

logger = logging.getLogger(__name__)

class GeminiIntegration:
    """Integration with Google Gemini API for trading signals and analysis"""
    
    def __init__(self, api_key: str):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library not available. Install with: pip install google-generativeai")
            
        self.api_key = api_key
        self.available = False
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.available = True
            logger.info("Gemini integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            self.available = False
        
    def analyze_market(self, market_data: Dict) -> Dict:
        """Analyze market data using Gemini AI"""
        if not self.available:
            return {
                'success': False,
                'error': 'Gemini integration not available'
            }
            
        try:
            prompt = self._create_market_analysis_prompt(market_data)
            response = self.model.generate_content(prompt)
            
            # Parse and validate response
            analysis = self._parse_analysis_response(response.text)
            
            return {
                'success': True,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing market with Gemini: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def _create_market_analysis_prompt(self, market_data: Dict) -> str:
        """Create a prompt for market analysis based on current data"""
        # Implementation for creating market analysis prompt
        return "Analyze the following market data and provide trading insights:\n" + str(market_data)
        
    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse and validate Gemini's response"""
        # Implementation for parsing analysis response
        # This would extract trading signals, sentiment, etc.
        return {
            'raw_response': response_text,
            'signal': 0,  # Placeholder for trading signal (-1, 0, 1)
            'confidence': 0.5,  # Placeholder for confidence score
            'reasoning': 'No reasoning extracted'  # Placeholder for reasoning
        }
        
    def generate_trading_signal(self, market_data: Dict) -> Dict:
        """Generate trading signal using Gemini AI"""
        analysis = self.analyze_market(market_data)
        
        if analysis['success']:
            # Convert analysis to trading signal
            signal = self._convert_analysis_to_signal(analysis['analysis'])
            
            return {
                'success': True,
                'signal': signal,
                'confidence': analysis['analysis']['confidence'],
                'source': 'Gemini',
                'timestamp': datetime.now().isoformat()
            }
        
        return analysis
        
    def _convert_analysis_to_signal(self, analysis: Dict) -> int:
        """Convert Gemini's analysis to trading signal"""
        # Implementation for converting analysis to trading signal
        # This would be based on the analysis content
        return 0  # Placeholder for actual signal


class AzureOpenAIIntegration:
    """Integration with Azure OpenAI API for trading signals and analysis"""
    
    def __init__(self, api_key: str, endpoint: str):
        if not AZURE_OPENAI_AVAILABLE:
            raise ImportError("Azure OpenAI library not available. Install with: pip install openai azure-identity")
            
        self.api_key = api_key
        self.endpoint = endpoint
        self.available = False
        
        try:
            # Initialize Azure OpenAI client
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2023-05-15",
                azure_endpoint=endpoint
            )
            self.available = True
            logger.info("Azure OpenAI integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI: {str(e)}")
            self.available = False
        
    def analyze_market(self, market_data: Dict) -> Dict:
        """Analyze market data using Azure OpenAI"""
        if not self.available:
            return {
                'success': False,
                'error': 'Azure OpenAI integration not available'
            }
            
        try:
            prompt = self._create_market_analysis_prompt(market_data)
            
            # Call Azure OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-35-turbo",  # Replace with your deployment name
                messages=[
                    {"role": "system", "content": "You are a financial market analysis AI."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse and validate response
            analysis = self._parse_analysis_response(response.choices[0].message.content)
            
            return {
                'success': True,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing market with Azure OpenAI: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def _create_market_analysis_prompt(self, market_data: Dict) -> str:
        """Create a prompt for market analysis based on current data"""
        # Implementation for creating market analysis prompt
        return "Analyze the following market data and provide trading insights:\n" + str(market_data)
        
    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse and validate Azure OpenAI's response"""
        # Implementation for parsing analysis response
        # This would extract trading signals, sentiment, etc.
        return {
            'raw_response': response_text,
            'signal': 0,  # Placeholder for trading signal (-1, 0, 1)
            'confidence': 0.5,  # Placeholder for confidence score
            'reasoning': 'No reasoning extracted'  # Placeholder for reasoning
        }
        
    def generate_trading_signal(self, market_data: Dict) -> Dict:
        """Generate trading signal using Azure OpenAI"""
        analysis = self.analyze_market(market_data)
        
        if analysis['success']:
            # Convert analysis to trading signal
            signal = self._convert_analysis_to_signal(analysis['analysis'])
            
            return {
                'success': True,
                'signal': signal,
                'confidence': analysis['analysis']['confidence'],
                'source': 'Azure OpenAI',
                'timestamp': datetime.now().isoformat()
            }
        
        return analysis
        
    def _convert_analysis_to_signal(self, analysis: Dict) -> int:
        """Convert Azure OpenAI's analysis to trading signal"""
        # Implementation for converting analysis to trading signal
        # This would be based on the analysis content
        return 0  # Placeholder for actual signal