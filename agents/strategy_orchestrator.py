"""
Strategy Orchestrator Agent for AI Trading System
Responsible for combining signals from different sources and making final trading decisions
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime

# Import AI integrations
from agents.ai_integrations import GeminiIntegration, AzureOpenAIIntegration

logger = logging.getLogger(__name__)

class StrategyRule:
    """Base class for strategy rules"""
    
    def __init__(self, priority: int = 1):
        self.priority = priority  # Higher priority rules are evaluated first
        
    def evaluate(self, data: Dict) -> Dict:
        """Evaluate the rule and generate a trading signal"""
        pass


class ConsensusRule(StrategyRule):
    """Rule that requires consensus among multiple signal sources"""
    
    def __init__(self, consensus_threshold: float = 0.7, priority: int = 1):
        super().__init__(priority)
        self.consensus_threshold = consensus_threshold  # Required percentage of sources agreeing
        
    def evaluate(self, data: Dict) -> Dict:
        """Evaluate based on consensus of multiple signal sources"""
        signals = data.get('signals', {})
        
        # Count buy and sell signals
        buy_signals = sum(1 for s in signals.values() if s.get('signal') == 1)
        sell_signals = sum(1 for s in signals.values() if s.get('signal') == -1)
        total_signals = len(signals)
        
        # Calculate confidence
        buy_confidence = buy_signals / total_signals if total_signals > 0 else 0
        sell_confidence = sell_signals / total_signals if total_signals > 0 else 0
        
        # Check for consensus
        if buy_confidence >= self.consensus_threshold:
            return {
                'signal': 1,
                'confidence': buy_confidence,
                'reason': f"Consensus buy signal ({buy_signals}/{total_signals} sources agree)"
            }
        elif sell_confidence >= self.consensus_threshold:
            return {
                'signal': -1,
                'confidence': sell_confidence,
                'reason': f"Consensus sell signal ({sell_signals}/{total_signals} sources agree)"
            }
        
        return {
            'signal': 0,
            'confidence': max(buy_confidence, sell_confidence),
            'reason': "No consensus among signal sources"
        }


class ConfidenceRule(StrategyRule):
    """Rule that evaluates signals based on confidence scores"""
    
    def __init__(self, confidence_threshold: float = 0.6, priority: int = 2):
        super().__init__(priority)
        self.confidence_threshold = confidence_threshold  # Minimum confidence to act
        
    def evaluate(self, data: Dict) -> Dict:
        """Evaluate based on confidence of individual signals"""
        signals = data.get('signals', {})
        
        # Find highest confidence signal
        best_signal = None
        best_confidence = 0
        
        for source, signal_data in signals.items():
            confidence = signal_data.get('confidence', 0)
            signal = signal_data.get('signal', 0)
            
            if abs(signal) > 0 and confidence > best_confidence:
                best_confidence = confidence
                best_signal = {
                    'signal': signal,
                    'confidence': confidence,
                    'source': source,
                    'reason': f"High confidence signal from {source} ({confidence:.2f})"
                }
        
        if best_confidence >= self.confidence_threshold and best_signal:
            return best_signal
        
        return {
            'signal': 0,
            'confidence': best_confidence,
            'reason': "No high confidence signals"
        }


class MarketConditionRule(StrategyRule):
    """Rule that evaluates market conditions before allowing trades"""
    
    def __init__(self, priority: int = 3):
        super().__init__(priority)
        # Market condition thresholds
        self.volatility_threshold = 0.02  # 2% daily volatility
        self.liquidity_threshold = 1000000  # $1M average daily volume
        
    def evaluate(self, data: Dict) -> Dict:
        """Evaluate based on current market conditions"""
        market_data = data.get('market_data', {})
        
        # Check volatility
        volatility = market_data.get('volatility', 0)
        if volatility > self.volatility_threshold:
            return {
                'signal': -1,
                'confidence': 0.9,
                'reason': f"Market too volatile ({volatility:.2%} vs threshold {self.volatility_threshold:.2%})"
            }
        
        # Check liquidity
        liquidity = market_data.get('liquidity', 0)
        if liquidity < self.liquidity_threshold:
            return {
                'signal': -1,
                'confidence': 0.8,
                'reason': f"Market insufficiently liquid ({liquidity} vs threshold {self.liquidity_threshold})"
            }
        
        # Market conditions are acceptable
        return {
            'signal': 1,  # Neutral signal, just indicating conditions are acceptable
            'confidence': 0.95,
            'reason': "Market conditions acceptable"
        }


class TimeBasedRule(StrategyRule):
    """Rule that enforces time-based trading constraints"""
    
    def __init__(self, priority: int = 4):
        super().__init__(priority)
        # Trading hours configuration
        self.trading_hours = {
            'start': 9 * 60 + 15,  # 9:15 AM in minutes
            'end': 15 * 60 + 30,  # 3:30 PM in minutes
            'close_position_time': 15 * 60 + 20  # 3:20 PM
        }
        
    def evaluate(self, data: Dict) -> Dict:
        """Evaluate based on current time and market schedule"""
        current_time = data.get('current_time', datetime.now())
        
        # Convert current time to minutes since midnight
        current_minutes = current_time.hour * 60 + current_time.minute
        
        # Check if within trading hours
        if current_minutes < self.trading_hours['start']:
            return {
                'signal': -1,
                'confidence': 1.0,
                'reason': "Before trading hours"
            }
        elif current_minutes > self.trading_hours['end']:
            return {
                'signal': -1,
                'confidence': 1.0,
                'reason': "After trading hours"
            }
        
        # Check if we should close positions before market close
        if current_minutes > self.trading_hours['close_position_time']:
            return {
                'signal': -1,
                'confidence': 1.0,
                'reason': "Closing positions before market close"
            }
        
        # Trading is allowed
        return {
            'signal': 1,  # Neutral signal, just indicating time is acceptable
            'confidence': 1.0,
            'reason': "Within trading hours"
        }


class StrategyOrchestrator:
    """
    Strategy Orchestrator Agent
    Combines signals from different sources with priority rules
    """
    
    def __init__(self, confidence_threshold: float = 0.7, 
                 require_consenus: bool = True):
        # Strategy rules (higher priority first)
        self.rules = [
            MarketConditionRule(priority=3),
            TimeBasedRule(priority=4),
            ConsensusRule(priority=1),
            ConfidenceRule(priority=2)
        ]
        
        # Confidence threshold for signal validation
        self.confidence_threshold = confidence_threshold
        
        # Flag to require consensus
        self.require_consenus = require_consenus
        
        # Initialize AI integrations
        self.ai_integrations = {}
        self._initialize_ai_integrations()
        
    def _initialize_ai_integrations(self):
        """Initialize AI integrations if API keys are available"""
        # Get API keys from environment variables
        gemini_key = os.getenv('GEMINI_API_KEY')
        azure_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        
        ai_integration_count = 0
        
        # Initialize Gemini integration if key is available
        if gemini_key and gemini_key != 'your_gemini_key_here' and len(gemini_key) > 10:
            try:
                self.ai_integrations['gemini'] = GeminiIntegration(gemini_key)
                ai_integration_count += 1
                logger.info("Gemini integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini integration: {str(e)}")
        
        # Initialize Azure OpenAI integration if keys are available
        if (azure_key and azure_endpoint and 
            azure_key != 'your_azure_openai_key_here' and 
            azure_endpoint != 'your_azure_openai_endpoint_here' and
            len(azure_key) > 10 and len(azure_endpoint) > 10):
            try:
                self.ai_integrations['azure_openai'] = AzureOpenAIIntegration(
                    azure_key, azure_endpoint
                )
                ai_integration_count += 1
                logger.info("Azure OpenAI integration initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI integration: {str(e)}")
        
        if ai_integration_count == 0:
            logger.warning("No AI integrations available. Please configure Gemini or Azure OpenAI API keys.")
        else:
            logger.info(f"Initialized {ai_integration_count} AI integration(s)")
        
    def add_ai_signal(self, signals: Dict, source: str, market_data: Dict) -> Dict:
        """Add signal from AI integration to the signals dictionary"""
        if source in self.ai_integrations:
            try:
                ai_signal = self.ai_integrations[source].generate_trading_signal(market_data)
                
                if ai_signal['success']:
                    signals[source] = ai_signal
                else:
                    logger.error(f"Error generating AI signal from {source}: {ai_signal.get('error')}")
            except Exception as e:
                logger.error(f"Error generating AI signal from {source}: {str(e)}")
        
        return signals
        
    def combine_signals(self, raw_signals: Dict, market_data: Dict) -> Dict:
        """Combine signals from different sources with priority rules"""
        # Add signals from AI integrations
        signals = raw_signals.copy()
        signals = self.add_ai_signal(signals, 'gemini', market_data)
        signals = self.add_ai_signal(signals, 'azure_openai', market_data)
        
        # Sort rules by priority (highest first)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority)
        
        # Evaluate rules in priority order
        for rule in sorted_rules:
            try:
                evaluation = rule.evaluate({
                    'signals': signals,
                    'market_data': market_data,
                    'current_time': datetime.now()
                })
                
                # If rule generates a non-zero signal, return it
                if evaluation['signal'] != 0:
                    return {
                        'signal': evaluation['signal'],
                        'confidence': evaluation['confidence'],
                        'source': f"rule_{rule.__class__.__name__}",
                        'reason': evaluation['reason']
                    }
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.__class__.__name__}: {str(e)}")
                continue
        
        # If no rule generates a signal, use the best confidence signal
        best_signal = self._get_best_confidence_signal(signals)
        if best_signal:
            return best_signal
        
        # No signal
        return {
            'signal': 0,
            'confidence': 0.0,
            'reason': "No trading signal generated"
        }
        
    def _get_best_confidence_signal(self, signals: Dict) -> Optional[Dict]:
        """Get the signal with highest confidence"""
        # Filter valid signals
        valid_signals = [
            {'signal': s['signal'], 'confidence': s['confidence'], 'source': source}
            for source, s in signals.items()
            if s.get('confidence', 0) > 0 and abs(s.get('signal', 0)) > 0
        ]
        
        if not valid_signals:
            return None
        
        # Sort by confidence
        valid_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return best signal
        best = valid_signals[0]
        return {
            'signal': best['signal'],
            'confidence': best['confidence'],
            'source': best['source'],
            'reason': f"Best confidence signal from {best['source']} ({best['confidence']:.2f})"
        }
        
    def evaluate_signal(self, signal: Dict, market_data: Dict) -> Dict:
        """Evaluate a trading signal against risk criteria"""
        # Implementation for signal evaluation
        # This would include:
        # - Checking against risk management rules
        # - Validating with market conditions
        # - Time-based constraints
        
        # For now, just return the signal
        return signal
        
    def make_trading_decision(self, signals: Dict, market_data: Dict) -> Dict:
        """Make a trading decision based on all inputs"""
        # Combine signals
        combined_signal = self.combine_signals(signals, market_data)
        
        # Evaluate signal against risk criteria
        final_signal = self.evaluate_signal(combined_signal, market_data)
        
        # Add timestamp
        final_signal['timestamp'] = datetime.now().isoformat()
        
        return final_signal
        
    async def start(self):
        """Start the strategy orchestrator"""
        logger.info("Starting strategy orchestrator...")
        # Implementation for starting the agent
        # This would typically involve:
        # - Connecting to data source
        # - Starting analysis loop
        # - Handling configuration updates
        
    async def emergency_stop(self):
        """Emergency stop for the strategy orchestrator"""
        logger.info("Initiating emergency stop for strategy orchestrator...")
        # Implementation for emergency stop procedures
        pass