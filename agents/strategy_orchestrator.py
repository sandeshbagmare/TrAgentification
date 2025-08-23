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

@dataclass
class SignalWeight:
    """Configuration for signal source weights"""
    technical: float = 1.0
    statistical: float = 0.9
    ml: float = 1.1
    gemini: float = 0.8
    azure_openai: float = 0.8
    sentiment: float = 0.6
    
class MarketRegimeRule(StrategyRule):
    """Rule that adjusts signal weights based on market regime"""
    
    def __init__(self, priority: int = 2):
        super().__init__(priority)
        # Define regime-specific signal weights
        self.regime_weights = {
            'trending': {
                'technical': 1.2,  # Technical analysis works well in trending markets
                'statistical': 0.8, # Mean reversion less effective
                'ml': 1.0
            },
            'mean_reverting': {
                'technical': 0.8,
                'statistical': 1.3, # Statistical strategies excel in mean-reverting markets
                'ml': 1.0
            },
            'volatile': {
                'technical': 0.9,
                'statistical': 0.7,
                'ml': 1.1  # ML might handle volatility better
            },
            'sideways': {
                'technical': 0.7,
                'statistical': 1.1,
                'ml': 0.9
            }
        }
    
    def evaluate(self, data: Dict) -> Dict:
        """Adjust signal weights based on detected market regime"""
        market_data = data.get('market_data', {})
        regime = market_data.get('regime', 'unknown')
        
        if regime in self.regime_weights:
            return {
                'signal': 0,  # This rule doesn't generate signals, just adjusts weights
                'confidence': 0.8,
                'weights': self.regime_weights[regime],
                'reason': f'Market regime detected: {regime}'
            }
        
        return {
            'signal': 0,
            'confidence': 0.0,
            'reason': 'Unknown market regime'
        }

class VolatilityAdjustmentRule(StrategyRule):
    """Rule that adjusts position sizing based on volatility"""
    
    def __init__(self, priority: int = 5):
        super().__init__(priority)
        self.base_volatility = 0.015  # 1.5% daily volatility baseline
        
    def evaluate(self, data: Dict) -> Dict:
        """Adjust signal strength based on current volatility"""
        market_data = data.get('market_data', {})
        current_volatility = market_data.get('volatility', self.base_volatility)
        
        # Calculate volatility adjustment factor
        vol_ratio = current_volatility / self.base_volatility
        
        if vol_ratio > 2.0:  # High volatility
            adjustment_factor = 0.5  # Reduce signal strength
            reason = f"High volatility detected ({current_volatility:.1%}), reducing signal strength"
        elif vol_ratio < 0.5:  # Low volatility
            adjustment_factor = 1.2  # Increase signal strength
            reason = f"Low volatility detected ({current_volatility:.1%}), increasing signal strength"
        else:
            adjustment_factor = 1.0
            reason = "Normal volatility levels"
        
        return {
            'signal': 0,
            'confidence': 0.9,
            'volatility_adjustment': adjustment_factor,
            'reason': reason
        }

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
    Combines signals from different sources with priority rules and intelligent weighting
    """
    
    def __init__(self, confidence_threshold: float = 0.7, 
                 require_consensus: bool = True):
        # Strategy rules (higher priority first)
        self.rules = {
            'market_condition': MarketConditionRule(priority=1),
            'time_based': TimeBasedRule(priority=2),
            'market_regime': MarketRegimeRule(priority=3),
            'consensus': ConsensusRule(priority=4),
            'confidence': ConfidenceRule(priority=5),
            'volatility_adjustment': VolatilityAdjustmentRule(priority=6)
        }
        
        # Default signal weights
        self.signal_weights = SignalWeight()
        
        # Confidence threshold for signal validation
        self.confidence_threshold = confidence_threshold
        
        # Flag to require consensus
        self.require_consensus = require_consensus
        
        # Signal history for learning
        self.signal_history = []
        self.performance_history = []
        
        # Initialize AI integrations
        self.ai_integrations = {}
        self._initialize_ai_integrations()
        
    def _initialize_ai_integrations(self):
        """Initialize AI integrations if API keys are available"""
        if not AI_INTEGRATIONS_AVAILABLE:
            logger.warning("AI integrations module not available")
            return
        
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
                
                if ai_signal.get('success', False):
                    signals[source] = {
                        'signal': ai_signal.get('signal', 0),
                        'confidence': ai_signal.get('confidence', 0.5),
                        'reason': ai_signal.get('reason', f'AI signal from {source}')
                    }
                else:
                    logger.error(f"Error generating AI signal from {source}: {ai_signal.get('error')}")
            except Exception as e:
                logger.error(f"Error generating AI signal from {source}: {str(e)}")
        
    def combine_signals(self, raw_signals: Dict, market_data: Dict) -> Dict:
        """Combine signals from different sources with priority rules and intelligent weighting"""
        # Add signals from AI integrations
        signals = raw_signals.copy()
        signals = self.add_ai_signal(signals, 'gemini', market_data)
        signals = self.add_ai_signal(signals, 'azure_openai', market_data)
        
        # Store original signals for analysis
        analysis_data = {
            'signals': signals,
            'market_data': market_data,
            'current_time': datetime.now()
        }
        
        # Apply rules to get adjustments and filters
        rule_results = {}
        signal_weights = {}
        adjustment_factors = {'volatility_adjustment': 1.0}
        
        # Evaluate all rules
        for rule_name, rule in self.rules.items():
            try:
                evaluation = rule.evaluate(analysis_data)
                rule_results[rule_name] = evaluation
                
                # Extract weights if provided
                if 'weights' in evaluation:
                    signal_weights.update(evaluation['weights'])
                
                # Extract adjustment factors
                if 'volatility_adjustment' in evaluation:
                    adjustment_factors['volatility_adjustment'] = evaluation['volatility_adjustment']
                
                # Check for blocking conditions (negative signals from high priority rules)
                if (evaluation['signal'] < 0 and 
                    rule.priority <= 2 and  # High priority rules
                    evaluation['confidence'] > 0.8):
                    return {
                        'signal': 0,  # Block trading
                        'confidence': evaluation['confidence'],
                        'source': f"blocked_by_{rule_name}",
                        'reason': evaluation['reason'],
                        'rule_results': rule_results
                    }
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {str(e)}")
                continue
        
        # Calculate weighted signal combination
        weighted_result = self.calculate_weighted_signal(signals, signal_weights)
        
        # Apply adjustment factors
        adjusted_confidence = weighted_result['confidence'] * adjustment_factors['volatility_adjustment']
        
        # Final decision logic
        final_signal = weighted_result['signal']
        
        # Apply confidence threshold
        if adjusted_confidence < self.confidence_threshold:
            final_signal = 0
            reason = f"Signal filtered out - confidence {adjusted_confidence:.2f} below threshold {self.confidence_threshold}"
        else:
            reason = weighted_result['reason']
        
        # Store signal for learning
        signal_record = {
            'timestamp': datetime.now(),
            'final_signal': final_signal,
            'confidence': adjusted_confidence,
            'raw_signals': signals,
            'rule_results': rule_results,
            'market_data': market_data
        }
        self.signal_history.append(signal_record)
        
        # Keep only last 1000 signals in memory
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
        
        return {
            'signal': final_signal,
            'confidence': adjusted_confidence,
            'raw_signal': weighted_result.get('raw_signal', final_signal),
            'source': 'strategy_orchestrator',
            'reason': reason,
            'signal_breakdown': weighted_result.get('signal_details', []),
            'rule_results': rule_results,
            'adjustment_factors': adjustment_factors,
            'timestamp': datetime.now().isoformat()
        }
        
    def get_signal_confidence(self, signal_data: Dict) -> float:
        """Calculate overall confidence for a signal"""
        base_confidence = signal_data.get('confidence', 0.0)
        
        # Adjust confidence based on number of agreeing sources
        signals = signal_data.get('raw_signals', {})
        if signals:
            agreeing_sources = sum(1 for s in signals.values() 
                                 if isinstance(s, dict) and 
                                 s.get('signal') == signal_data.get('signal'))
            total_sources = len(signals)
            
            if total_sources > 0:
                consensus_factor = agreeing_sources / total_sources
                adjusted_confidence = base_confidence * (0.5 + 0.5 * consensus_factor)
                return min(1.0, adjusted_confidence)
        
        return base_confidence
    
    def filter_signal(self, signal_data: Dict) -> Dict:
        """Apply additional filtering to signals before execution"""
        # Check signal age
        signal_time = signal_data.get('timestamp')
        if signal_time:
            try:
                signal_dt = datetime.fromisoformat(signal_time.replace('Z', '+00:00'))
                age_minutes = (datetime.now() - signal_dt.replace(tzinfo=None)).total_seconds() / 60
                
                if age_minutes > 5:  # Signal older than 5 minutes
                    return {
                        'signal': 0,
                        'confidence': 0.0,
                        'reason': f'Signal too old ({age_minutes:.1f} minutes)'
                    }
            except Exception as e:
                logger.error(f"Error parsing signal timestamp: {e}")
        
        # Check confidence threshold
        confidence = self.get_signal_confidence(signal_data)
        if confidence < self.confidence_threshold:
            return {
                'signal': 0,
                'confidence': confidence,
                'reason': f'Low confidence ({confidence:.2f} < {self.confidence_threshold})'
            }
        
        # Signal passes all filters
        return signal_data
    
    def update_performance(self, signal_id: str, actual_return: float):
        """Update performance tracking for signal learning"""
        performance_record = {
            'signal_id': signal_id,
            'actual_return': actual_return,
            'timestamp': datetime.now()
        }
        self.performance_history.append(performance_record)
        
        # Keep only last 1000 performance records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def analyze_signal_performance(self) -> Dict:
        """Analyze historical signal performance for learning"""
        if len(self.performance_history) < 10:
            return {'insufficient_data': True}
        
        # Calculate performance by signal type
        performance_by_source = {}
        
        for record in self.performance_history:
            # This would require matching performance to signal history
            # For now, return basic statistics
            pass
        
        returns = [r['actual_return'] for r in self.performance_history]
        
        return {
            'total_signals': len(self.performance_history),
            'average_return': np.mean(returns),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        }
    
    def calculate_weighted_signal(self, signals: Dict, weights: Dict = None) -> Dict:
        """Calculate weighted average of all signals"""
        if not signals:
            return {'signal': 0, 'confidence': 0.0, 'reason': 'No signals available'}
        
        # Use provided weights or default weights
        if weights is None:
            weights = {
                'technical': self.signal_weights.technical,
                'statistical': self.signal_weights.statistical,
                'ml': self.signal_weights.ml,
                'gemini': self.signal_weights.gemini,
                'azure_openai': self.signal_weights.azure_openai
            }
        
        total_weighted_signal = 0.0
        total_weight = 0.0
        total_confidence = 0.0
        signal_details = []
        
        for source, signal_data in signals.items():
            if not isinstance(signal_data, dict):
                continue
                
            signal = signal_data.get('signal', 0)
            confidence = signal_data.get('confidence', 0.5)
            
            # Get weight for this source
            source_weight = weights.get(source, 0.5)
            
            # Calculate weighted contribution
            weighted_signal = signal * confidence * source_weight
            total_weighted_signal += weighted_signal
            total_weight += source_weight
            total_confidence += confidence * source_weight
            
            signal_details.append({
                'source': source,
                'signal': signal,
                'confidence': confidence,
                'weight': source_weight,
                'contribution': weighted_signal
            })
        
        # Calculate final signal and confidence
        if total_weight > 0:
            final_signal = total_weighted_signal / total_weight
            avg_confidence = total_confidence / total_weight
            
            # Convert to discrete signal
            if final_signal > 0.3:
                discrete_signal = 1
            elif final_signal < -0.3:
                discrete_signal = -1
            else:
                discrete_signal = 0
            
            return {
                'signal': discrete_signal,
                'raw_signal': final_signal,
                'confidence': avg_confidence,
                'total_weight': total_weight,
                'signal_details': signal_details,
                'reason': f'Weighted combination of {len(signals)} signals'
            }
        
        return {'signal': 0, 'confidence': 0.0, 'reason': 'No valid signals to combine'}
        
    def evaluate_signal(self, signal: Dict, market_data: Dict) -> Dict:
        """Evaluate a trading signal against risk criteria"""
        # Apply signal filtering
        filtered_signal = self.filter_signal(signal)
        
        if filtered_signal.get('signal') == 0:
            return filtered_signal
        
        # Additional risk checks can be added here
        # For now, just return the filtered signal with evaluation metadata
        
        return {
            **filtered_signal,
            'evaluated': True,
            'evaluation_time': datetime.now().isoformat()
        }
        
    def make_trading_decision(self, signals: Dict, market_data: Dict) -> Dict:
        """Make a trading decision based on all inputs"""
        try:
            # Combine signals using intelligent weighting
            combined_signal = self.combine_signals(signals, market_data)
            
            # Evaluate signal against risk criteria
            final_signal = self.evaluate_signal(combined_signal, market_data)
            
            # Add comprehensive metadata
            final_signal.update({
                'decision_id': f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                'input_signals_count': len(signals),
                'market_data_quality': self._assess_market_data_quality(market_data),
                'processing_time': datetime.now().isoformat()
            })
            
            logger.info(f"Trading decision: {final_signal['signal']} (confidence: {final_signal.get('confidence', 0):.2f})")
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error making trading decision: {e}")
            return {
                'signal': 0,
                'confidence': 0.0,
                'reason': f'Error in decision making: {str(e)}',
                'error': True,
                'timestamp': datetime.now().isoformat()
            }
    
    def _assess_market_data_quality(self, market_data: Dict) -> str:
        """Assess the quality of market data for decision making"""
        required_fields = ['close', 'volume', 'timestamp']
        
        missing_fields = [field for field in required_fields if field not in market_data]
        
        if not missing_fields:
            return 'high'
        elif len(missing_fields) <= 1:
            return 'medium'
        else:
            return 'low'
    
    def get_strategy_status(self) -> Dict:
        """Get current status of the strategy orchestrator"""
        return {
            'active_rules': list(self.rules.keys()),
            'ai_integrations': list(self.ai_integrations.keys()),
            'signal_history_length': len(self.signal_history),
            'performance_history_length': len(self.performance_history),
            'confidence_threshold': self.confidence_threshold,
            'require_consensus': self.require_consensus,
            'last_signal_time': self.signal_history[-1]['timestamp'].isoformat() if self.signal_history else None
        }
    
    async def start(self):
        """Start the strategy orchestrator"""
        logger.info("Starting strategy orchestrator...")
        
        # Initialize all components
        status = self.get_strategy_status()
        
        logger.info(f"Strategy orchestrator started with {len(status['active_rules'])} rules")
        logger.info(f"AI integrations available: {status['ai_integrations']}")
        
        return {
            'success': True,
            'message': 'Strategy orchestrator started successfully',
            'status': status
        }
        
    async def emergency_stop(self):
        """Emergency stop for the strategy orchestrator"""
        logger.info("Initiating emergency stop for strategy orchestrator...")
        
        # Clear signal history to prevent stale signals
        self.signal_history.clear()
        
        # Disable all rules temporarily
        disabled_rules = list(self.rules.keys())
        self.rules.clear()
        
        logger.info(f"Emergency stop completed. Disabled {len(disabled_rules)} rules.")
        
        return {
            'success': True,
            'message': 'Strategy orchestrator stopped successfully',
            'disabled_rules': disabled_rules
        }