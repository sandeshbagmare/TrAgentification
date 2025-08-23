"""
Supervisor Agent for AI Trading System
Responsible for resolving conflicting signals and overseeing system operations
"""
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class SignalConflictResolution:
    """Base class for signal conflict resolution strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def resolve_conflict(self, signals: Dict) -> Dict:
        """Resolve conflicting signals from different agents"""
        pass

class PriorityBasedResolution(SignalConflictResolution):
    """Priority-based signal conflict resolution strategy"""
    
    def __init__(self, priority_order: List[str] = ['risk', 'ml', 'technical', 'statistical']):
        super().__init__("priority_based")
        self.priority_order = priority_order  # Order of agent priority
    
    def resolve_conflict(self, signals: Dict) -> Dict:
        """Resolve conflicting signals based on agent priority"""
        # Check if all signals agree
        signal_values = [signal.get('signal', 0) for signal in signals.values()]
        
        # If all signals agree, return that signal
        if len(set(signal_values)) == 1:
            return {
                'signal': signal_values[0],
                'confidence': 1.0,
                'reason': 'All agents agree',
                'resolution_method': self.name
            }
        
        # If there's a strong majority, return that signal
        buy_signals = sum(1 for s in signal_values if s == 1)
        sell_signals = sum(1 for s in signal_values if s == -1)
        
        majority_threshold = 0.6  # 60% majority required
        
        if buy_signals / len(signal_values) >= majority_threshold:
            # Find highest confidence buy signal
            buy_signals = [s for s in signals.values() if s.get('signal') == 1]
            best_signal = max(buy_signals, key=lambda s: s.get('confidence', 0))
            
            return {
                'signal': 1,
                'confidence': best_signal.get('confidence', 0.6),
                'reason': f"Majority buy signal ({buy_signals}/{len(signal_values)} agents agree)",
                'resolution_method': self.name,
                'best_source': best_signal.get('source')
            }
        elif sell_signals / len(signal_values) >= majority_threshold:
            # Find highest confidence sell signal
            sell_signals = [s for s in signals.values() if s.get('signal') == -1]
            best_signal = max(sell_signals, key=lambda s: s.get('confidence', 0))
            
            return {
                'signal': -1,
                'confidence': best_signal.get('confidence', 0.6),
                'reason': f"Majority sell signal ({sell_signals}/{len(signal_values)} agents agree)",
                'resolution_method': self.name,
                'best_source': best_signal.get('source')
            }
        
        # If no majority, check priority order
        for agent_type in self.priority_order:
            if agent_type in signals:
                signal = signals[agent_type]
                if signal.get('signal', 0) != 0:
                    return {
                        'signal': signal.get('signal', 0),
                        'confidence': signal.get('confidence', 0.5),
                        'reason': f"Signal from highest priority agent: {agent_type}",
                        'resolution_method': self.name,
                        'best_source': agent_type
                    }
        
        # Default to no signal
        return {
            'signal': 0,
            'confidence': 0.5,
            'reason': 'No consensus and no priority signal',
            'resolution_method': self.name
        }

class EnsembleVoting(SignalConflictResolution):
    """Ensemble voting signal conflict resolution strategy"""
    
    def __init__(self, name: str = "ensemble_voting"):
        super().__init__(name)
        
        # Weighting for different agents
        self.agent_weights = {
            'risk': 0.3,      # Risk manager has highest weight
            'ml': 0.25,       # ML agent has high weight
            'technical': 0.2, # Technical agent has medium weight
            'statistical': 0.2, # Statistical agent has medium weight
            'data': 0.05      # Data agent has lowest weight
        }
    
    def resolve_conflict(self, signals: Dict) -> Dict:
        """Resolve conflicting signals using weighted ensemble voting"""
        # Calculate weighted signal score
        signal_score = 0.0
        confidence_score = 0.0
        
        # Sum weighted signals and confidence
        for agent_type, signal in signals.items():
            weight = self.agent_weights.get(agent_type, 0.1)  # Default weight for unknown agents
            signal_score += signal.get('signal', 0) * weight
            confidence_score += signal.get('confidence', 0.5) * weight
        
        # Determine final signal
        if signal_score > 0:
            return {
                'signal': 1,
                'confidence': confidence_score,
                'reason': f"Ensemble vote: {signal_score:.2f} (positive)",
                'resolution_method': self.name,
                'weighted_scores': {k: v for k, v in signals.items()}
            }
        elif signal_score < 0:
            return {
                'signal': -1,
                'confidence': confidence_score,
                'reason': f"Ensemble vote: {signal_score:.2f} (negative)",
                'resolution_method': self.name,
                'weighted_scores': {k: v for k, v in signals.items()}
            }
        else:
            return {
                'signal': 0,
                'confidence': 0.5,
                'reason': "Ensemble vote: 0 (neutral)",
                'resolution_method': self.name,
                'weighted_scores': {k: v for k, v in signals.items()}
            }

class StrategySelector:
    """Class for selecting the best strategy based on performance"""
    
    def __init__(self, performance_window: int = 30):
        self.performance_window = performance_window  # days
        self.strategy_performance = {}  # Dictionary to store strategy performance
        
    def update_strategy_performance(self, strategy_name: str, pnl: float, win: bool):
        """Update performance statistics for a strategy"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'total_trades': 0,
                'total_pnl': 0.0,
                'win_count': 0,
                'last_used': datetime.now()
            }
        
        # Update performance
        self.strategy_performance[strategy_name]['total_trades'] += 1
        self.strategy_performance[strategy_name]['total_pnl'] += pnl
        if win:
            self.strategy_performance[strategy_name]['win_count'] += 1
        self.strategy_performance[strategy_name]['last_used'] = datetime.now()
        
        return {'success': True}
    
    def get_strategy_score(self, strategy_name: str) -> float:
        """Calculate a score for a strategy based on performance"""
        if strategy_name not in self.strategy_performance:
            return 0.5  # Default score for unknown strategies
        
        strategy = self.strategy_performance[strategy_name]
        
        # Calculate win rate
        win_rate = strategy['win_count'] / strategy['total_trades'] if strategy['total_trades'] > 0 else 0.5
        
        # Calculate P&L per trade
        avg_pnl = strategy['total_pnl'] / strategy['total_trades'] if strategy['total_trades'] > 0 else 0.0
        pnl_score = max(0.1, min(0.9, (avg_pnl + 0.01) / 0.02))  # Normalize to 0.1-0.9 range
        
        # Calculate recency factor (strategies not used in 7 days get score penalty)
        last_used = strategy['last_used']
        if (datetime.now() - last_used).days > 7:
            recency_factor = 0.5
        else:
            recency_factor = 1.0
        
        # Calculate overall score
        score = (win_rate * 0.5 + pnl_score * 0.5) * recency_factor
        
        return score
    
    def select_best_strategy(self, available_strategies: List[str]) -> Dict:
        """Select the best strategy based on performance metrics"""
        if not available_strategies:
            return {
                'success': False,
                'error': 'No strategies available for selection'
            }
        
        # Calculate scores for all available strategies
        strategy_scores = {
            strategy: self.get_strategy_score(strategy)
            for strategy in available_strategies
        }
        
        # Select strategy with highest score
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        
        return {
            'success': True,
            'selected_strategy': best_strategy,
            'strategy_score': strategy_scores[best_strategy],
            'strategy_scores': strategy_scores,
            'reason': f"Selected {best_strategy} with score {strategy_scores[best_strategy]:.2f}"
        }

class CircuitBreaker:
    """Class for implementing system-wide circuit breakers"""
    
    def __init__(self, daily_loss_cap: float = 0.03, 
                 volatility_threshold: float = 0.05):
        self.daily_loss_cap = daily_loss_cap
        self.volatility_threshold = volatility_threshold
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_start_value = 100000.0  # Default starting portfolio value
        self.daily_reset_time = datetime.now()
        
    def reset_daily_counters(self):
        """Reset daily counters at market close or when system starts"""
        self.daily_reset_time = datetime.now().replace(hour=3, minute=30, second=0, microsecond=0)  # Market open time IST
        self.daily_pnl = 0.0
        self.daily_trades = 0
        
    def update_pnl(self, realized_pnl: float = 0.0, 
                  unrealized_pnl: float = 0.0):
        """Update daily P&L and check circuit breakers"""
        # Reset if new day
        if (datetime.now() - self.daily_reset_time).days >= 1:
            self.reset_daily_counters()
        
        self.daily_pnl += realized_pnl + unrealized_pnl
        self.daily_trades += 1
        
        # Calculate daily loss percentage
        daily_loss_percentage = abs(self.daily_pnl) / self.daily_start_value
        
        # Check daily loss cap
        if daily_loss_percentage >= self.daily_loss_cap:
            return {
                'allow_trade': False,
                'reason': f"Daily loss cap hit ({daily_loss_percentage:.2%} >= {self.daily_loss_cap:.2%})",
                'circuit_breaker': 'daily_loss_cap'
            }
        
        # Check for market volatility (would typically get from risk manager)
        if self.volatility_threshold > 0:
            # In a real system, this would get current volatility from risk manager
            current_volatility = 0.03  # Example value
            if current_volatility >= self.volatility_threshold:
                return {
                    'allow_trade': False,
                    'reason': f"Market volatility too high ({current_volatility:.2%} >= {self.volatility_threshold:.2%})",
                    'circuit_breaker': 'volatility'
                }
        
        # Check if data feeds are available (would typically get from data agent)
        # In a real system, this would check data quality from data agent
        data_quality = 0.95  # Example value
        if data_quality < 0.7:  # 70% minimum data quality
            return {
                'allow_trade': False,
                'reason': f"Data quality too low ({data_quality:.2%} < 70%)",
                'circuit_breaker': 'data_quality'
            }
        
        # Check broker API error rate (would typically get from execution engine)
        # In a real system, this would get error rate from execution engine
        api_error_rate = 0.05  # Example value
        if api_error_rate > 0.2:  # 20% error rate threshold
            return {
                'allow_trade': False,
                'reason': f"Broker API error rate too high ({api_error_rate:.2%} > 20%)",
                'circuit_breaker': 'api_errors'
            }
        
        # All checks passed
        return {
            'allow_trade': True,
            'reason': "All circuit breakers passed",
            'daily_pnl': self.daily_pnl,
            'daily_loss_percentage': daily_loss_percentage
        }
    
    def check_circuit_breakers(self, system_status: Dict) -> Dict:
        """Check all system-wide circuit breakers"""
        # Reset if new day
        if (datetime.now() - self.daily_reset_time).days >= 1:
            self.reset_daily_counters()
        
        # Check daily loss cap
        if 'portfolio_value' in system_status and 'daily_pnl' in system_status:
            daily_loss_percentage = abs(system_status['daily_pnl']) / system_status['portfolio_value']
            if daily_loss_percentage >= self.daily_loss_cap:
                return {
                    'allow_trade': False,
                    'reason': f"Daily loss cap hit ({daily_loss_percentage:.2%} >= {self.daily_loss_cap:.2%})",
                    'circuit_breaker': 'daily_loss_cap'
                }
        
        # Check market volatility
        if 'market_volatility' in system_status:
            if system_status['market_volatility'] >= self.volatility_threshold:
                return {
                    'allow_trade': False,
                    'reason': f"Market volatility too high ({system_status['market_volatility']:.2%} >= {self.volatility_threshold:.2%})",
                    'circuit_breaker': 'volatility'
                }
        
        # Check data feed quality
        if 'data_quality' in system_status:
            if system_status['data_quality'] < 0.7:  # Minimum data quality 70%
                return {
                    'allow_trade': False,
                    'reason': f"Data quality too low ({system_status['data_quality']:.2%} < 70%)",
                    'circuit_breaker': 'data_quality'
                }
        
        # Check broker API error rate
        if 'api_error_rate' in system_status:
            if system_status['api_error_rate'] > 0.2:  # 20% error rate threshold
                return {
                    'allow_trade': False,
                    'reason': f"Broker API error rate too high ({system_status['api_error_rate']:.2%} > 20%)",
                    'circuit_breaker': 'api_errors'
                }
        
        # All checks passed
        return {
            'allow_trade': True,
            'reason': "All circuit breakers passed",
            'daily_pnl': self.daily_pnl,
            'daily_loss_percentage': daily_loss_percentage if 'portfolio_value' in system_status else 0
        }

class SupervisorAgent:
    """
    Supervisor Agent
    Oversees all other agents, resolves conflicting signals, and makes the final trade decision
    """
    
    def __init__(self, confidence_threshold: float = 0.7, 
                 signal_timeout: float = 5.0, 
                 max_trades_per_day: int = 200):
        self.confidence_threshold = confidence_threshold
        self.signal_timeout = signal_timeout
        self.max_trades_per_day = max_trades_per_day
        self.daily_trades = 0
        self.daily_reset_time = datetime.now()
        
        # Initialize signal conflict resolution strategies
        self.resolution_strategies = {
            'priority_based': PriorityBasedResolution(),
            'ensemble_voting': EnsembleVoting()
        }
        self.default_resolution = 'ensemble_voting'
        
        # Initialize strategy selector
        self.strategy_selector = StrategySelector()
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # System status
        self.system_status = {
            'healthy': True,
            'last_check': datetime.now().isoformat(),
            'agent_health': {},
            'portfolio_value': 100000.0,  # Default starting portfolio value
            'daily_pnl': 0.0,
            'market_volatility': 0.02,  # Default market volatility
            'data_quality': 0.95,  # Default data quality
            'api_error_rate': 0.05  # Default API error rate
        }
        
    async def get_signals(self, data: Dict) -> Dict:
        """Get signals from all strategy agents with timeout"""
        try:
            # Get signals from technical analysis agent
            technical_signal = await self._get_technical_signal(data)
            
            # Get signals from ML agent
            ml_signal = await self._get_ml_signal(data)
            
            # Get signals from statistical agent
            statistical_signal = await self._get_statistical_signal(data)
            
            # Combine signals
            signals = {
                'technical': technical_signal,
                'ml': ml_signal,
                'statistical': statistical_signal
            }
            
            return {
                'success': True,
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Error getting signals: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _get_technical_signal(self, data: Dict) -> Dict:
        """Get signal from technical analysis agent"""
        # In a real system, this would communicate with the technical agent
        # For now, return a mock signal
        return {
            'signal': 1,
            'confidence': 0.8,
            'source': 'technical',
            'message': 'Strong technical signal'
        }
    
    async def _get_ml_signal(self, data: Dict) -> Dict:
        """Get signal from ML agent"""
        # In a real system, this would communicate with the ML agent
        # For now, return a mock signal
        return {
            'signal': 1,
            'confidence': 0.7,
            'source': 'ml',
            'message': 'ML model prediction'
        }
    
    async def _get_statistical_signal(self, data: Dict) -> Dict:
        """Get signal from statistical agent"""
        # In a real system, this would communicate with the statistical agent
        # For now, return a mock signal
        return {
            'signal': 0,
            'confidence': 0.5,
            'source': 'statistical',
            'message': 'No clear statistical signal'
        }
    
    async def resolve_conflict(self, signals: Dict, resolution_method: str = None) -> Dict:
        """Resolve conflicting signals using the specified method"""
        if not signals:
            return {
                'success': False,
                'error': 'No signals to resolve'
            }
        
        resolution_method = resolution_method or self.default_resolution
        
        if resolution_method not in self.resolution_strategies:
            return {
                'success': False,
                'error': f"Unknown resolution method: {resolution_method}"
            }
        
        # Get the resolution strategy
        resolution_strategy = self.resolution_strategies[resolution_method]
        
        # Resolve the conflict
        resolved_signal = resolution_strategy.resolve_conflict(signals)
        
        return {
            'success': True,
            'signal': resolved_signal['signal'],
            'confidence': resolved_signal['confidence'],
            'reason': resolved_signal['reason'],
            'resolution_method': resolution_method
        }
    
    async def check_circuit_breakers(self, system_status: Dict) -> Dict:
        """Check system-wide circuit breakers"""
        # Update system status
        self.system_status.update({
            'last_check': datetime.now().isoformat(),
            'portfolio_value': system_status.get('portfolio_value', self.system_status['portfolio_value']),
            'market_volatility': system_status.get('market_volatility', self.system_status['market_volatility']),
            'data_quality': system_status.get('data_quality', self.system_status['data_quality']),
            'api_error_rate': system_status.get('api_error_rate', self.system_status['api_error_rate'])
        })
        
        # Check circuit breakers
        circuit_breaker_result = self.circuit_breaker.check_circuit_breakers(self.system_status)
        
        # Update system status
        self.system_status['healthy'] = circuit_breaker_result['allow_trade']
        
        return {
            'allow_trade': circuit_breaker_result['allow_trade'],
            'reason': circuit_breaker_result['reason'],
            'system_status': self.system_status
        }
    
    async def make_final_decision(self, signals: Dict, system_status: Dict) -> Dict:
        """Make the final trade decision after resolving conflicts and checking circuit breakers"""
        # Check if we've reached daily trade limit
        await self._check_daily_trade_limit()
        
        # Check system-wide circuit breakers
        circuit_breaker_result = await self.check_circuit_breakers(system_status)
        
        if not circuit_breaker_result['allow_trade']:
            return {
                'signal': 0,
                'confidence': 1.0,
                'reason': f"Circuit breaker active: {circuit_breaker_result['reason']}",
                'resolution_method': 'circuit_breaker'
            }
        
        # Resolve any conflicting signals
        resolution_result = await self.resolve_conflict(signals)
        
        # Check confidence threshold
        if resolution_result['confidence'] < self.confidence_threshold:
            return {
                'signal': 0,
                'confidence': resolution_result['confidence'],
                'reason': f"Signal confidence below threshold ({resolution_result['confidence']:.2f} < {self.confidence_threshold:.2f})",
                'resolution_method': 'confidence_threshold'
            }
        
        # Check daily trade limit
        if self.daily_trades >= self.max_trades_per_day:
            return {
                'signal': 0,
                'confidence': 0.0,
                'reason': f"Daily trade limit reached ({self.max_trades_per_day})",
                'resolution_method': 'daily_trade_limit'
            }
        
        # Final decision is to execute the signal
        return {
            'signal': resolution_result['signal'],
            'confidence': resolution_result['confidence'],
            'reason': resolution_result['reason'],
            'resolution_method': resolution_result['resolution_method']
        }
    
    async def _check_daily_trade_limit(self):
        """Check and reset daily trade counter if needed"""
        # Reset daily counter if new day
        if (datetime.now() - self.daily_reset_time).days >= 1:
            self.daily_trades = 0
            self.daily_reset_time = datetime.now()
        
        return {
            'success': True,
            'daily_trades': self.daily_trades,
            'trade_limit': self.max_trades_per_day,
            'remaining_trades': max(0, self.max_trades_per_day - self.daily_trades)
        }
    
    async def record_trade(self, trade: Dict, pnl: float = 0.0, is_win: bool = False):
        """Record a trade and update strategy performance"""
        # Update daily trade count
        self.daily_trades += 1
        
        # Update strategy performance
        strategy = trade.get('strategy', 'unknown')
        self.strategy_selector.update_strategy_performance(strategy, pnl, is_win)
        
        return {
            'success': True,
            'daily_trades': self.daily_trades,
            'remaining_trades': max(0, self.max_trades_per_day - self.daily_trades)
        }
    
    async def get_best_strategy(self, available_strategies: List[str]) -> Dict:
        """Get the best performing strategy based on historical performance"""
        # Update with strategy selector
        selection_result = self.strategy_selector.select_best_strategy(available_strategies)
        
        if not selection_result['success']:
            return selection_result
        
        best_strategy = selection_result['selected_strategy']
        
        return {
            'success': True,
            'selected_strategy': best_strategy,
            'score': selection_result['strategy_score'],
            'reason': selection_result['reason']
        }
    
    async def start(self):
        """Start the supervisor agent"""
        logger.info("Starting supervisor agent...")
        # Implementation for starting the agent
        # This would typically involve:
        # - Connecting to system components
        # - Initializing signal resolution strategies
        # - Starting monitoring loop
        
    async def emergency_stop(self):
        """Emergency stop for the supervisor agent"""
        logger.info("Initiating emergency stop for supervisor agent...")
        # Implementation for emergency stop procedures
        # This would include:
        # - Halting all trading
        # - Closing positions
        # - Notifying other agents
        # - Putting system in safe mode
        pass