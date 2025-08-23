"""
Risk Management Agent for AI Trading System
Responsible for enforcing strict risk rules and having ultimate veto power over trades
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Data class for tracking positions"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    risk_percentage: float
    volatility: float
    
    def current_value(self, current_price: float) -> float:
        """Calculate current value of the position"""
        return current_price * self.quantity
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized profit/loss"""
        return (current_price - self.entry_price) * self.quantity
    
    def risk_amount(self, current_price: float) -> float:
        """Calculate current risk amount"""
        return abs(self.unrealized_pnl(current_price))

class RiskRule(ABC):
    """Abstract base class for risk rules"""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
    
    @abstractmethod
    def evaluate(self, trade_proposal: Dict, portfolio: Dict, positions: List[Position]) -> Dict:
        """Evaluate the risk rule against a trade proposal"""
        pass

class PositionSizingRule(RiskRule):
    """Rule for enforcing position sizing limits"""
    
    def __init__(self, name: str = "position_sizing", enabled: bool = True, 
                 max_risk_per_trade: float = 0.005, max_position_size: float = 0.05):
        super().__init__(name, enabled)
        self.max_risk_per_trade = max_risk_per_trade  # Max risk per trade as percentage of portfolio
        self.max_position_size = max_position_size  # Max position size as percentage of portfolio
    
    def evaluate(self, trade_proposal: Dict, portfolio: Dict, positions: List[Position]) -> Dict:
        """Evaluate position sizing for the trade proposal"""
        if not self.enabled:
            return {
                'name': self.name,
                'allow_trade': True,
                'risk_score': 0.0,
                'reason': 'Position sizing rule disabled'
            }
        
        try:
            symbol = trade_proposal.get('symbol')
            current_price = trade_proposal.get('price')
            signal = trade_proposal.get('signal')
            
            if signal == 0:
                # No trade, so no risk
                return {
                    'name': self.name,
                    'allow_trade': True,
                    'risk_score': 0.0,
                    'reason': 'No trade signal'
                }
            
            # Calculate position size based on risk
            portfolio_value = portfolio.get('total_value', 0)
            
            # Calculate risk per unit (distance from entry to stop-loss)
            stop_loss_distance = abs(current_price - trade_proposal.get('stop_loss', 0))
            
            # Calculate maximum position size based on risk
            max_position_by_risk = (portfolio_value * self.max_risk_per_trade) / stop_loss_distance
            
            # Calculate position size based on portfolio percentage
            max_position_by_size = portfolio_value * self.max_position_size / current_price
            
            # Use the more conservative position size
            max_position_size = min(max_position_by_risk, max_position_by_size)
            
            # Update trade proposal with calculated position size
            trade_proposal['quantity'] = max_position_size
            
            # Check if position size is valid
            if max_position_size <= 0:
                return {
                    'name': self.name,
                    'allow_trade': False,
                    'risk_score': 1.0,
                    'reason': 'Calculated position size is invalid',
                    'position_size': max_position_size
                }
            
            return {
                'name': self.name,
                'allow_trade': True,
                'risk_score': max_position_size / portfolio_value,
                'reason': f"Position size calculated: {max_position_size} units ({max_position_size / portfolio_value * 100:.2f}% of portfolio)",
                'position_size': max_position_size
            }
            
        except Exception as e:
            logger.error(f"Error evaluating position sizing rule: {str(e)}")
            return {
                'name': self.name,
                'allow_trade': False,
                'risk_score': 1.0,
                'reason': f"Error evaluating position sizing rule: {str(e)}"
            }

class DailyLossCapRule(RiskRule):
    """Rule for enforcing daily loss cap"""
    
    def __init__(self, name: str = "daily_loss_cap", enabled: bool = True, max_daily_loss: float = 0.03):
        super().__init__(name, enabled)
        self.max_daily_loss = max_daily_loss  # Max daily loss as percentage of portfolio
        self.daily_pnl = 0.0
        self.day_start_value = 0.0
        self.reset_time = datetime.now()
        
    def reset_daily_counter(self):
        """Reset daily counters at market close or when system starts"""
        self.day_start_value = datetime.now().replace(hour=3, minute=30, second=0, microsecond=0)  # Market open time IST
        self.daily_pnl = 0.0
        self.reset_time = datetime.now()
    
    def update_pnl(self, realized_pnl: float = 0.0, unrealized_pnl: float = 0.0):
        """Update daily P&L"""
        # Reset if new day
        current_time = datetime.now()
        if current_time.date() != self.reset_time.date():
            self.reset_daily_counter()
        
        self.daily_pnl += realized_pnl + unrealized_pnl
    
    def evaluate(self, trade_proposal: Dict, portfolio: Dict, positions: List[Position]) -> Dict:
        """Evaluate daily loss cap for the trade proposal"""
        if not self.enabled:
            return {
                'name': self.name,
                'allow_trade': True,
                'risk_score': 0.0,
                'reason': 'Daily loss cap rule disabled'
            }
        
        try:
            # Reset if new day
            current_time = datetime.now()
            if current_time.date() != self.reset_time.date():
                self.reset_daily_counter()
            
            portfolio_value = portfolio.get('total_value', 0)
            
            # Calculate potential loss if trade goes against us (distance to stop-loss)
            current_price = trade_proposal.get('price')
            stop_loss = trade_proposal.get('stop_loss', current_price * 0.99)  # Default 1% stop-loss
            quantity = trade_proposal.get('quantity', 0)
            
            # Calculate potential loss
            potential_loss = (current_price - stop_loss) * quantity if trade_proposal.get('signal') == 1 else (stop_loss - current_price) * quantity
            
            # Calculate percentage of portfolio at risk
            risk_percentage = potential_loss / portfolio_value
            
            # Check if we've already hit the daily loss cap
            if self.daily_pnl / portfolio_value <= -self.max_daily_loss:
                return {
                    'name': self.name,
                    'allow_trade': False,
                    'risk_score': 1.0,
                    'reason': f"Daily loss cap already hit ({self.daily_pnl / portfolio_value * 100:.2f}% <= -{self.max_daily_loss * 100}%)",
                    'daily_pnl': self.daily_pnl,
                    'portfolio_value': portfolio_value
                }
            
            # Check if this trade would push us over the daily loss cap
            if (self.daily_pnl - potential_loss) / portfolio_value <= -self.max_daily_loss:
                return {
                    'name': self.name,
                    'allow_trade': False,
                    'risk_score': 1.0,
                    'reason': f"Trade would exceed daily loss cap. Potential loss: {potential_loss / portfolio_value * 100:.2f}% | Remaining capacity: {((self.daily_pnl / portfolio_value) + self.max_daily_loss) * 100:.2f}%",
                    'potential_loss': potential_loss,
                    'remaining_capacity': ((self.daily_pnl / portfolio_value) + self.max_daily_loss) * 100
                }
            
            # Trade is within daily limits
            return {
                'name': self.name,
                'allow_trade': True,
                'risk_score': (potential_loss / portfolio_value) / self.max_daily_loss,
                'reason': f"Daily loss cap: {self.daily_pnl / portfolio_value * 100:.2f}% used, potential loss: {potential_loss / portfolio_value * 100:.2f}%",
                'potential_loss': potential_loss,
                'daily_pnl': self.daily_pnl
            }
            
        except Exception as e:
            logger.error(f"Error evaluating daily loss cap rule: {str(e)}")
            return {
                'name': self.name,
                'allow_trade': False,
                'risk_score': 1.0,
                'reason': f"Error evaluating daily loss cap rule: {str(e)}"
            }

class DiversificationRule(RiskRule):
    """Rule for enforcing diversification of trades"""
    
    def __init__(self, name: str = "diversification", enabled: bool = True, 
                 max_position_per_symbol: float = 0.05, max_total_exposure: float = 0.2):
        super().__init__(name, enabled)
        self.max_position_per_symbol = max_position_per_symbol  # Max 5% per symbol
        self.max_total_exposure = max_total_exposure  # Max 20% total exposure
        self.positions = []  # Current positions
        
    def set_positions(self, positions: List[Position]):
        """Set current positions for diversification evaluation"""
        self.positions = positions
    
    def evaluate(self, trade_proposal: Dict, portfolio: Dict, positions: List[Position]) -> Dict:
        """Evaluate diversification requirements for the trade proposal"""
        if not self.enabled:
            return {
                'name': self.name,
                'allow_trade': True,
                'risk_score': 0.0,
                'reason': 'Diversification rule disabled'
            }
        
        try:
            symbol = trade_proposal.get('symbol')
            quantity = trade_proposal.get('quantity', 0)
            current_price = trade_proposal.get('price')
            portfolio_value = portfolio.get('total_value', 0)
            
            # Calculate current exposure for this symbol
            current_symbol_exposure = 0.0
            for position in positions:
                if position.symbol == symbol:
                    current_symbol_exposure += position.current_value(current_price)
            
            # Calculate new exposure if we take this trade
            new_symbol_exposure = current_symbol_exposure + (quantity * current_price)
            
            # Check symbol-specific diversification
            if new_symbol_exposure / portfolio_value > self.max_position_per_symbol:
                return {
                    'name': self.name,
                    'allow_trade': False,
                    'risk_score': 1.0,
                    'reason': f"Trade would exceed maximum position size for {symbol} ({new_symbol_exposure / portfolio_value * 100:.2f}% > {self.max_position_per_symbol * 100}%)",
                    'current_exposure': current_symbol_exposure,
                    'new_exposure': new_symbol_exposure
                }
            
            # Calculate current total exposure
            current_total_exposure = sum(p.current_value(current_price) for p in positions)
            new_total_exposure = current_total_exposure + (quantity * current_price)
            
            # Check total portfolio diversification
            if new_total_exposure / portfolio_value > self.max_total_exposure:
                return {
                    'name': self.name,
                    'allow_trade': False,
                    'risk_score': 1.0,
                    'reason': f"Trade would exceed maximum total exposure ({new_total_exposure / portfolio_value * 100:.2f}% > {self.max_total_exposure * 100}%)",
                    'current_total_exposure': current_total_exposure,
                    'new_total_exposure': new_total_exposure
                }
            
            # Trade meets diversification requirements
            return {
                'name': self.name,
                'allow_trade': True,
                'risk_score': (new_total_exposure / portfolio_value) / self.max_total_exposure,
                'reason': f"Diversification OK: {new_symbol_exposure / portfolio_value * 100:.2f}% for {symbol}, {new_total_exposure / portfolio_value * 100:.2f}% total exposure",
                'new_symbol_exposure': new_symbol_exposure
            }
            
        except Exception as e:
            logger.error(f"Error evaluating diversification rule: {str(e)}")
            return {
                'name': self.name,
                'allow_trade': False,
                'risk_score': 1.0,
                'reason': f"Error evaluating diversification rule: {str(e)}"
            }

class StopLossRule(RiskRule):
    """Rule for enforcing stop-loss requirements"""
    
    def __init__(self, name: str = "stop_loss", enabled: bool = True, 
                 min_stop_loss: float = 0.005, max_stop_loss: float = 0.05):
        super().__init__(name, enabled)
        self.min_stop_loss = min_stop_loss  # Minimum 0.5% stop-loss
        self.max_stop_loss = max_stop_loss  # Maximum 5% stop-loss
    
    def evaluate(self, trade_proposal: Dict, portfolio: Dict, positions: List[Position]) -> Dict:
        """Evaluate stop-loss requirements for the trade proposal"""
        if not self.enabled:
            return {
                'name': self.name,
                'allow_trade': True,
                'risk_score': 0.0,
                'reason': 'Stop-loss rule disabled'
            }
        
        try:
            signal = trade_proposal.get('signal')
            current_price = trade_proposal.get('price')
            stop_loss = trade_proposal.get('stop_loss', current_price * 0.99)  # Default 1% stop-loss
            
            # Calculate stop-loss distance
            if signal == 1:
                stop_loss_distance = current_price - stop_loss
            else:
                stop_loss_distance = stop_loss - current_price
            
            stop_loss_percentage = stop_loss_distance / current_price
            
            # Check if stop-loss is too tight
            if stop_loss_percentage < self.min_stop_loss:
                return {
                    'name': self.name,
                    'allow_trade': False,
                    'risk_score': 1.0,
                    'reason': f"Stop-loss too tight ({stop_loss_percentage * 100:.2f}% < {self.min_stop_loss * 100}%)",
                    'stop_loss': stop_loss,
                    'current_price': current_price
                }
            
            # Check if stop-loss is too wide
            if stop_loss_percentage > self.max_stop_loss:
                return {
                    'name': self.name,
                    'allow_trade': False,
                    'risk_score': 1.0,
                    'reason': f"Stop-loss too wide ({stop_loss_percentage * 100:.2f}% > {self.max_stop_loss * 100}%)",
                    'stop_loss': stop_loss,
                    'current_price': current_price
                }
            
            # Stop-loss is within acceptable range
            return {
                'name': self.name,
                'allow_trade': True,
                'risk_score': stop_loss_percentage / self.max_stop_loss,
                'reason': f"Stop-loss acceptable: {stop_loss_percentage * 100:.2f}%",
                'adjusted_stop_loss': stop_loss
            }
            
        except Exception as e:
            logger.error(f"Error evaluating stop-loss rule: {str(e)}")
            return {
                'name': self.name,
                'allow_trade': False,
                'risk_score': 1.0,
                'reason': f"Error evaluating stop-loss rule: {str(e)}"
            }

class VolatilityRule(RiskRule):
    """Rule for enforcing volatility-based risk limits"""
    
    def __init__(self, name: str = "volatility", enabled: bool = True, 
                 max_volatility: float = 0.03):
        super().__init__(name, enabled)
        self.max_volatility = max_volatility  # Max 3% ATR
        
    def evaluate(self, trade_proposal: Dict, portfolio: Dict, positions: List[Position]) -> Dict:
        """Evaluate volatility requirements for the trade proposal"""
        if not self.enabled:
            return {
                'name': self.name,
                'allow_trade': True,
                'risk_score': 0.0,
                'reason': 'Volatility rule disabled'
            }
        
        try:
            signal = trade_proposal.get('signal')
            current_price = trade_proposal.get('price')
            
            # Get ATR from technical indicators
            atr = trade_proposal.get('indicator_values', {}).get('atr', {}).get('atr', 0)
            
            # If no ATR data available, allow trade but with high risk score
            if atr == 0:
                return {
                    'name': self.name,
                    'allow_trade': True,
                    'risk_score': 0.5,
                    'reason': 'No ATR data available',
                    'atr': atr
                }
            
            # Calculate risk based on ATR
            risk_score = atr / self.max_volatility
            
            # Check if volatility is too high
            if risk_score > 1.0:
                return {
                    'name': self.name,
                    'allow_trade': False,
                    'risk_score': 1.0,
                    'reason': f"Volatility too high (ATR: {atr * 100:.2f}% > {self.max_volatility * 100}%)",
                    'atr': atr
                }
            
            # Volatility is acceptable
            return {
                'name': self.name,
                'allow_trade': True,
                'risk_score': risk_score,
                'reason': f"Volatility acceptable (ATR: {atr * 100:.2f}%)",
                'atr': atr
            }
            
        except Exception as e:
            logger.error(f"Error evaluating volatility rule: {str(e)}")
            return {
                'name': self.name,
                'allow_trade': False,
                'risk_score': 1.0,
                'reason': f"Error evaluating volatility rule: {str(e)}"
            }

class CircuitBreakerRule(RiskRule):
    """Rule for enforcing system-wide circuit breakers"""
    
    def __init__(self, name: str = "circuit_breaker", enabled: bool = True, 
                 max_api_errors: int = 5, max_data_discrepancies: int = 3):
        super().__init__(name, enabled)
        self.max_api_errors = max_api_errors
        self.max_data_discrepancies = max_data_discrepancies
        self.api_error_count = 0
        self.data_discrepancy_count = 0
        self.last_reset = datetime.now()
        
    def reset_counters(self):
        """Reset counters daily"""
        self.api_error_count = 0
        self.data_discrepancy_count = 0
        self.last_reset = datetime.now()
    
    def add_api_error(self):
        """Register an API error"""
        # Reset counters daily
        if (datetime.now() - self.last_reset).days >= 1:
            self.reset_counters()
            
        self.api_error_count += 1
    
    def add_data_discrepancy(self):
        """Register a data discrepancy"""
        # Reset counters daily
        if (datetime.now() - self.last_reset).days >= 1:
            self.reset_counters()
            
        self.data_discrepancy_count += 1
    
    def evaluate(self, trade_proposal: Dict, portfolio: Dict, positions: List[Position]) -> Dict:
        """Evaluate system health and apply circuit breakers"""
        if not self.enabled:
            return {
                'name': self.name,
                'allow_trade': True,
                'risk_score': 0.0,
                'reason': 'Circuit breaker rule disabled'
            }
        
        try:
            # Reset counters daily
            if (datetime.now() - self.last_reset).days >= 1:
                self.reset_counters()
            
            # Check API error count
            if self.api_error_count >= self.max_api_errors:
                return {
                    'name': self.name,
                    'allow_trade': False,
                    'risk_score': 1.0,
                    'reason': f"Too many API errors ({self.api_error_count} >= {self.max_api_errors})",
                    'api_errors': self.api_error_count
                }
            
            # Check data discrepancy count
            if self.data_discrepancy_count >= self.max_data_discrepancy_count:
                return {
                    'name': self.name,
                    'allow_trade': False,
                    'risk_score': 1.0,
                    'reason': f"Too many data discrepancies ({self.data_discrepancy_count} >= {self.max_data_discrepancy_count})",
                    'data_discrepancies': self.data_discrepancy_count
                }
            
            # Check if market data seems abnormal
            # This would typically be updated from the data agent
            if trade_proposal.get('data_quality', 1.0) < 0.7:
                return {
                    'name': self.name,
                    'allow_trade': False,
                    'risk_score': 1.0,
                    'reason': f"Market data quality low ({trade_proposal.get('data_quality', 0) * 100:.1f}% < 70%)",
                    'data_quality': trade_proposal.get('data_quality', 0)
                }
            
            # All system checks passed
            return {
                'name': self.name,
                'allow_trade': True,
                'risk_score': 0.0,
                'reason': "System health checks passed",
                'api_errors': self.api_error_count,
                'data_discrepancies': self.data_discrepancy_count
            }
            
        except Exception as e:
            logger.error(f"Error evaluating circuit breaker rule: {str(e)}")
            return {
                'name': self.name,
                'allow_trade': False,
                'risk_score': 1.0,
                'reason': f"Error evaluating circuit breaker rule: {str(e)}"
            }

class RiskManager:
    """
    Risk Management Agent
    The most critical component with ultimate veto power over all trades
    """
    
    def __init__(self, max_risk_per_trade: float = 0.005, 
                 daily_loss_cap: float = 0.03, 
                 stop_loss_levels: List[str] = ['fixed', 'adaptive', 'trailing']):
        self.max_risk_per_trade = max_risk_per_trade  # Max 0.5% risk per trade
        self.daily_loss_cap = daily_loss_cap  # Max 3% daily loss cap
        self.stop_loss_levels = stop_loss_levels
        
        # Initialize risk rules
        self.risk_rules = {
            'position_sizing': PositionSizingRule(max_risk_per_trade=self.max_risk_per_trade),
            'daily_loss_cap': DailyLossCapRule(max_daily_loss=self.daily_loss_cap),
            'diversification': DiversificationRule(),
            'stop_loss': StopLossRule(),
            'circuit_breaker': CircuitBreakerRule()
        }
        
        # Active rules (can be modified based on market conditions)
        self.active_rules = list(self.risk_rules.values())
        
        # Daily statistics
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.daily_win_count = 0
        self.daily_loss_count = 0
        self.daily_total_return = 0.0
        self.daily_sharpe_ratio = 0.0
        self.daily_max_drawdown = 0.0
        self.last_reset = datetime.now()
        
        # Portfolio statistics
        self.portfolio_value = 100000.0  # Default starting portfolio value
        self.portfolio_history = []
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.total_trades = 0
        self.total_return = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
    def evaluate_trade(self, trade_proposal: Dict, portfolio: Dict, positions: List[Position]) -> Dict:
        """
        Evaluate a trade proposal against all risk rules
        Returns a final decision to allow or block the trade
        
        Args:
            trade_proposal: Dictionary containing trade proposal details
            portfolio: Dictionary containing current portfolio information
            positions: List of current positions
            
        Returns:
            Dictionary with evaluation results
        """
        # Reset daily counters if needed
        if (datetime.now() - self.last_reset).days >= 1:
            self.reset_daily_stats()
        
        # Initialize portfolio value if not provided
        if portfolio.get('total_value') is None:
            portfolio['total_value'] = self.portfolio_value
            
        # Initialize return value
        final_decision = {
            'allow_trade': True,
            'risk_score': 0.0,
            'reason': 'Trade passed all risk checks',
            'rule_results': []
        }
        
        # Evaluate each risk rule
        for rule in self.active_rules:
            rule_result = rule.evaluate(trade_proposal, portfolio, positions)
            final_decision['rule_results'].append(rule_result)
            
            # Update trade proposal with rule-specific data
            if 'position_size' in rule_result:
                trade_proposal['quantity'] = rule_result['position_size']
            
            # If any rule blocks the trade, override the decision
            if not rule_result['allow_trade']:
                final_decision['allow_trade'] = False
                final_decision['risk_score'] = max(final_decision['risk_score'], rule_result.get('risk_score', 1.0))
                final_decision['reason'] = rule_result['reason']
        
        # Update trade proposal with final decision
        trade_proposal['risk_check'] = final_decision
        
        # If trade is allowed, add to daily statistics
        if final_decision['allow_trade']:
            self.daily_trades.append(trade_proposal)
            
        return final_decision
    
    def update_portfolio(self, portfolio_update: Dict):
        """Update portfolio information with new data"""
        # Update portfolio value
        if 'total_value' in portfolio_update:
            previous_value = self.portfolio_value
            self.portfolio_value = portfolio_update['total_value']
            
            # Update portfolio history
            self.portfolio_history.append({
                'time': datetime.now().isoformat(),
                'previous_value': previous_value,
                'new_value': self.portfolio_value,
                'change': self.portfolio_value - previous_value
            })
            
            # Update total P&L
            self.total_pnl = self.portfolio_value - 100000.0  # Assuming starting value of 100,000
            
            # Update portfolio value in portfolio_update
            portfolio_update['total_value'] = self.portfolio_value
            
        # Update other portfolio information
        # (implementation would include more detailed portfolio tracking)
        
        return portfolio_update
    
    def record_trade(self, trade: Dict, pnl: float = 0.0, is_win: bool = False):
        """Record a trade and update statistics"""
        # Update daily statistics
        self.daily_trades.append(trade)
        self.daily_pnl += pnl
        self.total_trades += 1
        
        # Update daily win/loss count
        if is_win:
            self.daily_win_count += 1
            self.win_rate = (self.daily_win_count / self.total_trades) * 100
        else:
            self.daily_loss_count += 1
            self.win_rate = (self.daily_win_count / max(self.total_trades, 1)) * 100
        
        # Update daily total return
        if self.daily_trades:
            start_value = self.daily_trades[0].get('portfolio_value', 100000.0)
            end_value = trade.get('portfolio_value', start_value + pnl)
            self.daily_total_return = (end_value - start_value) / start_value
            
        # Update total return
        start_value = 100000.0  # Assuming starting value
        self.total_return = (self.portfolio_value - start_value) / start_value
        
        # Update Sharpe ratio and drawdowns
        # (implementation would include more detailed statistics)
        
        return {
            'success': True,
            'trade_count': self.total_trades,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'daily_return': self.daily_total_return
        }
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.daily_win_count = 0
        self.daily_loss_count = 0
        self.daily_total_return = 0.0
        self.daily_sharpe_ratio = 0.0
        self.daily_max_drawdown = 0.0
        self.last_reset = datetime.now()
        
        # Reset circuit breaker counters
        for rule in self.active_rules:
            if isinstance(rule, CircuitBreakerRule):
                rule.reset_counters()
        
        return {
            'success': True,
            'message': 'Daily statistics reset'
        }
    
    def get_risk_metrics(self, portfolio: Dict, positions: List[Position]) -> Dict:
        """Get current risk metrics for the portfolio"""
        try:
            # Calculate total portfolio exposure
            total_exposure = sum(p.current_value(portfolio.get('price', 0)) for p in positions)
            
            # Calculate exposure percentage
            portfolio_value = portfolio.get('total_value', 100000.0)  # Default to starting value if not available
            exposure_percentage = total_exposure / portfolio_value
            
            # Calculate current drawdown
            if portfolio_value < self.portfolio_value:
                current_drawdown = (self.portfolio_value - portfolio_value) / self.portfolio_value
            else:
                current_drawdown = 0.0
                self.portfolio_value = portfolio_value  # Update high water mark
            
            # Get market volatility from data agent
            market_volatility = trade_proposal.get('indicator_values', {}).get('atr', {}).get('atr', 0)
            
            # Get system health from circuit breaker rule
            circuit_breaker_rule = next((r for r in self.active_rules if isinstance(r, CircuitBreakerRule)), None)
            system_health = 1.0 - (circuit_breaker_rule.api_error_count / circuit_breaker_rule.max_api_errors) if circuit_breaker_rule else 1.0
            
            return {
                'success': True,
                'exposure_percentage': exposure_percentage,
                'current_drawdown': current_drawdown,
                'market_volatility': market_volatility,
                'system_health': system_health,
                'daily_trades': len(self.daily_trades),
                'daily_pnl': self.daily_pnl,
                'win_rate': self.win_rate,
                'sharpe_ratio': self.daily_sharpe_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_trading_capacity(self, portfolio: Dict, positions: List[Position]) -> Dict:
        """Get the current trading capacity based on risk constraints"""
        try:
            portfolio_value = portfolio.get('total_value', 100000.0)  # Default to starting value if not available
            
            # Get position sizing capacity
            position_sizing_rule = next((r for r in self.active_rules if isinstance(r, PositionSizingRule)), None)
            if position_sizing_rule:
                position_sizing = position_sizing_rule.max_risk_per_trade
            else:
                position_sizing = self.max_risk_per_trade
            
            # Get diversification capacity
            diversification_rule = next((r for r in self.active_rules if isinstance(r, DiversificationRule)), None)
            if diversification_rule:
                diversification = diversification_rule.max_position_per_symbol
            else:
                diversification = 0.05  # Default 5% per symbol
            
            # Get daily loss capacity
            daily_loss_rule = next((r for r in self.active_rules if isinstance(r, DailyLossCapRule)), None)
            if daily_loss_rule:
                daily_loss_capacity = 1.0 - (daily_loss_rule.daily_pnl / portfolio_value) / daily_loss_rule.max_daily_loss
            else:
                daily_loss_capacity = 1.0
            
            # Calculate overall trading capacity
            overall_capacity = min(position_sizing, diversification, daily_loss_capacity)
            
            return {
                'success': True,
                'position_sizing_capacity': position_sizing,
                'diversification_capacity': diversification,
                'daily_loss_capacity': daily_loss_capacity,
                'overall_capacity': overall_capacity,
                'reason': f"Position sizing: {position_sizing * 100:.2f}% | Diversification: {diversification * 100:.2f}% | Daily loss: {daily_loss_capacity * 100:.2f}%"
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading capacity: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_api_error_count(self, count: int = 1):
        """Update API error count for circuit breaker"""
        circuit_breaker_rule = next((r for r in self.active_rules if isinstance(r, CircuitBreakerRule)), None)
        if circuit_breaker_rule:
            circuit_breaker_rule.add_api_error(count)
    
    def update_data_discrepancy_count(self, count: int = 1):
        """Update data discrepancy count for circuit breaker"""
        circuit_breaker_rule = next((r for r in self.active_rules if isinstance(r, CircuitBreakerRule)), None)
        if circuit_breaker_rule:
            circuit_breaker_rule.add_data_discrepancy(count)
    
    async def start(self):
        """Start the risk manager"""
        logger.info("Starting risk manager...")
        # Implementation for starting the agent
        # This would typically involve:
        # - Connecting to portfolio data
        # - Initializing risk rules
        # - Starting monitoring loop
        
    async def emergency_stop(self):
        """Emergency stop for the risk manager"""
        logger.info("Initiating emergency stop for risk manager...")
        # Implementation for emergency stop procedures
        # This would include:
        # - Closing all positions
        # - Halting all trading
        # - Notifying other agents
        pass