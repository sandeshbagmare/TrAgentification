"""
Unit tests for the Risk Management Agent
"""
import pytest
import asyncio
import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass
from datetime import datetime, timedelta
from agents.risk_manager import RiskManager, Position, PositionSizingRule, DailyLossCapRule, DiversificationRule, StopLossRule, VolatilityRule, CircuitBreakerRule

# Sample data for testing
@dataclass
class TradeProposal:
    """Data class for creating trade proposals for testing"""
    symbol: str = 'RELIANCE'
    price: float = 2500.0
    quantity: float = 10.0
    signal: int = 1  # 1 for buy, -1 for sell, 0 for no signal
    stop_loss: float = 2400.0
    take_profit: float = 2600.0
    timestamp: datetime = datetime.now()

# Sample portfolio data
def sample_portfolio(portfolio_value: float = 100000.0) -> dict:
    """Create a sample portfolio dictionary"""
    return {
        'total_value': portfolio_value,
        'price': 2500.0  # Current market price
    }

# Sample positions
def sample_positions(portfolio_value: float = 100000.0) -> List[Position]:
    """Create sample positions for testing"""
    return [
        Position(
            symbol='INFY',
            quantity=5.0,
            entry_price=2450.0,
            entry_time=datetime.now() - timedelta(minutes=10),
            stop_loss=2400.0,
            take_profit=2500.0,
            risk_percentage=0.005,
            volatility=0.015
        ),
        Position(
            symbol='TATA',
            quantity=8.0,
            entry_price=120.0,
            entry_time=datetime.now() - timedelta(minutes=5),
            stop_loss=115.0,
            take_profit=125.0,
            risk_percentage=0.004,
            volatility=0.012
        )
    ]

class TestPosition:
    """Test suite for the Position data class"""
    
    def test_position_value(self):
        """Test current value calculation for a position"""
        position = Position(
            symbol='RELIANCE',
            quantity=10.0,
            entry_price=2500.0,
            entry_time=datetime.now(),
            stop_loss=2400.0,
            take_profit=2600.0,
            risk_percentage=0.005,
            volatility=0.015
        )
        
        # Test with price above entry (profit)
        assert position.current_value(2550.0) == 25500.0
        # Test with price below entry (loss)
        assert position.current_value(2450.0) == 24500.0
        
    def test_unrealized_pnl(self):
        """Test unrealized profit/loss calculation"""
        position = Position(
            symbol='RELIANCE',
            quantity=10.0,
            entry_price=2500.0,
            entry_time=datetime.now(),
            stop_loss=2400.0,
            take_profit=2600.0,
            risk_percentage=0.005,
            volatility=0.015
        )
        
        # Test with price above entry (profit)
        assert position.unrealized_pnl(2550.0) == 500.0
        # Test with price below entry (loss)
        assert position.unrealized_pnl(2450.0) == -500.0
        
    def test_risk_amount(self):
        """Test risk amount calculation"""
        position = Position(
            symbol='RELIANCE',
            quantity=10.0,
            entry_price=2500.0,
            entry_time=datetime.now(),
            stop_loss=2400.0,
            take_profit=2600.0,
            risk_percentage=0.005,
            volatility=0.015
        )
        
        # Test with buy signal
        assert position.risk_amount(2450.0) == 500.0  # (2500 - 2450) * 10
        # Test with sell signal
        sell_position = Position(
            symbol='RELIANCE',
            quantity=10.0,
            entry_price=2500.0,
            entry_time=datetime.now(),
            stop_loss=2600.0,  # For sell, stop loss is above entry
            take_profit=2400.0,  # For sell, take profit is below entry
            risk_percentage=0.005,
            volatility=0.015
        )
        
        assert sell_position.risk_amount(2600.0) == 1000.0  # (2600 - 2500) * 10
        

class TestRiskRules:
    """Test suite for individual risk rules"""
    
    def test_position_sizing_rule(self):
        """Test position sizing based on risk percentage"""
        # Create test data
        trade_proposal = TradeProposal(signal=1, price=2500.0)
        portfolio = sample_portfolio(100000.0)
        positions = []
        
        # Test with max risk per trade = 0.005 (0.5% of portfolio)
        rule = PositionSizingRule(max_risk_per_trade=0.005)
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        
        # Calculate expected position size
        expected_size = (portfolio['total_value'] * 0.005) / (trade_proposal.price - trade_proposal.stop_loss)
        
        assert result['allow_trade']
        assert result['position_size'] == expected_size
        assert result['reason']
        
        # Test with invalid stop loss (equal to price)
        trade_proposal.stop_loss = 2500.0
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert not result['allow_trade']
        assert 'error' in result or 'reason' in result
        
    def test_daily_loss_cap_rule(self):
        """Test daily loss cap rule"""
        # Create test data
        trade_proposal = TradeProposal(signal=1, price=2500.0)
        portfolio = sample_portfolio(100000.0)
        positions = []
        
        # Create rule with 3% daily loss cap
        rule = DailyLossCapRule(max_daily_loss=0.03)
        
        # Test with no daily P&L
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert result['allow_trade']
        assert result['daily_pnl'] == 0.0
        
        # Test with daily P&L already at -2.5%
        rule.daily_pnl = -2500.0  # -2.5% of 100,000
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert result['allow_trade']
        assert result['daily_pnl'] == -2500.0
        
        # Test with daily P&L already at -3% (at cap)
        rule.daily_pnl = -3000.0  # -3% of 100,000
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert not result['allow_trade']
        assert result['risk_score'] == 1.0
        
        # Test with daily P&L already at -3.5% (over cap)
        rule.daily_pnl = -3500.0  # -3.5% of 100,000
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert not result['allow_trade']
        assert result['risk_score'] == 1.0
        
    def test_diversification_rule(self):
        """Test diversification rules for position size and total exposure"""
        # Create test data
        trade_proposal = TradeProposal(signal=1, price=2500.0, quantity=10.0)
        portfolio = sample_portfolio(100000.0)
        
        # Create some existing positions
        existing_positions = [
            Position(
                symbol='RELIANCE',
                quantity=5.0,
                entry_price=2500.0,
                entry_time=datetime.now(),
                stop_loss=2400.0,
                take_profit=2600.0,
                risk_percentage=0.005,
                volatility=0.015
            ),
            Position(
                symbol='INFY',
                quantity=5.0,
                entry_price=2450.0,
                entry_time=datetime.now(),
                stop_loss=2400.0,
                take_profit=2500.0,
                risk_percentage=0.005,
                volatility=0.015
            )
        ]
        
        # Test with max position per symbol = 0.05 (5%)
        rule = DiversificationRule(max_position_per_symbol=0.05, max_total_exposure=0.2)
        
        # Test with no existing positions
        result = rule.evaluate(trade_proposal.__dict__, portfolio, [])
        assert result['allow_trade']
        assert result['new_symbol_exposure'] / portfolio['total_value'] <= 0.05
        
        # Test with existing positions
        result = rule.evaluate(trade_proposal.__dict__, portfolio, existing_positions)
        assert result['allow_trade']
        assert result['new_total_exposure'] / portfolio['total_value'] <= 0.2
        
        # Test with existing position already at 5% (should block)
        existing_positions = [
            Position(
                symbol='RELIANCE',
                quantity=20.0,  # 20 units * 2500 = 50,000 (5% of 100,000)
                entry_price=2500.0,
                entry_time=datetime.now(),
                stop_loss=2400.0,
                take_profit=2600.0,
                risk_percentage=0.005,
                volatility=0.015
            )
        ]
        
        result = rule.evaluate(trade_proposal.__dict__, portfolio, existing_positions)
        assert not result['allow_trade']
        assert result['risk_score'] == 1.0
        
        # Test total exposure limit
        existing_positions = [
            Position(
                symbol='INFY',
                quantity=50.0,  # 50 * 2450 = 122500
                entry_price=2450.0,
                entry_time=datetime.now(),
                stop_loss=2400.0,
                take_profit=2500.0,
                risk_percentage=0.005,
                volatility=0.015
            ),
            Position(
                symbol='TATA',
                quantity=30.0,  # 30 * 120 = 3600
                entry_price=120.0,
                entry_time=datetime.now(),
                stop_loss=115.0,
                take_profit=125.0,
                risk_percentage=0.004,
                volatility=0.012
            )
        ]
        
        result = rule.evaluate(trade_proposal.__dict__, portfolio, existing_positions)
        assert not result['allow_trade']
        assert result['risk_score'] == 1.0
        
    def test_stop_loss_rule(self):
        """Test stop-loss requirements"""
        # Create test data
        trade_proposal = TradeProposal(signal=1, price=2500.0, stop_loss=2450.0)
        portfolio = sample_portfolio(100000.0)
        positions = []
        
        # Test with valid stop-loss
        rule = StopLossRule()
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert result['allow_trade']
        assert result['risk_score'] < 1.0
        
        # Test with stop-loss too tight
        trade_proposal.stop_loss = 2499.0  # Only 1 rupee stop-loss
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert not result['allow_trade']
        assert result['risk_score'] == 1.0
        
        # Test with stop-loss too wide
        trade_proposal.stop_loss = 2300.0  # 8% stop-loss
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert not result['allow_trade']
        assert result['risk_score'] == 1.0
        
    def test_volatility_rule(self):
        """Test volatility-based risk limits"""
        # Create test data
        trade_proposal = TradeProposal(signal=1, price=2500.0)
        trade_proposal['indicator_values'] = {'atr': {'atr': 0.015}}  # 1.5% ATR
        portfolio = sample_portfolio(100000.0)
        positions = []
        
        # Test with normal volatility
        rule = VolatilityRule()
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert result['allow_trade']
        assert result['atr'] == 0.015
        
        # Test with high volatility
        trade_proposal['indicator_values']['atr']['atr'] = 0.04  # 4% ATR
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert not result['allow_trade']
        assert result['risk_score'] == 1.0
        
        # Test with no ATR data
        trade_proposal.pop('indicator_values')
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert result['allow_trade']
        assert result['risk_score'] == 0.5  # Lower risk score due to missing data
        
    def test_circuit_breaker_rule(self):
        """Test system-wide circuit breakers"""
        # Create test data
        trade_proposal = TradeProposal(signal=1, price=2500.0)
        portfolio = sample_portfolio(100000.0)
        positions = []
        
        # Test with no issues
        rule = CircuitBreakerRule()
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert result['allow_trade']
        assert result['risk_score'] == 0.0
        
        # Test with API errors
        rule.add_api_error(3)  # Add 3 API errors
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert not result['allow_trade']
        assert result['risk_score'] == 1.0
        
        # Test with data discrepancies
        rule.add_data_discrepancy(2)  # Add 2 data discrepancies
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert not result['allow_trade']
        assert result['risk_score'] == 1.0
        
        # Test with low data quality
        trade_proposal['data_quality'] = 0.6  # 60% data quality
        result = rule.evaluate(trade_proposal.__dict__, portfolio, positions)
        assert not result['allow_trade']
        assert result['risk_score'] == 1.0
        

class TestRiskManager:
    """Test suite for the RiskManager class"""
    
    def test_evaluate_trade(self):
        """Test trade evaluation with all risk rules"""
        # Create test data
        trade_proposal = TradeProposal(signal=1, price=2500.0, stop_loss=2450.0)
        portfolio = sample_portfolio(100000.0)
        
        # Create some existing positions
        positions = [
            Position(
                symbol='INFY',
                quantity=5.0,
                entry_price=2450.0,
                entry_time=datetime.now(),
                stop_loss=2400.0,
                take_profit=2500.0,
                risk_percentage=0.005,
                volatility=0.015
            )
        ]
        
        # Create risk manager
        risk_manager = RiskManager()
        
        # Test with valid trade
        result = risk_manager.evaluate_trade(trade_proposal.__dict__, portfolio, positions)
        assert result['allow_trade']
        assert result['risk_score'] < 1.0
        
        # Test with high volatility
        trade_proposal['indicator_values'] = {'atr': {'atr': 0.04}}  # 4% ATR
        result = risk_manager.evaluate_trade(trade_proposal.__dict__, portfolio, positions)
        assert not result['allow_trade']
        assert result['risk_score'] == 1.0
        
        # Test with market closed
        trade_proposal['indicator_values'] = {'atr': {'atr': 0.015}}  # Reset to normal
        trade_proposal['signal'] = 1
        
        # Create time-based rule that blocks trading
        time_rule = TimeBasedRule()
        
        # Mock current time to be after market close
        original_now = datetime.now
        datetime.now = lambda: datetime(2023, 6, 1, 16, 0)  # 16:00 IST (10:30 UTC) - market closed
        
        # Add time-based rule to risk manager
        risk_manager.rules['time_based'] = time_rule
        result = risk_manager.evaluate_trade(trade_proposal.__dict__, portfolio, positions)
        assert not result['allow_trade']
        assert result['risk_score'] == 1.0
        
        # Restore datetime
        datetime.now = original_now
        
    def test_update_portfolio(self):
        """Test portfolio update functionality"""
        risk_manager = RiskManager()
        
        # Test portfolio update
        portfolio_update = {'total_value': 105000.0}  # Portfolio grew to 105,000
        result = risk_manager.update_portfolio(portfolio_update)
        
        assert result['total_value'] == 105000.0
        assert result['change'] == 5000.0
        
    def test_record_trade(self):
        """Test trade recording and statistics update"""
        risk_manager = RiskManager()
        
        # Create a trade proposal
        trade_proposal = TradeProposal(signal=1, price=2500.0, quantity=10.0)
        
        # Create a portfolio update
        portfolio_update = {'total_value': 100000.0}
        
        # Record a winning trade
        result = risk_manager.record_trade(trade_proposal.__dict__, pnl=1000.0, is_win=True)
        
        assert result['trade_count'] == 1
        assert result['win_rate'] == 100.0
        
        # Record a losing trade
        trade_proposal.signal = -1
        result = risk_manager.record_trade(trade_proposal.__dict__, pnl=-500.0, is_win=False)
        
        assert result['trade_count'] == 2
        assert result['win_rate'] == 50.0
        
    def test_get_risk_metrics(self):
        """Test risk metrics calculation"""
        risk_manager = RiskManager()
        
        # Create test data
        trade_proposal = TradeProposal(signal=1, price=2500.0, quantity=10.0)
        trade_proposal['indicator_values'] = {'atr': {'atr': 0.015}}  # 1.5% ATR
        portfolio = sample_portfolio(100000.0)
        
        # Test risk metrics calculation
        result = risk_manager.get_risk_metrics(portfolio, [])
        
        assert result['success']
        assert 'exposure_percentage' in result
        assert 'current_drawdown' in result
        assert 'market_volatility' in result
        assert 'system_health' in result
        
    def test_get_trading_capacity(self):
        """Test trading capacity calculation based on risk constraints"""
        risk_manager = RiskManager()
        
        # Create test data
        trade_proposal = TradeProposal(signal=1, price=2500.0, quantity=10.0)
        portfolio = sample_portfolio(100000.0)
        
        # Test trading capacity calculation
        result = risk_manager.get_trading_capacity(portfolio, [])
        
        assert result['success']
        assert 'position_sizing_capacity' in result
        assert 'diversification_capacity' in result
        assert 'daily_loss_capacity' in result
        assert 'overall_capacity' in result
        
        # Test that overall capacity is the minimum of all capacities
        assert result['overall_capacity'] == min(
            result['position_sizing_capacity'],
            result['diversification_capacity'],
            result['daily_loss_capacity']
        )
        
    @pytest.mark.asyncio
    async def test_start(self):
        """Test starting the risk manager"""
        risk_manager = RiskManager()
        result = await risk_manager.start()
        assert result['success']
        
    @pytest.mark.asyncio
    async def test_emergency_stop(self):
        """Test emergency stop functionality"""
        risk_manager = RiskManager()
        result = await risk_manager.emergency_stop()
        assert result['success']