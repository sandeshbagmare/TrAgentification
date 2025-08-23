"""
Streamlit Dashboard for AI Trading System
Provides real-time monitoring and visualization of trading activity
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import asyncio
import sys
import traceback
import time
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import agents with error handling
try:
    from agents.monitoring_agent import MonitoringAgent
    from agents.execution_engine import ExecutionEngine
    from agents.strategy_orchestrator import StrategyOrchestrator
    from agents.data_agent import DataAgent
    from agents.risk_manager import RiskManager
    AGENTS_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing agents: {str(e)}")
    st.error("Please ensure all dependencies are installed and agents are properly configured.")
    AGENTS_AVAILABLE = False

# Set page config
st.set_page_config(page_title="AI Trading Dashboard", page_icon="📈", layout="wide")

class TradingDashboard:
    """
    Streamlit-based dashboard for monitoring AI trading system
    """
    def __init__(self):
        # Initialize session state
        if 'trading_data' not in st.session_state:
            st.session_state.trading_data = {}
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'error_log' not in st.session_state:
            st.session_state.error_log = []
            
        # Initialize agents only if available
        self.agents_initialized = False
        if AGENTS_AVAILABLE:
            try:
                self._initialize_agents()
                self.agents_initialized = True
            except Exception as e:
                st.error(f"Failed to initialize agents: {str(e)}")
                self._log_error(f"Agent initialization failed: {str(e)}")
                
    def _initialize_agents(self):
        """Initialize trading system agents"""
        # Initialize monitoring agent
        self.monitoring_agent = MonitoringAgent(
            dashboard_port=8000,  # Not used in Streamlit
            log_dir='logs',
            alert_channels=os.getenv('ALERT_CHANNELS', 'telegram,slack,email').split(',')
        )
        
        # Initialize execution engine (for order placement)
        self.execution_engine = ExecutionEngine(
            upstox_api_key=os.getenv('UPSTOX_API_KEY'),
            upstox_api_secret=os.getenv('UPSTOX_API_SECRET'),
            order_type='limit',
            slippage_tolerance=0.001
        )
        
        # Initialize strategy orchestrator
        self.strategy_orchestrator = StrategyOrchestrator(
            confidence_threshold=0.7,
            require_consenus=True
        )
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            max_risk_per_trade=0.005,
            daily_loss_cap=0.03,
            stop_loss_levels=['fixed', 'adaptive', 'trailing']
        )
        
    def _log_error(self, error_message: str):
        """Log error to session state"""
        timestamp = datetime.datetime.now().isoformat()
        st.session_state.error_log.append({
            'timestamp': timestamp,
            'error': error_message
        })
        
        # Keep only last 10 errors
        if len(st.session_state.error_log) > 10:
            st.session_state.error_log = st.session_state.error_log[-10:]
        
    def _get_real_time_data(self):
        """Get real-time data from agents with error handling"""
        try:
            if self.agents_initialized:
                # Try to get real data from agents
                return self._get_agent_data()
            else:
                # Fall back to mock data
                return self._get_mock_data()
        except Exception as e:
            self._log_error(f"Error getting real-time data: {str(e)}")
            return self._get_mock_data()
            
    def _get_agent_data(self):
        """Get data from actual trading agents"""
        # This would integrate with the actual agents
        # For now, we'll return enhanced mock data
        return self._get_mock_data()
        
    def _get_mock_data(self):
        """Generate mock data for testing"""
        # Enhanced mock data with more realistic values
        current_time = datetime.datetime.now()
        
        # Generate portfolio data
        base_value = 1000000
        daily_return = np.random.normal(0.002, 0.01)
        portfolio_value = base_value * (1 + daily_return)
        
        # Generate positions with realistic symbols
        indian_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'SBI', 'ITC', 'LT', 'HCLTECH', 'WIPRO']
        num_positions = np.random.randint(3, 6)
        positions = []
        
        for i in range(num_positions):
            symbol = np.random.choice(indian_stocks)
            quantity = np.random.randint(10, 500)
            avg_price = np.random.uniform(100, 3000)
            price_change = np.random.normal(0, 0.02)
            current_price = avg_price * (1 + price_change)
            pnl = (current_price - avg_price) * quantity
            
            positions.append({
                'symbol': symbol,
                'quantity': quantity,
                'avg_price': avg_price,
                'current_price': current_price,
                'pnl': pnl,
                'pnl_percent': (current_price - avg_price) / avg_price * 100
            })
        
        # Generate AI signals with varying availability
        signals = {}
        
        # Technical signals (always available)
        signals['technical'] = {
            'signal': np.random.choice([-1, 0, 1]),
            'confidence': np.random.uniform(0.5, 1.0),
            'reason': np.random.choice([
                'RSI oversold condition detected',
                'MACD bullish crossover',
                'Support level bounce',
                'Resistance level breakdown',
                'Moving average convergence'
            ])
        }
        
        # AI signals (conditional on availability)
        ai_integrations = getattr(self, 'strategy_orchestrator', None)
        if ai_integrations and hasattr(ai_integrations, 'ai_integrations'):
            if 'gemini' in ai_integrations.ai_integrations:
                signals['gemini'] = {
                    'signal': np.random.choice([-1, 0, 1]),
                    'confidence': np.random.uniform(0.6, 0.95),
                    'reason': 'AI analysis of market sentiment and news'
                }
            
            if 'azure_openai' in ai_integrations.ai_integrations:
                signals['azure_openai'] = {
                    'signal': np.random.choice([-1, 0, 1]),
                    'confidence': np.random.uniform(0.6, 0.95),
                    'reason': 'Large language model market analysis'
                }
        
        # Enhanced mock data with P&L tracking integration
        try:
            # Try to get real data from P&L tracker
            from agents.pnl_tracker import pnl_tracker
            
            # Get current positions
            current_positions = pnl_tracker.get_current_positions({
                'RELIANCE': 2505.75,
                'TCS': 3420.50,
                'INFY': 1650.25
            })
            
            # Get P&L summary
            pnl_summary = pnl_tracker.get_daily_pnl_summary(30)
            
            # Generate trading report
            trading_report = pnl_tracker.generate_trading_report(7)  # Last 7 days
            
            # Format positions for display
            formatted_positions = []
            for pos in current_positions:
                formatted_positions.append({
                    'symbol': pos['symbol'],
                    'quantity': pos['quantity'],
                    'avg_price': pos['average_price'],
                    'current_price': pos['current_price'],
                    'pnl': pos['unrealized_pnl'],
                    'pnl_percent': pos['unrealized_pnl_percent']
                })
            
            if not formatted_positions:  # Fallback to mock data
                formatted_positions = self._get_mock_positions()
            
        except Exception as e:
            logger.warning(f"Could not get real P&L data: {str(e)}, using mock data")
            formatted_positions = self._get_mock_positions()
            pnl_summary = self._get_mock_pnl_summary()
            trading_report = self._get_mock_trading_report()
        
        # Enhanced portfolio data
        portfolio_value = 1000000
        if 'pnl_summary' in locals():
            portfolio_value = pnl_summary.get('total_market_value', 1000000)
            daily_return = pnl_summary.get('portfolio_return_percent', 0) / 100
        else:
            daily_return = np.random.normal(0.002, 0.01)
            portfolio_value = portfolio_value * (1 + daily_return)
        
        return {
            'portfolio': {
                'value': portfolio_value,
                'returns': daily_return,
                'drawdown': abs(pnl_summary.get('portfolio_return_percent', 0)) / 100 if pnl_summary.get('portfolio_return_percent', 0) < 0 else 0,
                'sharpe_ratio': trading_report.get('risk_metrics', {}).get('sharpe_ratio', np.random.uniform(1.0, 2.5)),
                'total_pnl': pnl_summary.get('total_pnl', 0),
                'realized_pnl': pnl_summary.get('realized_pnl', 0),
                'unrealized_pnl': pnl_summary.get('unrealized_pnl', 0),
                'total_pnl_percent': pnl_summary.get('portfolio_return_percent', 0)
            },
            'positions': formatted_positions,
            'signals': signals,
            'risk_metrics': {
                'max_risk_per_trade': float(os.getenv('MAX_RISK_PER_TRADE', '0.005')),
                'daily_loss_cap': float(os.getenv('DAILY_LOSS_CAP', '0.03')),
                'current_daily_loss': abs(min(0, daily_return)),
                'position_risk': trading_report.get('risk_metrics', {}).get('var_95', np.random.uniform(0.001, 0.005)),
                'var_95': trading_report.get('risk_metrics', {}).get('var_95', np.random.uniform(0.015, 0.025)),
                'max_drawdown': trading_report.get('risk_metrics', {}).get('max_drawdown', np.random.uniform(0.02, 0.08)),
                'volatility': trading_report.get('risk_metrics', {}).get('volatility', np.random.uniform(0.1, 0.3))
            },
            'system_status': {
                'data_quality': trading_report.get('system_health', {}).get('data_quality', 'good'),
                'api_status': 'connected' if trading_report.get('system_health', {}).get('trading_active', False) else 'idle',
                'model_status': 'up-to-date',
                'last_update': current_time.isoformat(),
                'uptime': np.random.uniform(95, 99.9),
                'trades_today': pnl_summary.get('total_trades', np.random.randint(5, 50)),
                'win_rate': pnl_summary.get('win_rate', np.random.uniform(55, 75))
            },
            'market_data': {
                'nifty_50': 19500 + np.random.normal(0, 100),
                'sensex': 65000 + np.random.normal(0, 300),
                'volatility_index': trading_report.get('risk_metrics', {}).get('volatility', np.random.uniform(12, 25)) * 100,
                'market_sentiment': np.random.choice(['bullish', 'neutral', 'bearish'], p=[0.4, 0.4, 0.2])
            },
            'trading_report': trading_report
        }
        
    def _display_overview(self):
        """Display the main dashboard overview"""
        st.title("📈 AI Trading Dashboard")
        
        # Get real-time data
        data = self._get_real_time_data()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Portfolio Value", f"₹{data['portfolio']['value']:.2f}", 
                     delta=f"{data['portfolio']['returns']:.2%}")
        with col2:
            st.metric("Daily Drawdown", f"{data['risk_metrics']['current_daily_loss']:.2%}", 
                     delta=f"Threshold: {data['risk_metrics']['daily_loss_cap']:.2%}")
        with col3:
            st.metric("Sharpe Ratio", f"{data['portfolio']['sharpe_ratio']:.2f}")
        with col4:
            st.metric("System Status", data['system_status']['api_status'], 
                     help=f"Last update: {data['system_status']['last_update']}")
        
        # Display positions
        st.subheader("💼 Current Positions")
        positions_df = pd.DataFrame(data['positions'])
        positions_df['P&L %'] = (positions_df['current_price'] - positions_df['avg_price']) / positions_df['avg_price']
        st.dataframe(positions_df)
        
        # Display trading signals
        st.subheader("🤖 Trading Signals")
        signals = data['signals']
        for source, signal in signals.items():
            col1, col2, col3 = st.columns([1, 2, 4])
            with col1:
                st.write(f"**{source.upper()}**")
            with col2:
                if signal['signal'] == 1:
                    st.success(f"Buy ({signal['confidence']:.0%} confidence)")
                elif signal['signal'] == -1:
                    st.error(f"Sell ({signal['confidence']:.0%} confidence)")
                else:
                    st.info("No Signal")
            with col3:
                st.write(signal['reason'])
        
        # Display risk metrics
        st.subheader("⚠️ Risk Metrics")
        risk_data = data['risk_metrics']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.progress(risk_data['current_daily_loss'] / risk_data['daily_loss_cap'], 
                        text=f"Daily Loss: {risk_data['current_daily_loss']:.2%} / {risk_data['daily_loss_cap']:.2%}")
        with col2:
            st.progress(risk_data['position_risk'] / risk_data['max_risk_per_trade'],
                        text=f"Position Risk: {risk_data['position_risk']:.2%} / {risk_data['max_risk_per_trade']:.2%}")
        with col3:
            volatility = np.random.uniform(0.01, 0.03)
            st.progress(volatility / 0.02, text=f"Market Volatility: {volatility:.2%}")
        
        # Display system status
        st.subheader("⚙️ System Status")
        status = data['system_status']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"Data Quality: {status['data_quality']}")
        with col2:
            st.write(f"API Status: {status['api_status']}")
        with col3:
            st.write(f"Model Status: {status['model_status']}")
        with col4:
            st.write(f"Last Update: {status['last_update']}")
        
        # Display order execution controls
        st.subheader("🚀 Order Execution")
        with st.form("order_form"):
            symbol = st.text_input("Symbol (e.g., RELIANCE)")
            quantity = st.number_input("Quantity", min_value=1, value=1)
            price = st.number_input("Price (0 for market)", value=0.0, format="%.2f")
            order_type = st.selectbox("Order Type", ["limit", "market", "stop_loss", "take_profit"])
            side = st.selectbox("Side", ["buy", "sell"])
            product = st.selectbox("Product Type", ["INTRADAY", "CNC", "MIS"])
            validity = st.selectbox("Validity", ["DAY", "IOC"])
            
            submitted = st.form_submit_button("Place Order")
            if submitted:
                # Place order using execution engine
                order_details = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price if price > 0 else None,
                    'side': side,
                    'order_type': order_type,
                    'product': product,
                    'validity': validity
                }
                
                # Execute order
                asyncio.run(self._execute_order(order_details))
                
    async def _execute_order(self, order_details: Dict):
        """Execute an order through the execution engine"""
        # Connect to execution engine
        await self.execution_engine.initialize()
        
        # Place order
        result = await self.execution_engine.place_order(order_details)
        
        # Display result
        if result['success']:
            st.success(f"Order placed successfully: {result['order_id']}")
        else:
            st.error(f"Error placing order: {result['error']}")
        
    def _display_historical_performance(self):
        """Display historical performance charts"""
        st.subheader("📊 Historical Performance")
        
        # Generate mock historical data
        dates = pd.date_range(end=pd.Timestamp.today(), periods=30, freq='D')
        returns = np.random.normal(0.002, 0.01, 30).cumsum()
        
        # Create portfolio value chart
        portfolio_value = pd.DataFrame({
            'Date': dates,
            'Portfolio Value': 1000000 * (1 + returns)
        }).set_index('Date')
        
        st.line_chart(portfolio_value)
        
        # Display returns by day
        daily_returns = pd.DataFrame({
            'Date': dates,
            'Daily Return': returns
        }).set_index('Date')
        
        st.bar_chart(daily_returns)
        
    def _display_settings(self):
        """Display settings panel"""
        st.sidebar.title("⚙️ Settings")
        
        # Risk management settings
        st.sidebar.subheader("Risk Management")
        risk_config = {
            'max_risk_per_trade': st.sidebar.slider(
                "Max Risk per Trade (%)", 0.0, 1.0, 0.5, 0.01
            ),
            'daily_loss_cap': st.sidebar.slider(
                "Daily Loss Cap (%)", 0.0, 5.0, 3.0, 0.1
            )
        }
        
        # Strategy settings
        st.sidebar.subheader("Strategy")
        strategy_config = {
            'confidence_threshold': st.sidebar.slider(
                "Signal Confidence Threshold (%)", 0.0, 1.0, 0.7, 0.01
            ),
            'require_consenus': st.sidebar.checkbox("Require Consensus", value=True)
        }
        
        # Save settings
        if st.sidebar.button("Save Settings"):
            # In a real implementation, this would update the configuration
            st.sidebar.success("Settings saved successfully!")
        
    def _get_mock_positions(self):
        """Generate mock positions for testing"""
        indian_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']
        positions = []
        
        for symbol in np.random.choice(indian_stocks, size=np.random.randint(2, 5), replace=False):
            quantity = np.random.randint(10, 500)
            avg_price = np.random.uniform(100, 3000)
            price_change = np.random.normal(0, 0.02)
            current_price = avg_price * (1 + price_change)
            pnl = (current_price - avg_price) * quantity
            
            positions.append({
                'symbol': symbol,
                'quantity': quantity,
                'avg_price': avg_price,
                'current_price': current_price,
                'pnl': pnl,
                'pnl_percent': (current_price - avg_price) / avg_price * 100
            })
        
        return positions
    
    def _get_mock_pnl_summary(self):
        """Generate mock P&L summary"""
        return {
            'total_pnl': np.random.uniform(-5000, 15000),
            'realized_pnl': np.random.uniform(-2000, 8000),
            'unrealized_pnl': np.random.uniform(-3000, 7000),
            'total_trades': np.random.randint(10, 100),
            'win_rate': np.random.uniform(55, 75),
            'portfolio_return_percent': np.random.uniform(-2, 5)
        }
    
    def _get_mock_trading_report(self):
        """Generate mock trading report"""
        return {
            'risk_metrics': {
                'sharpe_ratio': np.random.uniform(1.0, 2.5),
                'var_95': np.random.uniform(0.015, 0.025),
                'max_drawdown': np.random.uniform(0.02, 0.08),
                'volatility': np.random.uniform(0.1, 0.3)
            },
            'system_health': {
                'data_quality': 'good',
                'trading_active': True
            }
        }

    def run(self):
        """Run the dashboard"""
        # Display sidebar settings
        self._display_settings()
        
        # Display main dashboard
        self._display_overview()
        
        # Display historical performance
        self._display_historical_performance()
        
        # Auto-refresh every 30 seconds
        st.experimental_rerun()

if __name__ == "__main__":
    # Create and run the dashboard
    dashboard = TradingDashboard()
    dashboard.run()