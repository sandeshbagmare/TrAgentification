"""
Monitoring Agent for AI Trading System
Responsible for tracking trades, generating reports, and sending alerts
"""
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
import streamlit as st
from prometheus_client import start_http_server, Gauge, Counter, Histogram

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Data class for tracking trades"""
    trade_id: str
    symbol: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = datetime.now()
    exit_time: Optional[datetime] = None
    status: str = 'open'  # 'open', 'closed', 'cancelled'
    strategy: str = 'unknown'
    pnl: float = 0.0
    
    def calculate_pnl(self) -> float:
        """Calculate P&L for the trade"""
        if self.exit_price is None:
            return 0.0
        
        return (self.exit_price - self.entry_price) * self.quantity
    
    def duration(self) -> timedelta:
        """Calculate trade duration"""
        if self.exit_time is None:
            return datetime.now() - self.entry_time
        
        return self.exit_time - self.entry_time

class MonitoringMetric:
    """Class for tracking monitoring metrics"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.values = []
        self.timestamps = []
        
    def add_value(self, value: float, timestamp: datetime = None):
        """Add a new value to the metric"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.values.append(value)
        self.timestamps.append(timestamp)
    
    def get_values(self, start_time: datetime = None, end_time: datetime = None):
        """Get values within a time range"""
        if start_time is None and end_time is None:
            return self.values
        
        filtered_values = []
        for value, timestamp in zip(self.values, self.timestamps):
            if start_time is None or timestamp >= start_time:
                if end_time is None or timestamp <= end_time:
                    filtered_values.append(value)
        
        return filtered_values
    
    def get_recent_value(self):
        """Get the most recent value"""
        if self.values:
            return self.values[-1]
        return None

class PerformanceMetric(MonitoringMetric):
    """Class for tracking performance metrics like Sharpe ratio, win rate, etc."""
    
    def __init__(self, name: str, description: str, window_size: int = 30):
        super().__init__(name, description)
        self.window_size = window_size  # Window size in days for rolling calculations
        
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio based on P&L history"""
        if len(self.values) < 2:
            return 0.0
        
        # Calculate daily returns
        daily_returns = np.diff(self.values) / self.values[:-1]
        
        # Calculate excess returns
        excess_returns = daily_returns - risk_free_rate / 252  # Daily risk-free rate
        
        # Calculate Sharpe ratio
        if excess_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized
        return sharpe_ratio
    
    def calculate_win_rate(self) -> float:
        """Calculate win rate based on positive P&L"""
        if not self.values:
            return 0.0
        
        wins = sum(1 for value in self.values if value > 0)
        return wins / len(self.values)
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from P&L history"""
        if not self.values:
            return 0.0
        
        # Calculate cumulative P&L
        cumulative_pnl = np.cumsum(self.values)
        
        # Calculate drawdowns
        max_pnl = np.maximum.accumulate(cumulative_pnl)
        drawdowns = max_pnl - cumulative_pnl
        
        # Calculate maximum drawdown as percentage
        if max_pnl[-1] == 0:
            return 0.0
        
        return drawdowns.max() / max_pnl[-1]
    
    def calculate_average_return(self) -> float:
        """Calculate average return per trade"""
        if not self.values:
            return 0.0
        
        return np.mean(self.values)
    
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not self.values:
            return 0.0
        
        gross_profit = sum(value for value in self.values if value > 0)
        gross_loss = abs(sum(value for value in self.values if value < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss

class AlertSystem:
    """Class for managing alerts and notifications"""
    
    def __init__(self, channels: List[str] = ['telegram', 'slack', 'email']):
        self.channels = channels
        self.alert_history = []
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes cooldown between alerts of the same type
        
    def send_alert(self, message: str, alert_type: str = 'general', 
                  severity: str = 'info', channel: str = None):
        """Send an alert through the specified channel"""
        # Check if we should send this alert (based on severity and type)
        if not self._should_send_alert(alert_type, severity):
            return {'success': False, 'reason': 'Alert suppressed due to cooldown'}
        
        # If no specific channel is provided, send to all channels
        if channel is None:
            results = []
            for ch in self.channels:
                result = self._send_alert_to_channel(message, alert_type, severity, ch)
                results.append(result)
            
            return {'success': all(r['success'] for r in results), 'details': results}
        
        # Send to specific channel
        return self._send_alert_to_channel(message, alert_type, severity, channel)
    
    def _send_alert_to_channel(self, message: str, alert_type: str, 
                              severity: str, channel: str) -> Dict:
        """Send an alert to a specific channel"""
        # In a real system, this would interface with the specific channel API
        # For now, just log the alert
        logger.info(f"{channel.upper()} ALERT [{alert_type}]: {message}")
        
        # Record alert in history
        alert_record = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'type': alert_type,
            'severity': severity,
            'channel': channel,
            'success': True
        }
        
        self.alert_history.append(alert_record)
        self.last_alert_time[alert_type] = datetime.now()
        
        return {'success': True, 'channel': channel}
    
    def _should_send_alert(self, alert_type: str, severity: str) -> bool:
        """Determine if an alert should be sent based on type and severity"""
        # Always send critical alerts
        if severity == 'critical':
            return True
        
        # Check cooldown for other alerts
        if alert_type in self.last_alert_time:
            last_alert = self.last_alert_time[alert_type]
            if (datetime.now() - last_alert).total_seconds() < self.alert_cooldown:
                return False
        
        return True

class Dashboard:
    """Class for the real-time monitoring dashboard"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.trade_history = []
        self.performance_metrics = {
            'pnl': PerformanceMetric('pnl', 'Profit and Loss'),
            'win_rate': PerformanceMetric('win_rate', 'Win Rate'),
            'sharpe_ratio': PerformanceMetric('sharpe_ratio', 'Sharpe Ratio'),
            'drawdown': PerformanceMetric('drawdown', 'Drawdown')
        }
        self.system_metrics = {
            'api_errors': MonitoringMetric('api_errors', 'API Error Count'),
            'data_discrepancies': MonitoringMetric('data_discrepancies', 'Data Discrepancy Count'),
            'agent_health': MonitoringMetric('agent_health', 'Agent Health Status')
        }
        
    def start(self):
        """Start the monitoring dashboard"""
        # In a real system, this would start a web server for the dashboard
        # For now, we'll just initialize the metrics
        logger.info(f"Starting dashboard on port {self.port}")
        
        # Start Prometheus metrics server
        start_http_server(self.port)
        
        # Create Prometheus metrics
        self._initialize_prometheus_metrics()
        
        return {'success': True, 'port': self.port}
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics for monitoring"""
        # Define Prometheus metrics
        self.pnl_gauge = Gauge('trading_pnl', 'Current trading profit and loss')
        self.trade_count = Counter('total_trades', 'Total number of trades')
        self.win_rate_gauge = Gauge('win_rate', 'Current win rate')
        self.sharpe_ratio_gauge = Gauge('sharpe_ratio', 'Current Sharpe ratio')
        self.drawdown_gauge = Gauge('drawdown', 'Current drawdown')
        self.agent_health_gauge = Gauge('agent_health', 'Health status of agents', ['agent'])
        
        return {'success': True}
    
    def update(self, trade: Trade = None, system_status: Dict = None):
        """Update dashboard with new data"""
        # Update trade history
        if trade:
            self.trade_history.append(trade)
            
            # Update performance metrics
            if trade.status == 'closed' and trade.pnl != 0:
                self.performance_metrics['pnl'].add_value(trade.pnl)
                self.performance_metrics['win_rate'].add_value(1 if trade.pnl > 0 else 0)
                
                # Update Prometheus metrics
                self.pnl_gauge.set(trade.pnl)
                self.trade_count.inc()
                
        # Update system metrics
        if system_status:
            for agent, status in system_status.items():
                self.agent_health_gauge.labels(agent=agent).set(status['health_score'])
                
        return {'success': True}
    
    def generate_daily_report(self) -> Dict:
        """Generate a daily report with trading performance"""
        if not self.trade_history:
            return {'success': False, 'error': 'No trade history to generate report'}
        
        # Calculate report data
        today = datetime.now().date()
        daily_trades = [t for t in self.trade_history if t.entry_time.date() == today]
        
        # Calculate daily statistics
        daily_pnl = sum(t.pnl for t in daily_trades)
        daily_wins = sum(1 for t in daily_trades if t.pnl > 0)
        daily_losses = sum(1 for t in daily_trades if t.pnl < 0)
        
        # Calculate metrics
        win_rate = daily_wins / (daily_wins + daily_losses) * 100 if (daily_wins + daily_losses) > 0 else 0
        average_return = sum(t.pnl for t in daily_trades) / len(daily_trades) if daily_trades else 0
        
        # Calculate weekly and monthly statistics
        week_ago = datetime.now() - timedelta(days=7)
        monthly_ago = datetime.now() - timedelta(days=30)
        
        weekly_trades = [t for t in self.trade_history if t.entry_time >= week_ago]
        monthly_trades = [t for t in self.trade_history if t.entry_time >= monthly_ago]
        
        # Calculate Sharpe ratio
        daily_returns = [t.pnl for t in daily_trades]
        sharpe_ratio = self.performance_metrics['sharpe_ratio'].calculate_sharpe_ratio()
        
        # Calculate maximum drawdown
        max_drawdown = self.performance_metrics['drawdown'].calculate_max_drawdown()
        
        return {
            'success': True,
            'date': today.isoformat(),
            'total_trades': len(daily_trades),
            'total_pnl': daily_pnl,
            'win_rate': win_rate,
            'average_return': average_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'weekly_trades': len(weekly_trades),
            'weekly_pnl': sum(t.pnl for t in weekly_trades),
            'monthly_trades': len(monthly_trades),
            'monthly_pnl': sum(t.pnl for t in monthly_trades)
        }
    
    def generate_report_chart(self, report_data: Dict):
        """Generate a chart for the daily report"""
        # In a real system, this would generate a chart
        # For now, just return a placeholder
        return {'success': True, 'chart_url': 'https://example.com/charts/placeholder'}
    
    def plot_performance(self):
        """Plot trading performance metrics"""
        # Create a new figure
        plt.figure(figsize=(12, 6))
        
        # Plot P&L over time
        if self.performance_metrics['pnl'].values:
            plt.plot(self.performance_metrics['pnl'].timestamps, 
                     np.cumsum(self.performance_metrics['pnl'].values), 
                     label='Cumulative P&L')
            
        # Add labels and title
        plt.title('Trading Performance')
        plt.xlabel('Time')
        plt.ylabel('P&L')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Show the plot
        plt.show()
    
    def streamlit_dashboard(self):
        """Streamlit-based real-time dashboard implementation"""
        # Initialize Streamlit dashboard
        st.set_page_config(page_title='AI Trading System Dashboard', layout='wide')
        
        # Create dashboard UI
        st.title('AI Trading System Dashboard')
        st.markdown('Real-time monitoring of the trading system')
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Live Trading", "Performance", "System Health"])
        
        # Live trading tab
        with tab1:
            st.header('Live Trading Activity')
            # Implementation for live trading view
            # - Current positions
            # - Recent trades
            # - Order book
            
        # Performance tab
        with tab2:
            st.header('Performance Metrics')
            # Implementation for performance metrics
            # - P&L chart
            # - Win rate
            # - Sharpe ratio
            # - Drawdown
            
        # System health tab
        with tab3:
            st.header('System Health')
            # Implementation for system health monitoring
            # - Agent status
            # - API error rates
            # - Data quality
            
        return {'success': True}

class MonitoringAgent:
    """
    Monitoring Agent
    Oversees system health, generates reports, and sends alerts
    """
    
    def __init__(self, dashboard_port: int = 8000, 
                 log_dir: str = 'logs', 
                 alert_channels: List[str] = ['telegram', 'slack', 'email']):
        self.dashboard_port = dashboard_port
        self.log_dir = log_dir
        self.alert_channels = alert_channels
        
        # Initialize alert system
        self.alert_system = AlertSystem(alert_channels)
        
        # Initialize dashboard
        self.dashboard = Dashboard(dashboard_port)
        
        # Initialize metrics
        self.metrics = {
            'total_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'agent_health': {},
            'api_errors': 0,
            'data_discrepancies': 0
        }
        
        # Trade history
        self.trade_history = []
        
    async def monitor_system(self, system_components: Dict):
        """Monitor the health of all system components"""
        system_health = {}
        
        # Check each component
        for component_name, component in system_components.items():
            try:
                # Get component health status
                health_status = await self._check_component_health(component)
                
                # Update health metrics
                system_health[component_name] = health_status
                
                # Send alert if component is unhealthy
                if not health_status['healthy']:
                    await self._handle_component_failure(component_name, health_status)
                
            except Exception as e:
                logger.error(f"Error monitoring {component_name}: {str(e)}")
                system_health[component_name] = {
                    'healthy': False,
                    'error': str(e),
                    'health_score': 0.0
                }
                
                # Send critical alert for component failure
                await self.alert_system.send_alert(
                    f"Critical failure in {component_name}: {str(e)}",
                    alert_type='system_failure',
                    severity='critical'
                )
        
        # Update metrics
        self.metrics['agent_health'] = system_health
        
        return system_health
    
    async def _check_component_health(self, component: object) -> Dict:
        """Check the health of a system component"""
        # In a real system, this would check component status
        # For now, assume components have a health check method
        
        if hasattr(component, 'check_health'):
            return await component.check_health()
        
        # Default health check
        return {
            'healthy': True,
            'health_score': 1.0,
            'message': 'Component health check not implemented'
        }
    
    async def _handle_component_failure(self, component_name: str, health_status: Dict):
        """Handle component failure by sending alerts and initiating failover"""
        logger.error(f"Component {component_name} is unhealthy: {health_status.get('message', 'Unknown issue')}")
        
        # Send alert
        await self.alert_system.send_alert(
            f"Unhealthy component: {component_name} - {health_status.get('message', 'Unknown issue')}",
            alert_type='system_failure',
            severity='critical'
        )
        
        # Implement circuit breaker pattern
        logger.info(f"Initiating circuit breaker for {component_name}")
        
        # Implement failover procedures
        logger.info(f"Initiating failover procedures for {component_name}")
        
        return {'success': True}
    
    async def log_trade(self, trade: Trade, pnl: float = 0.0, is_win: bool = False):
        """Log a trade and update performance metrics"""
        # Add to trade history
        self.trade_history.append(trade)
        
        # Update metrics
        self.metrics['total_trades'] += 1
        self.metrics['total_pnl'] += trade.pnl
        
        # Update dashboard
        self.dashboard.update(trade)
        
        # Update win rate
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = (sum(1 for t in self.trade_history if t.pnl > 0) / 
                                      self.metrics['total_trades']) * 100
        
        # Generate report if needed
        if (datetime.now().hour == 15 and datetime.now().minute == 30):  # End of trading day
            await self.generate_daily_report()
        
        return {'success': True}
    
    async def generate_daily_report(self):
        """Generate and distribute the daily report"""
        # Generate report data
        report_data = self.dashboard.generate_daily_report()
        
        if not report_data['success']:
            logger.error("Failed to generate daily report")
            return {'success': False, 'error': report_data.get('error')}
        
        # Generate report chart
        chart_data = self.dashboard.generate_report_chart(report_data)
        
        # Create report message
        report_message = f"Daily Trading Report - {report_data['date']}
\nTotal Trades: {report_data['total_trades']}
\nTotal P&L: ₹{report_data['total_pnl']:.2f}
\nWin Rate: {report_data['win_rate']:.2f}%
\nAverage Return: ₹{report_data['average_return']:.2f}
\nSharpe Ratio: {report_data['sharpe_ratio']:.2f}
\nMax Drawdown: {report_data['max_drawdown']:.2f}%
\nWeekly Trades: {report_data['weekly_trades']}
\nWeekly P&L: ₹{report_data['weekly_pnl']:.2f}
\nMonthly Trades: {report_data['monthly_trades']}
\nMonthly P&L: ₹{report_data['monthly_pnl']:.2f}"
        
        # Send report via all alert channels
        for channel in self.alert_channels:
            await self.alert_system.send_alert(
                report_message,
                alert_type='daily_report',
                severity='info',
                channel=channel
            )
        
        # Save report to disk
        await self._save_daily_report(report_data)
        
        return {'success': True}
    
    async def _save_daily_report(self, report_data: Dict):
        """Save daily report to disk"""
        # Implementation to save reports to disk
        # This would typically include:
        # - Saving as CSV
        # - Generating PDF report
        # - Storing in database
        # For now, just log the report
        
        logger.info("Daily report generated:")
        logger.info(report_data)
        
        return {'success': True}
    
    async def start(self):
        """Start the monitoring agent"""
        logger.info("Starting monitoring agent...")
        
        # Start dashboard
        dashboard_result = self.dashboard.start()
        if not dashboard_result['success']:
            logger.warning("Failed to start dashboard")
        
        # Start real-time monitoring loop
        while True:
            try:
                # Update performance metrics
                # (implementation would include more detailed metrics)
                
                # Update dashboard
                # (implementation would include real-time data updates)
                
                # Sleep for health check interval
                await asyncio.sleep(60)  # Every 60 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                break
        
        return {'success': True}
    
    async def emergency_stop(self):
        """Emergency stop for the monitoring agent"""
        logger.info("Initiating emergency stop for monitoring agent...")
        # Implementation for emergency stop procedures
        # This would include:
        # - Closing dashboard
        # - Saving any unsaved data
        # - Notifying other agents of shutdown
        pass