"""
Main entry point for the AI Trading System
This module initializes the trading system and starts all agents
"""
import os
import logging
from dotenv import load_dotenv
from agents import DataAgent, TechnicalAnalysisAgent, MlAgent, StrategyOrchestrator, RiskManager, ExecutionEngine, MonitoringAgent
from agents.ai_integrations import GeminiIntegration, AzureOpenAIIntegration

def initialize_trading_system():
    """Initialize the trading system with all required components"""
    # Load environment variables
    load_dotenv()
    
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize data agent
    data_agent = DataAgent(
        market_data_sources=os.getenv('MARKET_DATA_SOURCES', 'upstox,finnhub,alpha_vantage').split(','),
        historical_data_dir='data/historical',
        tick_interval=int(os.getenv('TICK_INTERVAL', '1'))
    )
    
    # Initialize technical analysis agent
    technical_agent = TechnicalAnalysisAgent(
        indicators=os.getenv('TECHNICAL_INDICATORS', 'rsi,macd,bollinger,ma,atr').split(',')
    )
    
    # Initialize ML agent
    ml_agent = MlAgent(
        model_dir='models',
        training_interval=int(os.getenv('MODEL_UPDATE_INTERVAL', '86400')),
        models_to_train=os.getenv('ML_MODELS', 'lightgbm,xgboost,lstm').split(',')
    )
    
    # Initialize strategy orchestrator
    strategy_orchestrator = StrategyOrchestrator(
        confidence_threshold=float(os.getenv('STRATEGY_CONFIDENCE_THRESHOLD', '0.7')),
        require_consenus=bool(os.getenv('REQUIRE_CONSENSUS', 'True'))
    )
    
    # Initialize AI integrations if not already initialized
    if not hasattr(strategy_orchestrator, 'ai_integrations'):
        strategy_orchestrator._initialize_ai_integrations()
    
    # Initialize risk manager
    risk_manager = RiskManager(
        max_risk_per_trade=float(os.getenv('MAX_RISK_PER_TRADE', '0.005')),
        daily_loss_cap=float(os.getenv('DAILY_LOSS_CAP', '0.03')),
        stop_loss_levels=os.getenv('STOP_LOSS_LEVELS', 'fixed,adaptive,trailing').split(',')
    )
    
    # Initialize execution engine
    execution_engine = ExecutionEngine(
        upstox_api_key=os.getenv('UPSTOX_API_KEY'),
        upstox_api_secret=os.getenv('UPSTOX_API_SECRET'),
        order_type=os.getenv('DEFAULT_ORDER_TYPE', 'limit'),
        slippage_tolerance=float(os.getenv('SLIPPAGE_TOLERANCE', '0.001'))
    )
    
    # Initialize monitoring agent
    monitoring_agent = MonitoringAgent(
        dashboard_port=int(os.getenv('DASHBOARD_PORT', '8000')),
        log_dir='logs',
        alert_channels=os.getenv('ALERT_CHANNELS', 'telegram,slack,email').split(',')
    )
    
    return {
        'data_agent': data_agent,
        'technical_agent': technical_agent,
        'ml_agent': ml_agent,
        'strategy_orchestrator': strategy_orchestrator,
        'risk_manager': risk_manager,
        'execution_engine': execution_engine,
        'monitoring_agent': monitoring_agent
    }


def run_trading_system():
    """Run the trading system with all initialized components"""
    logging.info("Initializing trading system...")
    components = initialize_trading_system()
    
    logging.info("Starting agents...")
    for name, component in components.items():
        try:
            component.start()
            logging.info(f"{name} started successfully")
        except Exception as e:
            logging.error(f"Failed to start {name}: {str(e)}")
            # Implement circuit breaker pattern - if critical component fails, halt the system
            if name in ['risk_manager', 'execution_engine', 'data_agent']:
                logging.critical("Critical component failed, initiating system shutdown")
                shutdown_trading_system(components)
                return
    
    logging.info("Trading system is now running")
    

def shutdown_trading_system(components):
    """Gracefully shutdown the trading system"""
    logging.info("Initiating system shutdown...")
    
    # Implement emergency stop mechanism
    for name, component in components.items():
        try:
            component.emergency_stop()
            logging.info(f"{name} stopped successfully")
        except Exception as e:
            logging.error(f"Error stopping {name}: {str(e)}")
    
    # Implement circuit breaker pattern - if any component fails to stop, force shutdown
    logging.info("System shutdown complete")


if __name__ == "__main__":
    run_trading_system()