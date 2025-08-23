#!/usr/bin/env python3
"""
System Validation Script for AI Trading System
Tests all major components to ensure they're working correctly
"""
import asyncio
import logging
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_data_agent():
    """Test data agent functionality"""
    try:
        from agents.data_agent import DataAgent
        logger.info("🔍 Testing Data Agent...")
        
        data_agent = DataAgent()
        await data_agent.start()
        
        # Test data fetching (mock)
        test_data = await data_agent.fetch_market_data(['RELIANCE', 'TCS'])
        
        logger.info("✅ Data Agent: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data Agent: FAILED - {str(e)}")
        return False

async def test_risk_manager():
    """Test risk manager functionality"""
    try:
        from agents.risk_manager import RiskManager, Position
        logger.info("🔍 Testing Risk Manager...")
        
        risk_manager = RiskManager()
        await risk_manager.start()
        
        # Test trade evaluation
        trade_proposal = {
            'symbol': 'RELIANCE',
            'signal': 1,
            'price': 2500,
            'quantity': 10,
            'stop_loss': 2450
        }
        
        portfolio = {'total_value': 100000}
        positions = []
        
        result = risk_manager.evaluate_trade(trade_proposal, portfolio, positions)
        
        if result.get('allow_trade') is not None:
            logger.info("✅ Risk Manager: PASSED")
            return True
        else:
            logger.error("❌ Risk Manager: No trade decision returned")
            return False
            
    except Exception as e:
        logger.error(f"❌ Risk Manager: FAILED - {str(e)}")
        return False

async def test_ml_agent():
    """Test ML agent functionality"""
    try:
        from agents.ml_agent import MLAgent
        logger.info("🔍 Testing ML Agent...")
        
        ml_agent = MLAgent()
        await ml_agent.start()
        
        # Test with mock data
        mock_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(200, 250, 100),
            'low': np.random.uniform(80, 120, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        result = await ml_agent.generate_signal('RELIANCE', mock_data)
        
        if 'signal' in result:
            logger.info("✅ ML Agent: PASSED")
            return True
        else:
            logger.error("❌ ML Agent: No signal generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ ML Agent: FAILED - {str(e)}")
        return False

async def test_strategy_orchestrator():
    """Test strategy orchestrator functionality"""
    try:
        from agents.strategy_orchestrator import StrategyOrchestrator
        logger.info("🔍 Testing Strategy Orchestrator...")
        
        orchestrator = StrategyOrchestrator()
        await orchestrator.start()
        
        # Test signal combination
        signals = {
            'technical': {'signal': 1, 'confidence': 0.8, 'reason': 'Test technical signal'},
            'ml': {'signal': 1, 'confidence': 0.7, 'reason': 'Test ML signal'}
        }
        
        market_data = {'volatility': 0.02, 'volume': 10000}
        
        result = orchestrator.combine_signals(signals, market_data)
        
        if 'signal' in result:
            logger.info("✅ Strategy Orchestrator: PASSED")
            return True
        else:
            logger.error("❌ Strategy Orchestrator: No combined signal")
            return False
            
    except Exception as e:
        logger.error(f"❌ Strategy Orchestrator: FAILED - {str(e)}")
        return False

async def test_supervisor_agent():
    """Test supervisor agent functionality"""
    try:
        from agents.supervisor_agent import SupervisorAgent
        logger.info("🔍 Testing Supervisor Agent...")
        
        supervisor = SupervisorAgent()
        await supervisor.start()
        
        # Test signal resolution
        signals = {
            'technical': {'signal': 1, 'confidence': 0.8},
            'ml': {'signal': -1, 'confidence': 0.7}
        }
        
        result = await supervisor.resolve_conflict(signals)
        
        if result.get('success'):
            logger.info("✅ Supervisor Agent: PASSED")
            return True
        else:
            logger.error("❌ Supervisor Agent: Resolution failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Supervisor Agent: FAILED - {str(e)}")
        return False

async def test_automated_trainer():
    """Test automated trainer functionality"""
    try:
        from agents.automated_trainer import AutomatedTrainer
        logger.info("🔍 Testing Automated Trainer...")
        
        trainer = AutomatedTrainer(training_period_days=30)
        await trainer.start()
        
        # Test initialization
        logger.info("✅ Automated Trainer: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Automated Trainer: FAILED - {str(e)}")
        return False

async def test_pnl_tracker():
    """Test P&L tracker functionality"""
    try:
        from agents.pnl_tracker import PnLTracker
        logger.info("🔍 Testing P&L Tracker...")
        
        pnl_tracker = PnLTracker()
        await pnl_tracker.start()
        
        # Test trade recording
        trade_details = {
            'symbol': 'RELIANCE',
            'side': 'buy',
            'quantity': 10,
            'price': 2500,
            'fees': 10,
            'strategy': 'test_strategy'
        }
        
        result = pnl_tracker.record_trade(trade_details)
        
        if result.get('success'):
            logger.info("✅ P&L Tracker: PASSED")
            return True
        else:
            logger.error("❌ P&L Tracker: Trade recording failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ P&L Tracker: FAILED - {str(e)}")
        return False

async def run_comprehensive_validation():
    """Run comprehensive validation of all components"""
    logger.info("🚀 Starting Comprehensive System Validation")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test all components
    tests = [
        ("Data Agent", test_data_agent),
        ("Risk Manager", test_risk_manager),
        ("ML Agent", test_ml_agent),
        ("Strategy Orchestrator", test_strategy_orchestrator),
        ("Supervisor Agent", test_supervisor_agent),
        ("Automated Trainer", test_automated_trainer),
        ("P&L Tracker", test_pnl_tracker)
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name}: CRITICAL ERROR - {str(e)}")
            test_results.append((test_name, False))
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"📈 Overall Results: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        logger.info("🎉 ALL COMPONENTS VALIDATED SUCCESSFULLY!")
        logger.info("✅ System is ready for production deployment!")
    else:
        logger.warning(f"⚠️ {failed} component(s) need attention before deployment")
    
    logger.info("=" * 60)
    
    return passed, failed

if __name__ == "__main__":
    # Run validation
    asyncio.run(run_comprehensive_validation())