"""
Unit tests for the Execution Engine
"""
import pytest
import asyncio
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from agents.execution_engine import ExecutionEngine, Order, OrderManager, ImmediateOrCancelExecution, SmartOrderRouter

# Sample order data for testing
@dataclass
class OrderDetails:
    """Data class for creating order details for testing"""
    symbol: str = 'RELIANCE'
    quantity: float = 10.0
    price: float = 2500.0
    side: str = 'buy'  # 'buy' or 'sell'
    order_type: str = 'limit'  # 'limit', 'market', 'stop_loss', 'take_profit'
    timestamp: datetime = datetime.now()


class TestOrder:
    """Test suite for the Order data class"""
    
    def test_order_initialization(self):
        """Test basic order initialization"""
        order = Order(
            order_id='ORDER_1',
            symbol='RELIANCE',
            quantity=10.0,
            price=2500.0,
            order_type='limit',
            side='buy',
            status='pending',
            timestamp=datetime.now()
        )
        
        assert order.order_id == 'ORDER_1'
        assert order.symbol == 'RELIANCE'
        assert order.quantity == 10.0
        assert order.price == 2500.0
        assert order.order_type == 'limit'
        assert order.side == 'buy'
        assert order.status == 'pending'
        
    def test_is_filled(self):
        """Test is_filled method of Order class"""
        order = Order(
            order_id='ORDER_1',
            symbol='RELIANCE',
            quantity=10.0,
            price=2500.0,
            order_type='limit',
            side='buy',
            status='pending',
            timestamp=datetime.now()
        )
        
        # Test with no filled quantity
        assert not order.is_filled()
        
        # Test with partial fill
        order.filled_quantity = 5.0
        assert not order.is_filled()
        
        # Test with full fill
        order.filled_quantity = 10.0
        assert order.is_filled()
        
    def test_remaining_quantity(self):
        """Test remaining_quantity method of Order class"""
        order = Order(
            order_id='ORDER_1',
            symbol='RELIANCE',
            quantity=10.0,
            price=2500.0,
            order_type='limit',
            side='buy',
            status='pending',
            timestamp=datetime.now()
        )
        
        # Test with no fills
        assert order.remaining_quantity() == 10.0
        
        # Test with partial fill
        order.filled_quantity = 3.0
        assert order.remaining_quantity() == 7.0
        
        # Test with full fill
        order.filled_quantity = 10.0
        assert order.remaining_quantity() == 0.0
        

class TestOrderManager:
    """Test suite for the OrderManager class"""
    
    def test_order_lifecycle(self):
        """Test order creation, update, and status tracking"""
        order_manager = OrderManager()
        
        # Create an order
        order = Order(
            order_id=order_manager.generate_order_id(),
            symbol='RELIANCE',
            quantity=10.0,
            price=2500.0,
            order_type='limit',
            side='buy',
            status='pending',
            timestamp=datetime.now()
        )
        
        # Test order addition
        order_manager.add_order(order)
        assert order_manager.get_order(order.order_id).order_id == order.order_id
        
        # Test order update
        update_data = {
            'status': 'filled',
            'filled_quantity': 10.0,
            'filled_price': 2500.0
        }
        
        result = order_manager.update_order(order.order_id, update_data)
        assert result['success']
        updated_order = order_manager.get_order(order.order_id)
        assert updated_order.status == 'filled'
        assert updated_order.filled_quantity == 10.0
        
        # Test order history
        assert len(order_manager.order_history) == 1
        assert order_manager.order_history[0]['new_status'] == 'filled'
        
    def test_order_cancellation(self):
        """Test order cancellation functionality"""
        order_manager = OrderManager()
        
        # Create and add an order
        order = Order(
            order_id=order_manager.generate_order_id(),
            symbol='RELIANCE',
            quantity=10.0,
            price=2500.0,
            order_type='limit',
            side='buy',
            status='pending',
            timestamp=datetime.now()
        )
        
        order_manager.add_order(order)
        
        # Test single order cancellation
        result = order_manager.cancel_order(order.order_id)
        assert result['success']
        cancelled_order = order_manager.get_order(order.order_id)
        assert cancelled_order.status == 'cancelled'
        
        # Test cancellation of all orders
        # Create another order
        order2 = Order(
            order_id=order_manager.generate_order_id(),
            symbol='INFY',
            quantity=5.0,
            price=1200.0,
            order_type='limit',
            side='sell',
            status='pending',
            timestamp=datetime.now()
        )
        
        order_manager.add_order(order2)
        result = order_manager.cancel_all_orders()
        assert result['success']
        assert len(result['cancelled_orders']) == 2
        
        # Check that both orders are cancelled
        assert order_manager.get_order(order.order_id).status == 'cancelled'
        assert order_manager.get_order(order2.order_id).status == 'cancelled'
        
    def test_pending_orders(self):
        """Test getting pending orders"""
        order_manager = OrderManager()
        
        # Create and add multiple orders
        orders = []
        for i in range(3):
            order = Order(
                order_id=order_manager.generate_order_id(),
                symbol='RELIANCE',
                quantity=10.0,
                price=2500.0 - i * 0.5,  # Slightly different prices
                order_type='limit',
                side='buy',
                status='pending' if i < 2 else 'filled',
                timestamp=datetime.now()
            )
            order_manager.add_order(order)
            orders.append(order)
        
        # Test get_pending_orders
        pending_orders = order_manager.get_pending_orders()
        assert len(pending_orders) == 2
        assert all(order.status == 'pending' for order in pending_orders)
        
    def test_order_history(self):
        """Test order history tracking"""
        order_manager = OrderManager()
        
        # Create and process orders
        order = Order(
            order_id=order_manager.generate_order_id(),
            symbol='RELIANCE',
            quantity=10.0,
            price=2500.0,
            order_type='limit',
            side='buy',
            status='pending',
            timestamp=datetime.now()
        )
        
        order_manager.add_order(order)
        
        # Update order status to filled
        update_data = {
            'status': 'filled',
            'filled_quantity': 10.0,
            'filled_price': 2500.0
        }
        
        order_manager.update_order(order.order_id, update_data)
        
        # Create another order
        order2 = Order(
            order_id=order_manager.generate_order_id(),
            symbol='INFY',
            quantity=5.0,
            price=1200.0,
            order_type='limit',
            side='sell',
            status='pending',
            timestamp=datetime.now()
        )
        
        order_manager.add_order(order2)
        
        # Test get_order_history for specific symbol
        history = order_manager.get_order_history('RELIANCE')
        assert len(history) == 1
        assert history[0]['symbol'] == 'RELIANCE'
        
        # Test get_order_history for all symbols
        history = order_manager.get_order_history()
        assert len(history) == 2
        

class TestImmediateOrCancelExecution:
    """Test suite for the ImmediateOrCancelExecution strategy"""
    
    @pytest.fixture
    def order_manager(self):
        """Create a new OrderManager instance for each test"""
        return OrderManager()
    
    @pytest.fixture
    def execution_strategy(self, order_manager):
        """Create an ImmediateOrCancelExecution instance for each test"""
        return ImmediateOrCancelExecution(order_manager=order_manager, slippage_tolerance=0.001)
    
    def test_order_execution(self, execution_strategy, order_manager):
        """Test order execution with different scenarios"""
        # Create a test order
        order_details = OrderDetails()
        order = Order(
            order_id=order_manager.generate_order_id(),
            symbol=order_details.symbol,
            quantity=order_details.quantity,
            price=order_details.price,
            order_type=order_details.order_type,
            side=order_details.side,
            status='pending',
            timestamp=datetime.now()
        )
        
        # Test successful order execution
        result = asyncio.run(execution_strategy.execute_order(order))
        assert result['success']
        assert result['status'] == 'filled'
        assert result['filled_quantity'] == order.quantity
        
        # Test order execution failure
        # Create an order with invalid data
        order_details = OrderDetails(price=0.0)  # Zero price should fail
        order = Order(
            order_id=order_manager.generate_order_id(),
            symbol=order_details.symbol,
            quantity=order_details.quantity,
            price=order_details.price,
            order_type=order_details.order_type,
            side=order_details.side,
            status='pending',
            timestamp=datetime.now()
        )
        
        result = asyncio.run(execution_strategy.execute_order(order))
        assert not result['success']
        assert result['status'] == 'rejected'
        
    def test_order_cancellation(self, execution_strategy, order_manager):
        """Test order cancellation functionality"""
        # Create and add an order
        order = Order(
            order_id=order_manager.generate_order_id(),
            symbol='RELIANCE',
            quantity=10.0,
            price=2500.0,
            order_type='limit',
            side='buy',
            status='pending',
            timestamp=datetime.now()
        )
        
        order_manager.add_order(order)
        
        # Test order cancellation
        result = asyncio.run(execution_strategy.cancel_order(order.order_id))
        assert result['success']
        assert order_manager.get_order(order.order_id).status == 'cancelled'
        
    def test_order_status(self, execution_strategy, order_manager):
        """Test order status checking"""
        # Create and add an order
        order = Order(
            order_id=order_manager.generate_order_id(),
            symbol='RELIANCE',
            quantity=10.0,
            price=2500.0,
            order_type='limit',
            side='buy',
            status='pending',
            timestamp=datetime.now()
        )
        
        order_manager.add_order(order)
        
        # Test getting order status
        status = asyncio.run(execution_strategy.get_order_status(order.order_id))
        assert status['success']
        assert status['order_id'] == order.order_id
        assert status['status'] == 'pending'
        

class TestExecutionEngine:
    """Test suite for the ExecutionEngine class"""
    
    @pytest.fixture
    def execution_engine(self):
        """Create an ExecutionEngine instance with test credentials"""
        return ExecutionEngine(
            upstox_api_key='test_api_key',
            upstox_api_secret='test_api_secret',
            order_type='immediate_or_cancel',
            slippage_tolerance=0.001
        )
    
    def test_order_validation(self, execution_engine):
        """Test order validation with different scenarios"""
        # Test with no symbol
        order_details = OrderDetails(symbol=None)
        result = asyncio.run(execution_engine.place_order(order_details.__dict__))
        assert not result['success']
        assert 'Symbol is required' in result['error']
        
        # Test with invalid quantity
        order_details = OrderDetails(quantity=0)
        result = asyncio.run(execution_engine.place_order(order_details.__dict__))
        assert not result['success']
        assert 'Valid quantity is required' in result['error']
        
        # Test with invalid side
        order_details = OrderDetails(side='hold')  # Invalid side
        result = asyncio.run(execution_engine.place_order(order_details.__dict__))
        assert not result['success']
        assert 'Order side must be buy or sell' in result['error']
        
    def test_order_placement(self, execution_engine):
        """Test order placement with different order types"""
        # Test limit order
        order_details = OrderDetails(order_type='limit')
        result = asyncio.run(execution_engine.place_order(order_details.__dict__))
        assert result['success']
        assert result['order_id']
        assert result['status'] in ['filled', 'partial', 'rejected']
        
        # Test market order
        order_details = OrderDetails(order_type='market')
        result = asyncio.run(execution_engine.place_order(order_details.__dict__))
        assert result['success']
        assert result['order_id']
        assert result['status'] in ['filled', 'partial', 'rejected']
        
        # Test stop-loss order
        order_details = OrderDetails(order_type='stop_loss')
        result = asyncio.run(execution_engine.place_order(order_details.__dict__))
        assert result['success']
        assert result['order_id']
        assert result['status'] in ['filled', 'partial', 'rejected']
        
        # Test take-profit order
        order_details = OrderDetails(order_type='take_profit')
        result = asyncio.run(execution_engine.place_order(order_details.__dict__))
        assert result['success']
        assert result['order_id']
        assert result['status'] in ['filled', 'partial', 'rejected']
        
    def test_order_modification(self, execution_engine):
        """Test order modification functionality"""
        # Create and place an order
        order_details = OrderDetails()
        result = asyncio.run(execution_engine.place_order(order_details.__dict__))
        
        # Test modifying the order
        if result['success']:
            update_data = {'price': 2501.0}
            result = asyncio.run(execution_engine.modify_order(result['order_id'], update_data))
            assert result['success']
            assert result['status'] in ['pending', 'partial']
        
    def test_order_cancellation(self, execution_engine):
        """Test order cancellation functionality"""
        # Create and place an order
        order_details = OrderDetails()
        result = asyncio.run(execution_engine.place_order(order_details.__dict__))
        
        # Test cancelling a single order
        if result['success']:
            result = asyncio.run(execution_engine.cancel_order(result['order_id']))
            assert result['success']
            assert result['order_id']
        
        # Test cancelling all orders
        # Place a few more orders
        results = asyncio.run(execution_engine.place_orders([
            OrderDetails(symbol='INFY', price=1200.0).__dict__,
            OrderDetails(symbol='TATA', price=150.0).__dict__
        ]))
        
        if results['success']:
            result = asyncio.run(execution_engine.cancel_all_orders())
            assert result['success']
            assert len(result['cancelled_orders']) == 2
        
    def test_order_failure_handling(self, execution_engine):
        """Test order failure handling with fallback strategies"""
        # Create a failing order (invalid symbol)
        order_details = OrderDetails(symbol='INVALID_SYMBOL')
        result = asyncio.run(execution_engine.place_order(order_details.__dict__))
        
        # The order should fail but the failure handler should attempt a fallback
        assert not result['success']
        assert 'error' in result
        assert 'order_id' in result
        assert 'status' in result
        
    @pytest.mark.asyncio
    async def test_start(self, execution_engine):
        """Test starting the execution engine"""
        result = await execution_engine.start()
        assert result['success']
        
    @pytest.mark.asyncio
    async def test_emergency_stop(self, execution_engine):
        """Test emergency stop functionality"""
        result = await execution_engine.emergency_stop()
        assert result['success']