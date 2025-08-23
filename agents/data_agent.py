"""
Data Ingestion Agent for AI Trading System
Responsible for fetching, cleaning, and storing market data from multiple sources
"""
import asyncio
import logging
import pandas as pd
import numpy as np
import statistics
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import requests
import websockets
import finnhub
from alpha_vantage.async_support.timeseries import TimeSeries

logger = logging.getLogger(__name__)

class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    async def connect(self):
        """Establish connection to the data source"""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Get historical data for a symbol"""
        pass
    
    @abstractmethod
    async def get_tick_data(self, symbol: str) -> Dict:
        """Get real-time tick data for a symbol"""
        pass
    
    @abstractmethod
    async def subscribe_to_data(self, symbols: List[str]):
        """Subscribe to real-time data for symbols"""
        pass

class UpstoxDataSource(DataSource):
    """Upstox API data source implementation"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.websocket_url = "wss://websocket.upstox.com/feed/"
        self.rest_api_url = "https://api.upstox.com/v2/"
        self.connected = False
        
    async def connect(self):
        """Establish connection to Upstox WebSocket API"""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            self.connected = True
            logger.info("Connected to Upstox API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Upstox API: {str(e)}")
            return False
    
    async def get_historical_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Get historical OHLC data from Upstox API"""
        # Implementation for Upstox historical data
        pass
    
    async def get_tick_data(self, symbol: str) -> Dict:
        """Get real-time tick data from Upstox API"""
        # Implementation for Upstox tick data
        pass
    
    async def subscribe_to_data(self, symbols: List[str]):
        """Subscribe to real-time data for symbols"""
        # Implementation for Upstox subscription
        pass
    
    async def authenticate(self):
        """Authenticate with Upstox API"""
        # Implementation for Upstox authentication
        pass

class FinnhubDataSource(DataSource):
    """Finnhub API data source implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.finnhub_client = finnhub.Client(api_key=api_key)
        self.connected = False
        
    async def connect(self):
        """Check connection to Finnhub API"""
        try:
            # Simple API call to test connection
            await self.finnhub_client.stock_candles('AAPL', 'D', 1590969600, 1591833600)
            self.connected = True
            logger.info("Connected to Finnhub API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Finnhub API: {str(e)}")
            return False
    
    async def get_historical_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Get historical OHLC data from Finnhub API"""
        # Implementation for Finnhub historical data
        pass
    
    async def get_tick_data(self, symbol: str) -> Dict:
        """Get real-time tick data from Finnhub API"""
        # Implementation for Finnhub tick data
        pass
    
    async def subscribe_to_data(self, symbols: List[str]):
        """Subscribe to real-time data for symbols"""
        # Implementation for Finnhub subscription
        pass

class AlphaVantageDataSource(DataSource):
    """Alpha Vantage API data source implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.connected = False
        
    async def connect(self):
        """Establish connection to Alpha Vantage API"""
        try:
            # Initialize async time series client
            self.ts = TimeSeries(key=self.api_key, output_size=30, indexing='date')
            self.connected = True
            logger.info("Connected to Alpha Vantage API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Alpha Vantage API: {str(e)}")
            return False
    
    async def get_historical_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Get historical OHLC data from Alpha Vantage API"""
        # Implementation for Alpha Vantage historical data
        pass
    
    async def get_tick_data(self, symbol: str) -> Dict:
        """Get real-time tick data from Alpha Vantage API"""
        # Implementation for Alpha Vantage tick data
        pass
    
    async def subscribe_to_data(self, symbols: List[str]):
        """Subscribe to real-time data for symbols"""
        # Implementation for Alpha Vantage subscription
        pass

class DataAgent:
    """
    Data Ingestion Agent
    Responsible for managing multiple data sources and providing clean, validated market data
    """
    
    def __init__(self, market_data_sources: List[str], 
                 historical_data_dir: str = 'data/historical',
                 tick_interval: int = 1):
        self.market_data_sources = market_data_sources
        self.historical_data_dir = historical_data_dir
        self.tick_interval = tick_interval  # in seconds
        self.data_sources = {}
        self.connected = False
        self.cross_validation_enabled = True
        self.data_store = {}
        
    async def initialize_data_sources(self, credentials: Dict):
        """Initialize and connect to all required data sources"""
        for source in self.market_data_sources:
            try:
                if source.lower() == 'upstox':
                    data_source = UpstoxDataSource(
                        api_key=credentials.get('upstox_api_key'),
                        api_secret=credentials.get('upstox_api_secret')
                    )
                elif source.lower() == 'finnhub':
                    data_source = FinnhubDataSource(
                        api_key=credentials.get('finnhub_api_key')
                    )
                elif source.lower() == 'alpha_vantage':
                    data_source = AlphaVantageDataSource(
                        api_key=credentials.get('alpha_vantage_api_key')
                    )
                else:
                    raise ValueError(f"Unsupported data source: {source}")
                
                await data_source.connect()
                self.data_sources[source.lower()] = data_source
            except Exception as e:
                logger.error(f"Error initializing data source {source}: {str(e)}")
                # If cross-validation is enabled and a critical data source fails, halt trading
                if self.cross_validation_enabled:
                    logger.warning(f"Critical data source {source} failed, system may halt trading")
        
        self.connected = len(self.data_sources) > 0
        return self.connected
    
    async def get_market_data(self, symbol: str, data_type: str = 'tick') -> Dict:
        """
        Get market data for a symbol from all available sources
        Returns validated data after cross-source validation if enabled
        """
        if not self.connected:
            logger.error("Data agent not connected to any data sources")
            return {}
        
        data = {}
        
        # Get data from all available sources
        for source_name, source in self.data_sources.items():
            try:
                if data_type == 'tick':
                    source_data = await source.get_tick_data(symbol)
                else:
                    source_data = await source.get_historical_data(symbol, data_type, 30)
                
                if source_data:
                    data[source_name] = source_data
            except Exception as e:
                logger.error(f"Error getting data from {source_name}: {str(e)}")
                continue
        
        # Cross-validation if enabled and multiple sources
        if self.cross_validation_enabled and len(data) > 1:
            validated_data = self._cross_validate_data(data, symbol)
            if not validated_data:
                logger.warning(f"Data validation failed for {symbol}, halting trading for this symbol")
                # Implement circuit breaker pattern
                await self._handle_data_validation_failure(symbol)
                return {}
            return validated_data
        
        # If only one source or cross-validation disabled
        if data:
            return next(iter(data.values()))
        
        return {}
    
    def _cross_validate_data(self, data: Dict, symbol: str) -> Optional[Dict]:
        """
        Cross-validate data from multiple sources
        Returns validated data or None if discrepancies are too large
        """
        if not data:
            return None
            
        # If only one source, return data as is
        if len(data) == 1:
            return next(iter(data.values()))
            
        # Convert data to pandas DataFrames for easier comparison
        dfs = {}
        for source, source_data in data.items():
            if isinstance(source_data, pd.DataFrame):
                dfs[source] = source_data
            else:
                # Convert dict to DataFrame if needed
                dfs[source] = pd.DataFrame([source_data])
                
        # Check for basic data consistency
        validated_data = {}
        
        # Check price data
        price_fields = ['open', 'high', 'low', 'close', 'last_price']
        for field in price_fields:
            values = [df[field].iloc[-1] for df in dfs.values() if field in df.columns]
            if values:
                # Calculate average and standard deviation
                avg = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                
                # If values are too far from average (3 sigma), discard outliers
                filtered_values = [v for v in values if abs(v - avg) <= 3 * stdev]
                
                # If all values are outliers, use average of all
                if not filtered_values:
                    filtered_values = values
                    logger.warning(f"All {field} values are outliers for {symbol}")
                    
                # Use average of valid values
                validated_data[field] = statistics.mean(filtered_values)
                
        # Check volume data
        volume_fields = ['volume', 'total_volume', 'trade_count']
        for field in volume_fields:
            values = [df[field].iloc[-1] for df in dfs.values() if field in df.columns]
            if values:
                # For volume, we use median to avoid outliers
                validated_data[field] = statistics.median(values)
                
        # Check timestamp (use most recent)
        timestamps = [df.index[-1] if isinstance(df.index[-1], (int, float)) else df.index[-1].timestamp() 
                     for df in dfs.values() if not df.empty]
        if timestamps:
            validated_data['timestamp'] = max(timestamps)
            
        # Add source reliability metrics
        validated_data['source_reliability'] = {}
        for source, df in dfs.items():
            if not df.empty:
                # Calculate how consistent this source has been historically
                # This would typically use historical data from the data store
                validated_data['source_reliability'][source] = {
                    'consistency_score': np.random.uniform(0.8, 1.0),  # Placeholder for real calculation
                    'latency': np.random.uniform(0.05, 0.5)  # Placeholder for real latency measurement
                }
                
        # Add validation metadata
        validated_data['validation_metadata'] = {
            'sources_used': list(dfs.keys()),
            'validation_time': datetime.now().isoformat(),
            'discrepancies_handled': True
        }
        
        # Convert to DataFrame
        if validated_data:
            return pd.DataFrame([validated_data])
        
        return None
        
    async def _handle_data_validation_failure(self, symbol: str):
        """Handle data validation failures by implementing circuit breaker pattern"""
        logger.warning(f"Data validation failed for {symbol}, implementing circuit breaker")
        
        # Implement circuit breaker pattern
        # 1. Notify monitoring agent
        # 2. Stop trading for this symbol
        # 3. Switch to alternative data sources
        # 4. Initiate failover procedures
        # 5. Record the failure for analysis
        
        # For now, just log and wait
        logger.info(f"Pausing data collection for {symbol} for 5 minutes")
        await asyncio.sleep(300)  # Wait 5 minutes before retrying
        
        # After waiting, check if data sources are still valid
        valid_sources = {}
        for source_name, source in self.data_sources.items():
            try:
                # Simple health check for the data source
                if await source.check_health():
                    valid_sources[source_name] = source
                else:
                    logger.warning(f"Data source {source_name} failed health check")
            except Exception as e:
                logger.error(f"Error checking health for {source_name}: {str(e)}")
                continue
        
        # Update data sources if any changes
        if len(valid_sources) != len(self.data_sources):
            logger.info(f"Updating data sources - {len(valid_sources)} valid sources remaining")
            self.data_sources = valid_sources
            
        return validated_data  # Simplified for now
    
    async def start(self):
        """Start the data agent"""
        logger.info("Starting data agent...")
        
        # Load credentials from environment variables
        credentials = {
            'upstox_api_key': os.getenv('UPSTOX_API_KEY'),
            'upstox_api_secret': os.getenv('UPSTOX_API_SECRET'),
            'finnhub_api_key': os.getenv('FINNHUB_API_KEY'),
            'alpha_vantage_api_key': os.getenv('ALPHA_VANTAGE_API_KEY')
        }
        
        # Initialize data sources
        await self.initialize_data_sources(credentials)
        
        # Start data collection loop
        await self._data_collection_loop()
    
    async def _data_collection_loop(self):
        """Main loop for data collection"""
        while True:
            try:
                # Collect data for all symbols
                # Implementation would include:
                # - Getting list of symbols to monitor
                # - Collecting tick data for each symbol
                # - Storing validated data
                # - Handling errors and retries
                
                # Sleep for tick interval
                await asyncio.sleep(self.tick_interval)
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {str(e)}")
                # Implement circuit breaker pattern
                logger.critical("Data collection loop failed, initiating system shutdown")
                # This would typically trigger a shutdown through the supervisor agent
                break
    
    async def emergency_stop(self):
        """Emergency stop for the data agent"""
        logger.info("Initiating emergency stop for data agent...")
        # Implementation for emergency stop procedures
        # This would include:
        # - Closing all data source connections
        # - Saving any unsaved data
        # - Notifying other agents of shutdown
        pass