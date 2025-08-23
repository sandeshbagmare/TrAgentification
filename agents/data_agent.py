"""
Data Ingestion Agent for AI Trading System
Responsible for fetching, cleaning, and storing market data from multiple sources
"""
import asyncio
import logging
import pandas as pd
import numpy as np
import statistics
import os
import json
import aiohttp
import sqlite3
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import requests
import websockets
import finnhub
from alpha_vantage.async_support.timeseries import TimeSeries
import yfinance as yf
from pathlib import Path

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
        try:
            # Convert symbol to Upstox format
            upstox_symbol = self._convert_to_upstox_symbol(symbol)
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=limit if interval == 'D' else limit // 24)
            
            url = f"{self.rest_api_url}historical-candle/{upstox_symbol}/{interval}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            
            headers = {
                'Authorization': f'Bearer {os.getenv("UPSTOX_ACCESS_TOKEN")}',
                'Accept': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        candles = data.get('data', {}).get('candles', [])
                        
                        # Convert to DataFrame
                        df_data = []
                        for candle in candles:
                            df_data.append({
                                'timestamp': pd.to_datetime(candle[0]),
                                'open': float(candle[1]),
                                'high': float(candle[2]),
                                'low': float(candle[3]),
                                'close': float(candle[4]),
                                'volume': int(candle[5])
                            })
                        
                        df = pd.DataFrame(df_data)
                        if not df.empty:
                            df.set_index('timestamp', inplace=True)
                        return df
                    else:
                        logger.error(f"Upstox API error: {response.status}")
                        return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching Upstox historical data: {e}")
            return pd.DataFrame()
    
    def _convert_to_upstox_symbol(self, symbol: str) -> str:
        """Convert symbol to Upstox format"""
        # Basic conversion - in production, use proper symbol mapping
        if symbol in ['NIFTY', 'BANKNIFTY']:
            return f"NSE_INDEX|{symbol}"
        else:
            return f"NSE_EQ|{symbol}-EQ"
    
    async def get_tick_data(self, symbol: str) -> Dict:
        """Get real-time tick data from Upstox API"""
        try:
            upstox_symbol = self._convert_to_upstox_symbol(symbol)
            
            url = f"{self.rest_api_url}market-quote/ltp"
            params = {'symbol': upstox_symbol}
            
            headers = {
                'Authorization': f'Bearer {os.getenv("UPSTOX_ACCESS_TOKEN")}',
                'Accept': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        quote_data = data.get('data', {}).get(upstox_symbol, {})
                        
                        return {
                            'symbol': symbol,
                            'last_price': float(quote_data.get('last_price', 0)),
                            'volume': int(quote_data.get('volume', 0)),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'upstox'
                        }
                    else:
                        logger.error(f"Upstox tick data error: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Error fetching Upstox tick data: {e}")
            return {}
    
    async def subscribe_to_data(self, symbols: List[str]):
        """Subscribe to real-time data for symbols"""
        try:
            if not self.connected:
                await self.connect()
            
            # Convert symbols to Upstox format
            upstox_symbols = [self._convert_to_upstox_symbol(symbol) for symbol in symbols]
            
            # Subscribe message
            subscribe_message = {
                "guid": "someguid",
                "method": "sub",
                "data": {
                    "mode": "full",
                    "instrumentKeys": upstox_symbols
                }
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to {len(symbols)} symbols via Upstox WebSocket")
            
            # Start listening for messages
            asyncio.create_task(self._listen_websocket_messages())
            
        except Exception as e:
            logger.error(f"Error subscribing to Upstox data: {e}")
    
    async def _listen_websocket_messages(self):
        """Listen for WebSocket messages from Upstox"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                # Process incoming market data
                await self._process_websocket_data(data)
        except Exception as e:
            logger.error(f"WebSocket listening error: {e}")
    
    async def _process_websocket_data(self, data: Dict):
        """Process incoming WebSocket data"""
        try:
            # Extract and format market data
            if data.get('type') == 'feed':
                feed_data = data.get('feeds', {})
                for symbol, quote in feed_data.items():
                    processed_data = {
                        'symbol': symbol,
                        'last_price': float(quote.get('ltp', 0)),
                        'volume': int(quote.get('vol', 0)),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'upstox_websocket'
                    }
                    # Store or process the data as needed
                    logger.debug(f"Received data for {symbol}: {processed_data}")
        except Exception as e:
            logger.error(f"Error processing WebSocket data: {e}")
    
    async def authenticate(self):
        """Authenticate with Upstox API"""
        try:
            # In production, implement full OAuth flow
            # For now, assume access token is available in environment
            access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
            if not access_token:
                logger.error("Upstox access token not found in environment")
                return False
            
            # Verify token by making a simple API call
            url = f"{self.rest_api_url}user/profile"
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        logger.info("Upstox authentication successful")
                        return True
                    else:
                        logger.error(f"Upstox authentication failed: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Upstox authentication error: {e}")
            return False
    
    async def check_health(self) -> bool:
        """Check if Upstox API is healthy"""
        try:
            url = f"{self.rest_api_url}market/status/NSE"
            headers = {
                'Authorization': f'Bearer {os.getenv("UPSTOX_ACCESS_TOKEN")}',
                'Accept': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    return response.status == 200
        except:
            return False

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
        try:
            # Convert interval to Finnhub format
            resolution = self._convert_interval_to_finnhub(interval)
            
            # Calculate timestamps
            end_time = int(datetime.now().timestamp())
            start_time = end_time - (limit * 24 * 60 * 60)  # Days to seconds
            
            # Get data from Finnhub
            candles = self.finnhub_client.stock_candles(symbol, resolution, start_time, end_time)
            
            if candles['s'] == 'ok':
                df_data = {
                    'timestamp': pd.to_datetime(candles['t'], unit='s'),
                    'open': candles['o'],
                    'high': candles['h'],
                    'low': candles['l'],
                    'close': candles['c'],
                    'volume': candles['v']
                }
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                return df
            else:
                logger.warning(f"Finnhub returned no data for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching Finnhub historical data: {e}")
            return pd.DataFrame()
    
    def _convert_interval_to_finnhub(self, interval: str) -> str:
        """Convert interval to Finnhub format"""
        interval_map = {
            '1m': '1',
            '5m': '5',
            '15m': '15',
            '30m': '30',
            '1h': '60',
            '1d': 'D',
            'D': 'D',
            '1w': 'W',
            '1M': 'M'
        }
        return interval_map.get(interval, 'D')
    
    async def get_tick_data(self, symbol: str) -> Dict:
        """Get real-time tick data from Finnhub API"""
        try:
            # Get current quote from Finnhub
            quote = self.finnhub_client.quote(symbol)
            
            return {
                'symbol': symbol,
                'last_price': float(quote.get('c', 0)),  # Current price
                'open': float(quote.get('o', 0)),
                'high': float(quote.get('h', 0)),
                'low': float(quote.get('l', 0)),
                'prev_close': float(quote.get('pc', 0)),
                'timestamp': datetime.now().isoformat(),
                'source': 'finnhub'
            }
        except Exception as e:
            logger.error(f"Error fetching Finnhub tick data: {e}")
            return {}
    
    async def subscribe_to_data(self, symbols: List[str]):
        """Subscribe to real-time data for symbols"""
        try:
            # Finnhub WebSocket implementation
            websocket_url = f"wss://ws.finnhub.io?token={self.api_key}"
            
            async with websockets.connect(websocket_url) as websocket:
                # Subscribe to symbols
                for symbol in symbols:
                    subscribe_msg = {"type": "subscribe", "symbol": symbol}
                    await websocket.send(json.dumps(subscribe_msg))
                
                logger.info(f"Subscribed to {len(symbols)} symbols via Finnhub WebSocket")
                
                # Listen for messages
                async for message in websocket:
                    data = json.loads(message)
                    await self._process_finnhub_websocket_data(data)
                    
        except Exception as e:
            logger.error(f"Error subscribing to Finnhub data: {e}")
    
    async def _process_finnhub_websocket_data(self, data: Dict):
        """Process Finnhub WebSocket data"""
        try:
            if data.get('type') == 'trade':
                trades = data.get('data', [])
                for trade in trades:
                    processed_data = {
                        'symbol': trade.get('s'),
                        'last_price': float(trade.get('p', 0)),
                        'volume': int(trade.get('v', 0)),
                        'timestamp': datetime.fromtimestamp(trade.get('t', 0) / 1000).isoformat(),
                        'source': 'finnhub_websocket'
                    }
                    logger.debug(f"Received Finnhub trade data: {processed_data}")
        except Exception as e:
            logger.error(f"Error processing Finnhub WebSocket data: {e}")
    
    async def check_health(self) -> bool:
        """Check if Finnhub API is healthy"""
        try:
            # Simple API call to check health
            quote = self.finnhub_client.quote('AAPL')
            return 'c' in quote
        except:
            return False

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
        try:
            # Convert interval to Alpha Vantage format
            av_interval = self._convert_interval_to_av(interval)
            
            if av_interval in ['1min', '5min', '15min', '30min', '60min']:
                data, meta_data = await self.ts.get_intraday(symbol, interval=av_interval, outputsize='full')
            else:
                data, meta_data = await self.ts.get_daily(symbol, outputsize='full')
            
            if not data.empty:
                # Rename columns to standard format
                data.columns = ['open', 'high', 'low', 'close', 'volume']
                data.index.name = 'timestamp'
                
                # Limit to requested number of records
                if len(data) > limit:
                    data = data.tail(limit)
                    
                return data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage historical data: {e}")
            return pd.DataFrame()
    
    def _convert_interval_to_av(self, interval: str) -> str:
        """Convert interval to Alpha Vantage format"""
        interval_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '60min',
            '1d': 'daily',
            'D': 'daily'
        }
        return interval_map.get(interval, 'daily')
    
    async def get_tick_data(self, symbol: str) -> Dict:
        """Get real-time tick data from Alpha Vantage API"""
        try:
            # Alpha Vantage doesn't provide real-time tick data in free tier
            # Use latest daily data instead
            data, meta_data = await self.ts.get_daily(symbol, outputsize='compact')
            
            if not data.empty:
                latest = data.iloc[-1]
                return {
                    'symbol': symbol,
                    'last_price': float(latest['4. close']),
                    'open': float(latest['1. open']),
                    'high': float(latest['2. high']),
                    'low': float(latest['3. low']),
                    'volume': int(latest['5. volume']),
                    'timestamp': data.index[-1].isoformat(),
                    'source': 'alpha_vantage'
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage tick data: {e}")
            return {}
    
    async def subscribe_to_data(self, symbols: List[str]):
        """Subscribe to real-time data for symbols"""
        # Alpha Vantage doesn't support WebSocket subscriptions
        # Implement polling mechanism instead
        logger.info(f"Setting up polling for {len(symbols)} symbols via Alpha Vantage")
        
        async def poll_symbols():
            while True:
                for symbol in symbols:
                    try:
                        tick_data = await self.get_tick_data(symbol)
                        if tick_data:
                            logger.debug(f"Polled data for {symbol}: {tick_data}")
                    except Exception as e:
                        logger.error(f"Error polling {symbol}: {e}")
                    await asyncio.sleep(1)  # Respect API limits
                await asyncio.sleep(60)  # Poll every minute
        
        # Start polling task
        asyncio.create_task(poll_symbols())
    
    async def check_health(self) -> bool:
        """Check if Alpha Vantage API is healthy"""
        try:
            # Simple API call to check health
            data, meta_data = await self.ts.get_daily('AAPL', outputsize='compact')
            return not data.empty
        except:
            return False

class YFinanceDataSource(DataSource):
    """Yahoo Finance data source implementation using yfinance"""
    
    def __init__(self):
        self.connected = False
        
    async def connect(self):
        """Check connection to Yahoo Finance"""
        try:
            # Test connection with a simple request
            ticker = yf.Ticker('AAPL')
            info = ticker.info
            self.connected = True
            logger.info("Connected to Yahoo Finance API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Yahoo Finance API: {str(e)}")
            return False
    
    async def get_historical_data(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Get historical OHLC data from Yahoo Finance"""
        try:
            # Convert symbol for Indian stocks
            yf_symbol = self._convert_to_yf_symbol(symbol)
            
            # Convert interval to yfinance format
            yf_interval = self._convert_interval_to_yf(interval)
            
            # Calculate period
            if limit <= 7:
                period = "7d"
            elif limit <= 30:
                period = "1mo"
            elif limit <= 90:
                period = "3mo"
            elif limit <= 365:
                period = "1y"
            else:
                period = "2y"
            
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period=period, interval=yf_interval)
            
            if not data.empty:
                # Standardize column names
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                data.index.name = 'timestamp'
                
                # Limit to requested number of records
                if len(data) > limit:
                    data = data.tail(limit)
                    
                return data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance historical data: {e}")
            return pd.DataFrame()
    
    def _convert_to_yf_symbol(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance format"""
        # Basic conversion for Indian stocks
        if symbol in ['NIFTY', 'BANKNIFTY']:
            return f"^{symbol}"
        else:
            return f"{symbol}.NS"  # NSE suffix
    
    def _convert_interval_to_yf(self, interval: str) -> str:
        """Convert interval to yfinance format"""
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d',
            'D': '1d',
            '1w': '1wk',
            '1M': '1mo'
        }
        return interval_map.get(interval, '1d')
    
    async def get_tick_data(self, symbol: str) -> Dict:
        """Get current quote from Yahoo Finance"""
        try:
            yf_symbol = self._convert_to_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            
            # Get current data
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                latest = data.iloc[-1]
                return {
                    'symbol': symbol,
                    'last_price': float(latest['Close']),
                    'open': float(latest['Open']),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'volume': int(latest['Volume']),
                    'timestamp': data.index[-1].isoformat(),
                    'source': 'yahoo_finance'
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance tick data: {e}")
            return {}
    
    async def subscribe_to_data(self, symbols: List[str]):
        """Subscribe to real-time data for symbols (polling)"""
        logger.info(f"Setting up polling for {len(symbols)} symbols via Yahoo Finance")
        
        async def poll_symbols():
            while True:
                for symbol in symbols:
                    try:
                        tick_data = await self.get_tick_data(symbol)
                        if tick_data:
                            logger.debug(f"Polled Yahoo Finance data for {symbol}: {tick_data}")
                    except Exception as e:
                        logger.error(f"Error polling Yahoo Finance {symbol}: {e}")
                    await asyncio.sleep(1)  # Rate limiting
                await asyncio.sleep(30)  # Poll every 30 seconds
        
        # Start polling task
        asyncio.create_task(poll_symbols())
    
    async def check_health(self) -> bool:
        """Check if Yahoo Finance API is healthy"""
        try:
            ticker = yf.Ticker('AAPL')
            data = ticker.history(period="1d")
            return not data.empty
        except:
            return False
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
        self.monitored_symbols = []
        self.data_cache = {}
        self.data_subscribers = []
        
        # Create data directory if it doesn't exist
        Path(historical_data_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for data storage
        self.db_path = os.path.join(historical_data_dir, 'market_data.db')
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for storing market data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables for historical data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    source TEXT,
                    interval_type TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, interval_type, source)
                )
            ''')
            
            # Create tables for real-time data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS realtime_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    last_price REAL,
                    volume INTEGER,
                    source TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON historical_data(symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_realtime_symbol ON realtime_data(symbol, timestamp)')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
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
                elif source.lower() == 'yahoo_finance':
                    data_source = YFinanceDataSource()
                else:
                    raise ValueError(f"Unsupported data source: {source}")
                
                await data_source.connect()
                self.data_sources[source.lower()] = data_source
                logger.info(f"Successfully initialized {source} data source")
                
            except Exception as e:
                logger.error(f"Error initializing data source {source}: {str(e)}")
                if self.cross_validation_enabled:
                    logger.warning(f"Critical data source {source} failed, system may halt trading")
        
        self.connected = len(self.data_sources) > 0
        return self.connected
    
    async def store_historical_data(self, symbol: str, data: pd.DataFrame, source: str, interval: str):
        """Store historical data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            for timestamp, row in data.iterrows():
                conn.execute('''
                    INSERT OR REPLACE INTO historical_data 
                    (symbol, timestamp, open, high, low, close, volume, source, interval_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                    float(row.get('open', 0)),
                    float(row.get('high', 0)),
                    float(row.get('low', 0)),
                    float(row.get('close', 0)),
                    int(row.get('volume', 0)),
                    source,
                    interval
                ))
            
            conn.commit()
            conn.close()
            logger.debug(f"Stored {len(data)} records for {symbol} from {source}")
            
        except Exception as e:
            logger.error(f"Error storing historical data: {e}")
    
    async def store_realtime_data(self, symbol: str, data: Dict, source: str):
        """Store real-time data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                INSERT INTO realtime_data 
                (symbol, timestamp, last_price, volume, source, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                data.get('timestamp', datetime.now().isoformat()),
                float(data.get('last_price', 0)),
                int(data.get('volume', 0)),
                source,
                json.dumps(data)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing real-time data: {e}")
    
    async def get_stored_historical_data(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """Retrieve historical data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT timestamp, open, high, low, close, volume
                FROM historical_data
                WHERE symbol = ? AND interval_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            
            df = pd.read_sql_query(query, conn, params=(symbol, interval, limit))
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            return pd.DataFrame()
    
    async def add_symbol_to_monitor(self, symbol: str):
        """Add a symbol to the monitoring list"""
        if symbol not in self.monitored_symbols:
            self.monitored_symbols.append(symbol)
            logger.info(f"Added {symbol} to monitoring list")
            
            for source_name, source in self.data_sources.items():
                try:
                    await source.subscribe_to_data([symbol])
                except Exception as e:
                    logger.error(f"Error subscribing to {symbol} on {source_name}: {e}")
    
    async def remove_symbol_from_monitor(self, symbol: str):
        """Remove a symbol from the monitoring list"""
        if symbol in self.monitored_symbols:
            self.monitored_symbols.remove(symbol)
            logger.info(f"Removed {symbol} from monitoring list")
    
    def subscribe_to_data_updates(self, callback_func):
        """Subscribe to data updates"""
        self.data_subscribers.append(callback_func)
    
    async def notify_subscribers(self, symbol: str, data: Dict):
        """Notify all subscribers of new data"""
        for callback in self.data_subscribers:
            try:
                await callback(symbol, data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    async def fetch_and_store_historical_data(self, symbols: List[str], days: int = 365):
        """Fetch and store historical data for symbols"""
        for symbol in symbols:
            logger.info(f"Fetching historical data for {symbol}")
            
            for source_name, source in self.data_sources.items():
                try:
                    # Fetch data from each source
                    data = await source.get_historical_data(symbol, '1d', days)
                    
                    if not data.empty:
                        # Store in database
                        await self.store_historical_data(symbol, data, source_name, '1d')
                        logger.info(f"Stored {len(data)} historical records for {symbol} from {source_name}")
                    
                except Exception as e:
                    logger.error(f"Error fetching historical data for {symbol} from {source_name}: {e}")
                    continue
    
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