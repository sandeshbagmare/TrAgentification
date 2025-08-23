"""
Technical Analysis Agent for AI Trading System
Responsible for computing technical indicators and generating signals based on classical technical analysis
"""
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class TechnicalIndicator(ABC):
    """Abstract base class for technical indicators"""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate the technical indicator"""
        pass
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate trading signal based on the indicator"""
        pass

class MovingAverage(TechnicalIndicator):
    """Moving Average indicator implementation"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        result = data.copy()
        result['ma_fast'] = result['close'].rolling(window=self.fast_period).mean()
        result['ma_slow'] = result['close'].rolling(window=self.slow_period).mean()
        return result
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on moving average crossover"""
        if len(data) < self.slow_period:
            return {"signal": 0, "message": "Not enough data"}
        
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Golden cross (buy signal)
        if previous['ma_fast'] < previous['ma_slow'] and latest['ma_fast'] > latest['ma_slow']:
            return {"signal": 1, "message": "Golden cross detected"}
        # Death cross (sell signal)
        elif previous['ma_fast'] > previous['ma_slow'] and latest['ma_fast'] < latest['ma_slow']:
            return {"signal": -1, "message": "Death cross detected"}
        # No signal
        else:
            return {"signal": 0, "message": "No crossover detected"}

class RSI(TechnicalIndicator):
    """Relative Strength Index (RSI) indicator implementation"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI"""
        result = data.copy()
        
        # Calculate price changes
        delta = result['close'].diff()
        
        # Calculate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        result['rsi'] = rsi
        return result
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on RSI levels"""
        if len(data) < self.period:
            return {"signal": 0, "message": "Not enough data"}
        
        latest = data.iloc[-1]
        
        # Overbought (sell signal)
        if latest['rsi'] > 70:
            return {"signal": -1, "message": "RSI overbought"}
        # Oversold (buy signal)
        elif latest['rsi'] < 30:
            return {"signal": 1, "message": "RSI oversold"}
        # Neutral
        else:
            return {"signal": 0, "message": "RSI neutral"}

class MACD(TechnicalIndicator):
    """Moving Average Convergence Divergence (MACD) indicator implementation"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD"""
        result = data.copy()
        
        # Calculate EMAs
        result['ema_fast'] = result['close'].ewm(span=self.fast_period, adjust=False).mean()
        result['ema_slow'] = result['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        result['macd_line'] = result['ema_fast'] - result['ema_slow']
        result['macd_signal'] = result['macd_line'].ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate histogram
        result['macd_hist'] = result['macd_line'] - result['macd_signal']
        
        return result
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on MACD crossovers"""
        if len(data) < self.slow_period:
            return {"signal": 0, "message": "Not enough data"}
        
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Bullish crossover (buy signal)
        if previous['macd_line'] <= previous['macd_signal'] and latest['macd_line'] > latest['macd_signal']:
            return {"signal": 1, "message": "MACD bullish crossover"}
        # Bearish crossover (sell signal)
        elif previous['macd_line'] >= previous['macd_signal'] and latest['macd_line'] < latest['macd_signal']:
            return {"signal": -1, "message": "MACD bearish crossover"}
        # No signal
        else:
            return {"signal": 0, "message": "No MACD crossover detected"}

class BollingerBands(TechnicalIndicator):
    """Bollinger Bands indicator implementation"""
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        result = data.copy()
        
        # Calculate moving average and standard deviation
        result['ma'] = result['close'].rolling(window=self.period).mean()
        result['std'] = result['close'].rolling(window=self.period).std()
        
        # Calculate upper and lower bands
        result['upper_band'] = result['ma'] + (self.num_std * result['std'])
        result['lower_band'] = result['ma'] - (self.num_std * result['std'])
        
        return result
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on price touching Bollinger Bands"""
        if len(data) < self.period:
            return {"signal": 0, "message": "Not enough data"}
        
        latest = data.iloc[-1]
        
        # Price touching or breaking lower band (buy signal)
        if latest['close'] <= latest['lower_band']:
            return {"signal": 1, "message": "Price touching lower Bollinger Band"}
        # Price touching or breaking upper band (sell signal)
        elif latest['close'] >= latest['upper_band']:
            return {"signal": -1, "message": "Price touching upper Bollinger Band"}
        # Price in middle region (no signal)
        else:
            return {"signal": 0, "message": "Price in Bollinger Bands middle region"}

class ATR(TechnicalIndicator):
    """Average True Range (ATR) indicator implementation"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range"""
        result = data.copy()
        
        # Calculate True Range
        result['tr0'] = abs(result['high'] - result['low'])
        result['tr1'] = abs(result['high'] - result['close'].shift(1))
        result['tr2'] = abs(result['low'] - result['close'].shift(1))
        result['tr'] = result[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Calculate ATR
        result['atr'] = result['tr'].rolling(window=self.period).mean()
        
        return result
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on ATR volatility changes"""
        if len(data) < self.period:
            return {"signal": 0, "message": "Not enough data"}
        
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Increasing volatility (potential for larger moves)
        if latest['atr'] > previous['atr'] * 1.1:  # 10% increase
            return {"signal": 0, "message": "Increasing volatility detected"}
        # Decreasing volatility (potential for consolidation)
        elif latest['atr'] < previous['atr'] * 0.9:  # 10% decrease
            return {"signal": 0, "message": "Decreasing volatility detected"}
        # Stable volatility
        else:
            return {"signal": 0, "message": "Volatility stable"}

class Stochastic(TechnicalIndicator):
    """Stochastic Oscillator indicator implementation"""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, smooth: int = 3):
        self.k_period = k_period
        self.d_period = d_period
        self.smooth = smooth
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        result = data.copy()
        
        # Calculate %K
        low_min = result['low'].rolling(window=self.k_period).min()
        high_max = result['high'].rolling(window=self.k_period).max()
        result['stoch_k'] = 100 * (result['close'] - low_min) / (high_max - low_min)
        
        # Smooth %K if needed
        if self.smooth > 1:
            result['stoch_k'] = result['stoch_k'].rolling(window=self.smooth).mean()
        
        # Calculate %D (moving average of %K)
        result['stoch_d'] = result['stoch_k'].rolling(window=self.d_period).mean()
        
        return result
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on Stochastic levels and crossovers"""
        if len(data) < self.k_period:
            return {"signal": 0, "message": "Not enough data"}
        
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Oversold condition (buy signal)
        if latest['stoch_k'] < 20 and latest['stoch_d'] < 20:
            if previous['stoch_k'] <= previous['stoch_d'] and latest['stoch_k'] > latest['stoch_d']:
                return {"signal": 1, "message": "Stochastic bullish crossover in oversold region"}
        
        # Overbought condition (sell signal)
        elif latest['stoch_k'] > 80 and latest['stoch_d'] > 80:
            if previous['stoch_k'] >= previous['stoch_d'] and latest['stoch_k'] < latest['stoch_d']:
                return {"signal": -1, "message": "Stochastic bearish crossover in overbought region"}
        
        return {"signal": 0, "message": "No stochastic signal"}

class WilliamsR(TechnicalIndicator):
    """Williams %R indicator implementation"""
    
    def __init__(self, period: int = 14):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams %R"""
        result = data.copy()
        
        # Calculate highest high and lowest low over period
        high_max = result['high'].rolling(window=self.period).max()
        low_min = result['low'].rolling(window=self.period).min()
        
        # Calculate Williams %R
        result['williams_r'] = -100 * (high_max - result['close']) / (high_max - low_min)
        
        return result
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on Williams %R levels"""
        if len(data) < self.period:
            return {"signal": 0, "message": "Not enough data"}
        
        latest = data.iloc[-1]
        
        # Oversold (buy signal)
        if latest['williams_r'] <= -80:
            return {"signal": 1, "message": "Williams %R oversold"}
        # Overbought (sell signal)
        elif latest['williams_r'] >= -20:
            return {"signal": -1, "message": "Williams %R overbought"}
        # Neutral
        else:
            return {"signal": 0, "message": "Williams %R neutral"}

class CCI(TechnicalIndicator):
    """Commodity Channel Index (CCI) indicator implementation"""
    
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate CCI"""
        result = data.copy()
        
        # Calculate Typical Price
        result['tp'] = (result['high'] + result['low'] + result['close']) / 3
        
        # Calculate Simple Moving Average of Typical Price
        result['tp_sma'] = result['tp'].rolling(window=self.period).mean()
        
        # Calculate Mean Deviation
        result['mean_dev'] = result['tp'].rolling(window=self.period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        # Calculate CCI
        result['cci'] = (result['tp'] - result['tp_sma']) / (0.015 * result['mean_dev'])
        
        return result
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on CCI levels"""
        if len(data) < self.period:
            return {"signal": 0, "message": "Not enough data"}
        
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Oversold turning up (buy signal)
        if previous['cci'] <= -100 and latest['cci'] > -100:
            return {"signal": 1, "message": "CCI emerging from oversold"}
        # Overbought turning down (sell signal)
        elif previous['cci'] >= 100 and latest['cci'] < 100:
            return {"signal": -1, "message": "CCI emerging from overbought"}
        # Neutral
        else:
            return {"signal": 0, "message": "CCI neutral"}

class VWAP(TechnicalIndicator):
    """Volume Weighted Average Price (VWAP) indicator implementation"""
    
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP"""
        result = data.copy()
        
        # Calculate typical price
        result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3
        
        # Calculate volume * typical price
        result['vp'] = result['typical_price'] * result['volume']
        
        # Calculate VWAP
        result['vwap'] = result['vp'].rolling(window=self.period).sum() / result['volume'].rolling(window=self.period).sum()
        
        return result
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on price vs VWAP"""
        if len(data) < self.period:
            return {"signal": 0, "message": "Not enough data"}
        
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Price crossing above VWAP (buy signal)
        if previous['close'] <= previous['vwap'] and latest['close'] > latest['vwap']:
            return {"signal": 1, "message": "Price crossing above VWAP"}
        # Price crossing below VWAP (sell signal)
        elif previous['close'] >= previous['vwap'] and latest['close'] < latest['vwap']:
            return {"signal": -1, "message": "Price crossing below VWAP"}
        # No crossover
        else:
            return {"signal": 0, "message": "No VWAP crossover"}

class Ichimoku(TechnicalIndicator):
    """Ichimoku Kinko Hyo indicator implementation"""
    
    def __init__(self, conversion_period: int = 9, base_period: int = 26, span_b_period: int = 52, displacement: int = 26):
        self.conversion_period = conversion_period
        self.base_period = base_period
        self.span_b_period = span_b_period
        self.displacement = displacement
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku components"""
        result = data.copy()
        
        # Tenkan-sen (Conversion Line)
        period_9_high = result['high'].rolling(window=self.conversion_period).max()
        period_9_low = result['low'].rolling(window=self.conversion_period).min()
        result['tenkan_sen'] = (period_9_high + period_9_low) / 2
        
        # Kijun-sen (Base Line)
        period_26_high = result['high'].rolling(window=self.base_period).max()
        period_26_low = result['low'].rolling(window=self.base_period).min()
        result['kijun_sen'] = (period_26_high + period_26_low) / 2
        
        # Senkou Span A (Leading Span A)
        result['senkou_span_a'] = ((result['tenkan_sen'] + result['kijun_sen']) / 2).shift(self.displacement)
        
        # Senkou Span B (Leading Span B)
        period_52_high = result['high'].rolling(window=self.span_b_period).max()
        period_52_low = result['low'].rolling(window=self.span_b_period).min()
        result['senkou_span_b'] = ((period_52_high + period_52_low) / 2).shift(self.displacement)
        
        # Chikou Span (Lagging Span)
        result['chikou_span'] = result['close'].shift(-self.displacement)
        
        return result
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate signal based on Ichimoku cloud analysis"""
        if len(data) < self.span_b_period:
            return {"signal": 0, "message": "Not enough data"}
        
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Check if price is above or below cloud
        cloud_top = max(latest['senkou_span_a'], latest['senkou_span_b'])
        cloud_bottom = min(latest['senkou_span_a'], latest['senkou_span_b'])
        
        # Bullish signal: Tenkan above Kijun and price above cloud
        if (latest['tenkan_sen'] > latest['kijun_sen'] and 
            latest['close'] > cloud_top and 
            previous['tenkan_sen'] <= previous['kijun_sen']):
            return {"signal": 1, "message": "Ichimoku bullish signal"}
        
        # Bearish signal: Tenkan below Kijun and price below cloud
        elif (latest['tenkan_sen'] < latest['kijun_sen'] and 
              latest['close'] < cloud_bottom and 
              previous['tenkan_sen'] >= previous['kijun_sen']):
            return {"signal": -1, "message": "Ichimoku bearish signal"}
        
        return {"signal": 0, "message": "No clear Ichimoku signal"}

class TechnicalAnalysisAgent:
    """
    Technical Analysis Agent
    Responsible for computing classical technical indicators and generating signals
    """
    
    def __init__(self, indicators: List[str]):
        self.indicators = indicators
        self.available_indicators = {
            'ma': MovingAverage,
            'rsi': RSI,
            'macd': MACD,
            'bollinger': BollingerBands,
            'atr': ATR,
            'stochastic': Stochastic,
            'williams_r': WilliamsR,
            'cci': CCI,
            'vwap': VWAP,
            'ichimoku': Ichimoku
        }
        self.active_indicators = {}
        self.signal_threshold = 0.6  # 60% of indicators need to agree for a signal
        
        # Initialize indicators
        self._initialize_indicators()
    
    def _initialize_indicators(self):
        """Initialize the requested technical indicators"""
        for indicator in self.indicators:
            if indicator.lower() in self.available_indicators:
                self.active_indicators[indicator.lower()] = self.available_indicators[indicator.lower()]()
            else:
                logger.warning(f"Unsupported technical indicator: {indicator}")
    
    def analyze_data(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Analyze data using all active technical indicators"""
        if data.empty:
            logger.warning("No data to analyze")
            return {}
        
        analysis_result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signals': {},
            'indicator_values': {},
            'overall_signal': 0,
            'confidence': 0.0,
            'details': {}
        }
        
        # Calculate all active indicators
        for name, indicator in self.active_indicators.items():
            try:
                # Calculate indicator values
                indicator_data = indicator.calculate(data)
                analysis_result['indicator_values'][name] = indicator_data.iloc[-1].to_dict()
                
                # Generate signal
                signal = indicator.generate_signal(data)
                analysis_result['signals'][name] = signal
                
                # Store detailed analysis
                analysis_result['details'][name] = {
                    'value': indicator_data.iloc[-1].to_dict(),
                    'signal': signal
                }
            except Exception as e:
                logger.error(f"Error analyzing with {name} indicator: {str(e)}")
                continue
        
        # Generate overall signal based on all indicators
        analysis_result.update(self._generate_overall_signal(analysis_result['signals']))
        
        return analysis_result
    
    def _generate_overall_signal(self, signals: Dict) -> Dict:
        """Generate overall trading signal based on all indicator signals"""
        buy_signals = 0
        sell_signals = 0
        total_signals = len(signals)
        
        # Count buy and sell signals from indicators
        for indicator, signal_info in signals.items():
            if signal_info['signal'] == 1:
                buy_signals += 1
            elif signal_info['signal'] == -1:
                sell_signals += 1
        
        # Calculate confidence
        confidence = max(buy_signals, sell_signals) / total_signals if total_signals > 0 else 0
        
        # Generate overall signal
        if confidence >= self.signal_threshold:
            if buy_signals > sell_signals:
                return {
                    'overall_signal': 1,
                    'confidence': confidence,
                    'message': f"Strong buy signal ({buy_signals}/{total_signals} indicators agree)"
                }
            elif sell_signals > buy_signals:
                return {
                    'overall_signal': -1,
                    'confidence': confidence,
                    'message': f"Strong sell signal ({sell_signals}/{total_signals} indicators agree)"
                }
        
        # No strong consensus
        return {
            'overall_signal': 0,
            'confidence': confidence,
            'message': "No strong consensus among indicators"
        }
    
    async def start(self):
        """Start the technical analysis agent"""
        logger.info("Starting technical analysis agent...")
        # Implementation for starting the agent
        # This would typically involve:
        # - Connecting to data source
        # - Starting analysis loop
        # - Handling configuration updates
        
    async def emergency_stop(self):
        """Emergency stop for the technical analysis agent"""
        logger.info("Initiating emergency stop for technical analysis agent...")
        # Implementation for emergency stop procedures
        pass