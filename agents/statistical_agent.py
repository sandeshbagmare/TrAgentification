"""
Statistical & Pattern Agent for AI Trading System
Responsible for detecting mean reversion opportunities, volatility breakouts, and momentum bursts
"""
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from scipy.stats import zscore

logger = logging.getLogger(__name__)

class StatisticalSignal(ABC):
    """Abstract base class for statistical signals"""
    
    @abstractmethod
    async def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate the statistical signal"""
        pass
    
    @abstractmethod
    async def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate trading signal based on statistical patterns"""
        pass

class MeanReversionSignal(StatisticalSignal):
    """Mean reversion signal implementation"""
    
    def __init__(self, window: int = 20, zscore_threshold: float = 2.0):
        self.window = window  # Lookback window in days
        self.zscore_threshold = zscore_threshold  # Z-score threshold for mean reversion
        
    async def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate mean reversion metrics"""
        if data.empty:
            return {'success': False, 'error': 'No data to analyze'}
        
        result = data.copy()
        
        # Calculate rolling mean and standard deviation
        result['rolling_mean'] = result['close'].rolling(window=self.window).mean()
        result['rolling_std'] = result['close'].rolling(window=self.window).std()
        
        # Calculate Z-score
        result['zscore'] = zscore(result['close'][-self.window:])
        
        # Calculate Bollinger Band-like metrics
        result['distance_to_mean'] = (result['close'] - result['rolling_mean']) / result['rolling_std']
        
        return {
            'success': True,
            'signal_data': result
        }
    
    async def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate mean reversion trading signal"""
        if data.empty:
            return {'success': False, 'error': 'No data to analyze'}
        
        try:
            # Get latest data
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            
            # Calculate Z-score
            zscore_value = latest['zscore']
            
            # Generate signal based on Z-score
            if abs(zscore_value) >= self.zscore_threshold:
                if zscore_value < -self.zscore_threshold:  # Price below mean (buy signal)
                    return {
                        'signal': 1,
                        'confidence': 1.0 - (abs(zscore_value) - self.zscore_threshold) / 5,
                        'reason': f"Mean reversion detected: Z-score = {zscore_value:.2f}"
                    }
                elif zscore_value > self.zscore_threshold:  # Price above mean (sell signal)
                    return {
                        'signal': -1,
                        'confidence': 1.0 - (abs(zscore_value) - self.zscore_threshold) / 5,
                        'reason': f"Mean reversion detected: Z-score = {zscore_value:.2f}"
                    }
            
            # No mean reversion signal
            return {
                'signal': 0,
                'confidence': 0.0,
                'reason': "No mean reversion signal detected"
            }
            
        except Exception as e:
            logger.error(f"Error generating mean reversion signal: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

class VolatilityBreakoutSignal(StatisticalSignal):
    """Volatility breakout signal implementation"""
    
    def __init__(self, window: int = 20, volatility_threshold: float = 0.02):
        self.window = window  # Lookback window in days
        self.volatility_threshold = volatility_threshold  # Minimum volatility to trigger a signal
        
    async def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate volatility breakout metrics"""
        if data.empty:
            return {'success': False, 'error': 'No data to analyze'}
        
        result = data.copy()
        
        # Calculate daily returns
        result['returns'] = result['close'].pct_change()
        
        # Calculate rolling volatility (standard deviation of returns)
        result['volatility'] = result['returns'].rolling(window=self.window).std()
        
        # Calculate ATR (Average True Range)
        result['tr0'] = abs(result['high'] - result['low'])
        result['tr1'] = abs(result['high'] - result['close'].shift(1))
        result['tr2'] = abs(result['low'] - result['close'].shift(1))
        result['tr'] = result[['tr0', 'tr1', 'tr2']].max(axis=1)
        result['atr'] = result['tr'].rolling(window=self.window).mean()
        
        # Calculate volatility squeeze (Bollinger Bands width vs ATR)
        result['bb_upper'] = result['close'].rolling(window=self.window).mean() + 2 * result['close'].rolling(window=self.window).std()
        result['bb_lower'] = result['close'].rolling(window=self.window).mean() - 2 * result['close'].rolling(window=self.window).std()
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['close']
        result['squeeze'] = result['bb_width'] < result['atr']
        
        return {
            'success': True,
            'signal_data': result
        }
    
    async def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate volatility breakout trading signal"""
        if data.empty:
            return {'success': False, 'error': 'No data to analyze'}
        
        try:
            # Get latest data
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            
            # Calculate current volatility
            current_volatility = latest['volatility']
            previous_volatility = previous['volatility']
            
            # Generate signal based on volatility changes
            if current_volatility > self.volatility_threshold:
                # Check if volatility is increasing
                if current_volatility > previous_volatility:
                    # Volatility breakout with increasing volatility
                    
                    # Determine direction based on price movement
                    if latest['close'] > latest['open']:
                        # Bullish breakout
                        return {
                            'signal': 1,
                            'confidence': min(1.0, current_volatility / self.volatility_threshold),
                            'reason': f"Volatility breakout (increasing): {current_volatility * 100:.2f}% > threshold {self.volatility_threshold * 100:.2f}%"
                        }
                    else:
                        # Bearish breakout
                        return {
                            'signal': -1,
                            'confidence': min(1.0, current_volatility / self.volatility_threshold),
                            'reason': f"Volatility breakout (increasing): {current_volatility * 100:.2f}% > threshold {self.volatility_threshold * 100:.2f}%"
                        }
                else:
                    # Volatility is high but decreasing, less confidence
                    return {
                        'signal': 0,
                        'confidence': 0.3,
                        'reason': f"High volatility ({current_volatility * 100:.2f}%) but decreasing"
                    }
            
            # Check for volatility squeeze (potential breakout)
            if len(data) >= self.window:
                # Get historical volatility
                historical_volatility = data['volatility'][-self.window:].mean()
                
                # Check if we're in a volatility squeeze
                if latest['squeeze']:
                    # Calculate squeeze strength
                    squeeze_strength = (latest['atr'] - latest['bb_width']) / latest['atr']
                    
                    # Check if squeeze is releasing
                    if previous['squeeze'] and not latest['squeeze']:
                        # Potential breakout in the direction of recent price movement
                        if latest['close'] > latest['bb_upper']:  # Breakout above upper band
                            return {
                                'signal': 1,
                                'confidence': 0.7 + (squeeze_strength * 0.3),
                                'reason': f"Volatility squeeze breakout above upper band: {squeeze_strength * 100:.2f}% squeeze strength"
                            }
                        elif latest['close'] < latest['bb_lower']:  # Breakout below lower band
                            return {
                                'signal': -1,
                                'confidence': 0.7 + (squeeze_strength * 0.3),
                                'reason': f"Volatility squeeze breakout below lower band: {squeeze_strength * 100:.2f}% squeeze strength"
                            }
                        else:
                            # Squeeze but no breakout yet
                            return {
                                'signal': 0,
                                'confidence': 0.4,
                                'reason': f"Volatility squeeze detected ({squeeze_strength * 100:.2f}% strength)"
                            }
            
            # No volatility breakout signal
            return {
                'signal': 0,
                'confidence': 0.0,
                'reason': "No volatility breakout signal detected"
            }
            
        except Exception as e:
            logger.error(f"Error generating volatility breakout signal: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

class MomentumSignal(StatisticalSignal):
    """Momentum signal implementation"""
    
    def __init__(self, window: int = 20, momentum_threshold: float = 0.02):
        self.window = window  # Lookback window in days
        self.momentum_threshold = momentum_threshold  # Minimum momentum to generate signal
        
    async def calculate(self, data: pd.DataFrame) -> Dict:
        """Calculate momentum metrics"""
        if data.empty:
            return {'success': False, 'error': 'No data to analyze'}
        
        result = data.copy()
        
        # Calculate simple momentum
        result['momentum'] = result['close'].pct_change(periods=self.window)
        
        # Calculate rate of change
        result['roc'] = result['close'].pct_change(periods=5)  # 5-day rate of change
        
        # Calculate relative strength index (RSI) - simplified
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        return {
            'success': True,
            'signal_data': result
        }
    
    async def generate_signal(self, data: pd.DataFrame) -> Dict:
        """Generate momentum trading signal"""
        if data.empty:
            return {'success': False, 'error': 'No data to analyze'}
        
        try:
            # Get latest data
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            
            # Calculate momentum strength
            momentum_strength = abs(latest['momentum'])
            
            # Generate signal based on momentum
            if momentum_strength >= self.momentum_threshold:
                # Strong momentum
                if latest['momentum'] > 0:  # Upward momentum
                    return {
                        'signal': 1,
                        'confidence': min(1.0, momentum_strength / self.momentum_threshold),
                        'reason': f"Strong upward momentum: {latest['momentum'] * 100:.2f}%"
                    }
                else:  # Downward momentum
                    return {
                        'signal': -1,
                        'confidence': min(1.0, momentum_strength / self.momentum_threshold),
                        'reason': f"Strong downward momentum: {latest['momentum'] * 100:.2f}%"
                    }
            
            # Check for momentum divergence
            # This would typically be done in conjunction with technical analysis agent
            # For simplicity, we'll include a basic divergence check
            
            # Calculate if price is making new highs but momentum is decreasing
            if latest['close'] > data['close'].rolling(window=self.window).max().iloc[-2] and 
               latest['momentum'] < data['momentum'].rolling(window=self.window).max().iloc[-2]:
                # Bullish divergence (price higher, momentum weaker)
                return {
                    'signal': 0,
                    'confidence': 0.6,
                    'reason': "Momentum divergence detected (price higher, momentum weaker)"
                }
            
            # Calculate if price is making new lows but momentum is decreasing
            if latest['close'] < data['close'].rolling(window=self.window).min().iloc[-2] and 
               latest['momentum'] > data['momentum'].rolling(window=self.window).min().iloc[-2]:
                # Bearish divergence (price lower, momentum stronger)
                return {
                    'signal': 0,
                    'confidence': 0.6,
                    'reason': "Momentum divergence detected (price lower, momentum stronger)"
                }
            
            # No momentum signal
            return {
                'signal': 0,
                'confidence': 0.0,
                'reason': "No momentum signal detected"
            }
            
        except Exception as e:
            logger.error(f"Error generating momentum signal: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

class PairTradingSignal(StatisticalSignal):
    """Pair trading signal implementation"""
    
    def __init__(self, lookback_window: int = 30, 
                 entry_zscore: float = 2.0, 
                 exit_zscore: float = 0.5):
        self.lookback_window = lookback_window
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.pair_correlations = {}  # Store pair correlations
        
    async def calculate(self, data: Dict[pd.DataFrame]) -> Dict:
        """Calculate pair trading metrics for multiple instruments"""
        if not data or not isinstance(data, dict):
            return {'success': False, 'error': 'Invalid data format'}
        
        result = {}
        
        # Calculate correlations between all pairs
        for symbol1, df1 in data.items():
            for symbol2, df2 in data.items():
                if symbol1 == symbol2:
                    continue
                
                # Find common dates between the two time series
                common_dates = df1.index.intersection(df2.index)
                
                if len(common_dates) < self.lookback_window:
                    continue
                
                # Calculate returns
                returns1 = df1.loc[common_dates, 'close'].pct_change()
                returns2 = df2.loc[common_dates, 'close'].pct_change()
                
                # Calculate correlation
                correlation = np.corrcoef(returns1[1:], returns2[1:])[0, 1]
                
                # Store correlation
                if symbol1 not in self.pair_correlations:
                    self.pair_correlations[symbol1] = {}
                self.pair_correlations[symbol1][symbol2] = correlation
                
                # Calculate spread
                spread = df1.loc[common_dates, 'close'] - df2.loc[common_dates, 'close']
                
                # Calculate z-score of spread
                zscore_spread = zscore(spread)
                
                # Store signal data
                result[f"{symbol1}_{symbol2}"] = {
                    'correlation': correlation,
                    'spread': spread.values,
                    'zscore': zscore_spread
                }
        
        return {
            'success': True,
            'signal_data': result
        }
    
    async def generate_signal(self, data: Dict[pd.DataFrame], 
                             primary_symbol: str, 
                             secondary_symbol: str) -> Dict:
        """Generate pair trading signal based on cointegration and z-score"""
        if not data or not isinstance(data, dict):
            return {'success': False, 'error': 'Invalid data format'}
        
        if primary_symbol not in data or secondary_symbol not in data:
            return {'success': False, 'error': 'Missing data for one or both symbols'}
        
        try:
            # Get correlation between the pair
            correlation = self.pair_correlations.get(primary_symbol, {}).get(secondary_symbol, 0)
            
            # Only consider pairs with strong correlation
            if abs(correlation) < 0.7:  # Minimum correlation threshold
                return {
                    'signal': 0,
                    'confidence': 0.0,
                    'reason': f"Low correlation between {primary_symbol} and {secondary_symbol} ({abs(correlation):.2f} < 0.7)"
                }
            
            # Get the spread data for this pair
            spread_data = data[primary_symbol]['close'] - data[secondary_symbol]['close']
            zscore_spread = zscore(spread_data)
            latest_zscore = zscore_spread[-1]
            previous_zscore = zscore_spread[-2]
            
            # Check for entry conditions
            if abs(latest_zscore) >= self.entry_zscore:
                # Strong deviation from mean
                if latest_zscore > self.entry_zscore and previous_zscore <= self.entry_zscore:
                    # Spread above threshold (sell primary, buy secondary)
                    return {
                        'signal': -1,
                        'confidence': min(1.0, abs(latest_zscore) / self.entry_zscore),
                        'reason': f"Pair spread above threshold: Z-score = {latest_zscore:.2f}"
                    }
                elif latest_zscore < -self.entry_zscore and previous_zscore >= -self.entry_zscore:
                    # Spread below threshold (buy primary, sell secondary)
                    return {
                        'signal': 1,
                        'confidence': min(1.0, abs(latest_zscore) / self.entry_zscore),
                        'reason': f"Pair spread below threshold: Z-score = {latest_zscore:.2f}"
                    }
            
            # Check for exit conditions
            if abs(latest_zscore) <= self.exit_zscore:
                # Spread has reverted to mean
                return {
                    'signal': 0,  # Exit signal
                    'confidence': 1.0 - abs(latest_zscore) / self.entry_zscore,
                    'reason': f"Pair spread reverted to mean: Z-score = {latest_zscore:.2f}"
                }
            
            # No signal
            return {
                'signal': 0,
                'confidence': 0.0,
                'reason': "No pair trading signal detected"
            }
            
        except Exception as e:
            logger.error(f"Error generating pair trading signal: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

class StatisticalAgent:
    """
    Statistical Agent
    Detects mean reversion opportunities, volatility breakouts, and momentum bursts
    """
    
    def __init__(self, strategies: List[str] = ['mean_reversion', 'volatility', 'momentum', 'pair_trading']):
        self.strategies = strategies
        self.available_strategies = {
            'mean_reversion': MeanReversionSignal(window=20, zscore_threshold=2.0),
            'volatility': VolatilityBreakoutSignal(window=20, volatility_threshold=0.02),
            'momentum': MomentumSignal(window=20, momentum_threshold=0.02),
            'pair_trading': PairTradingSignal(lookback_window=30, entry_zscore=2.0, exit_zscore=0.5)
        }
        self.active_strategies = {}
        self.signal_threshold = 0.7  # Minimum confidence threshold for a signal
        
        # Initialize active strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize the requested statistical strategies"""
        for strategy in self.strategies:
            if strategy.lower() in self.available_strategies:
                self.active_strategies[strategy.lower()] = self.available_strategies[strategy.lower()]
            else:
                logger.warning(f"Unsupported statistical strategy: {strategy}")
    
    async def analyze_data(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Analyze data using all active statistical strategies"""
        if data.empty:
            logger.warning("No data to analyze")
            return {}
        
        analysis_result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'signals': {},
            'strategy_values': {},
            'overall_signal': 0,
            'confidence': 0.0,
            'details': {}
        }
        
        # Calculate for all active strategies
        for name, strategy in self.active_strategies.items():
            try:
                # Calculate strategy metrics
                calculation_result = await strategy.calculate(data)
                
                if calculation_result['success']:
                    # Store strategy values
                    analysis_result['strategy_values'][name] = calculation_result['signal_data'].iloc[-1].to_dict()
                
                # Generate signal
                signal = await strategy.generate_signal(data)
                
                if 'signal' in signal:
                    analysis_result['signals'][name] = signal
                
                # Store detailed analysis
                analysis_result['details'][name] = {
                    'value': calculation_result.get('signal_data', {}).iloc[-1].to_dict() if calculation_result.get('signal_data') is not None else {},
                    'signal': signal
                }
            except Exception as e:
                logger.error(f"Error analyzing with {name} strategy: {str(e)}")
                continue
        
        # Generate overall signal based on all strategies
        analysis_result.update(self._generate_overall_signal(analysis_result['signals']))
        
        return analysis_result
    
    def _generate_overall_signal(self, signals: Dict) -> Dict:
        """Generate overall trading signal based on all statistical strategies"""
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        confidence = 0.0
        
        # Count buy and sell signals from strategies
        for strategy, signal_info in signals.items():
            if signal_info.get('signal') == 1:
                buy_signals += 1
                confidence += signal_info.get('confidence', 0.5)
            elif signal_info.get('signal') == -1:
                sell_signals += 1
                confidence += signal_info.get('confidence', 0.5)
            
            if signal_info.get('signal') != 0:
                total_signals += 1
        
        # Calculate confidence (average of strategy confidences)
        confidence = confidence / total_signals if total_signals > 0 else 0
        
        # Generate overall signal
        if confidence >= self.signal_threshold:
            if buy_signals > sell_signals:
                return {
                    'overall_signal': 1,
                    'confidence': confidence,
                    'message': f"Strong buy signal ({buy_signals}/{total_signals} strategies agree)"
                }
            elif sell_signals > buy_signals:
                return {
                    'overall_signal': -1,
                    'confidence': confidence,
                    'message': f"Strong sell signal ({sell_signals}/{total_signals} strategies agree)"
                }
        
        # No strong consensus
        return {
            'overall_signal': 0,
            'confidence': confidence,
            'message': "No strong consensus among statistical strategies"
        }
    
    async def detect_pairs(self, symbols: List[str], data: Dict[pd.DataFrame]) -> Dict:
        """Detect correlated pairs for pair trading strategy"""
        if not symbols or not data:
            return {'success': False, 'error': 'Missing symbols or data'}
        
        # Calculate correlations between all pairs
        correlations = {}
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if i >= j or symbol1 not in data or symbol2 not in data:
                    continue
                
                try:
                    # Ensure both dataframes have the same date range
                    df1 = data[symbol1].copy()
                    df2 = data[symbol2].copy()
                    
                    # Find common indices
                    common_idx = df1.index.intersection(df2.index)
                    
                    if len(common_idx) < 30:  # Minimum 30 data points
                        continue
                    
                    # Align data
                    aligned_df1 = df1.loc[common_idx]
                    aligned_df2 = df2.loc[common_idx]
                    
                    # Calculate returns
                    returns1 = aligned_df1['close'].pct_change().dropna()
                    returns2 = aligned_df2['close'].pct_change().dropna()
                    
                    if len(returns1) < 20 or len(returns2) < 20:
                        continue
                    
                    # Calculate correlation
                    correlation = returns1.corr(returns2)
                    
                    # Store correlation if above threshold
                    if abs(correlation) > 0.7 and not pd.isna(correlation):
                        pair = f"{symbol1}_{symbol2}"
                        correlations[pair] = {
                            'correlation': float(correlation),
                            'symbols': [symbol1, symbol2],
                            'confidence': float(abs(correlation))
                        }
                except Exception as e:
                    logger.error(f"Error calculating correlation for {symbol1}-{symbol2}: {e}")
                    continue
        
        # Sort by correlation strength
        sorted_correlations = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        )
        
        # Return top 5 correlated pairs
        return {
            'success': True,
            'correlated_pairs': dict(sorted_correlations[:5])
        }
    
    async def analyze_market_regime(self, data: pd.DataFrame) -> Dict:
        """Analyze the current market regime (trending, mean-reverting, volatile)"""
        if len(data) < 50:
            return {'regime': 'unknown', 'confidence': 0.0, 'reason': 'Insufficient data'}
        
        try:
            # Calculate various regime indicators
            returns = data['close'].pct_change().dropna()
            
            # 1. Hurst Exponent for mean reversion vs trending
            hurst_exp = self._calculate_hurst_exponent(returns)
            
            # 2. Volatility clustering (GARCH effects)
            volatility = returns.rolling(window=20).std()
            vol_clustering = volatility.autocorr(lag=1)
            
            # 3. Trend strength using ADX-like calculation
            trend_strength = self._calculate_trend_strength(data)
            
            # 4. Market efficiency (randomness test)
            efficiency = self._calculate_market_efficiency(returns)
            
            # Determine regime based on indicators
            if hurst_exp < 0.4:
                regime = 'mean_reverting'
                confidence = (0.5 - hurst_exp) * 2
            elif hurst_exp > 0.6:
                regime = 'trending'
                confidence = (hurst_exp - 0.5) * 2
            else:
                regime = 'random_walk'
                confidence = 1 - abs(hurst_exp - 0.5) * 2
            
            # Adjust confidence based on trend strength
            if trend_strength > 0.7:
                if regime == 'trending':
                    confidence = min(1.0, confidence + 0.2)
                else:
                    confidence = max(0.0, confidence - 0.2)
            
            return {
                'regime': regime,
                'confidence': confidence,
                'hurst_exponent': hurst_exp,
                'trend_strength': trend_strength,
                'volatility_clustering': vol_clustering,
                'market_efficiency': efficiency,
                'reason': f'Hurst: {hurst_exp:.3f}, Trend: {trend_strength:.3f}'
            }
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return {'regime': 'unknown', 'confidence': 0.0, 'reason': str(e)}
    
    def _calculate_hurst_exponent(self, returns: pd.Series, lags: List[int] = None) -> float:
        """Calculate Hurst exponent to determine mean reversion vs trending behavior"""
        if lags is None:
            lags = range(2, min(100, len(returns) // 4))
        
        try:
            # R/S analysis
            rs_values = []
            for lag in lags:
                if lag >= len(returns):
                    continue
                
                # Divide series into chunks
                chunks = [returns[i:i+lag] for i in range(0, len(returns)-lag+1, lag)]
                rs_chunk_values = []
                
                for chunk in chunks:
                    if len(chunk) < lag:
                        continue
                    
                    # Calculate mean
                    mean_chunk = chunk.mean()
                    
                    # Calculate cumulative deviations
                    cumulative_deviations = (chunk - mean_chunk).cumsum()
                    
                    # Calculate range
                    R = cumulative_deviations.max() - cumulative_deviations.min()
                    
                    # Calculate standard deviation
                    S = chunk.std()
                    
                    if S > 0:
                        rs_chunk_values.append(R / S)
                
                if rs_chunk_values:
                    rs_values.append((lag, np.mean(rs_chunk_values)))
            
            if len(rs_values) < 5:
                return 0.5  # Default to random walk
            
            # Linear regression to find Hurst exponent
            lags_log = [np.log(lag) for lag, _ in rs_values]
            rs_log = [np.log(rs) for _, rs in rs_values if rs > 0]
            
            if len(lags_log) != len(rs_log) or len(rs_log) < 5:
                return 0.5
            
            # Simple linear regression
            n = len(lags_log)
            sum_x = sum(lags_log)
            sum_y = sum(rs_log)
            sum_xy = sum(x * y for x, y in zip(lags_log, rs_log))
            sum_x2 = sum(x * x for x in lags_log)
            
            hurst = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Clamp to reasonable range
            return max(0.0, min(1.0, hurst))
            
        except Exception as e:
            logger.error(f"Error calculating Hurst exponent: {e}")
            return 0.5
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using directional movement"""
        try:
            # Calculate directional movement
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            dm_plus = high.diff()
            dm_minus = low.diff() * -1
            
            dm_plus[dm_plus < 0] = 0
            dm_minus[dm_minus < 0] = 0
            
            # Smooth with 14-period average
            tr_smooth = tr.rolling(window=14).mean()
            dm_plus_smooth = dm_plus.rolling(window=14).mean()
            dm_minus_smooth = dm_minus.rolling(window=14).mean()
            
            # Directional Indicators
            di_plus = 100 * (dm_plus_smooth / tr_smooth)
            di_minus = 100 * (dm_minus_smooth / tr_smooth)
            
            # ADX calculation
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
            adx = dx.rolling(window=14).mean()
            
            return adx.iloc[-1] / 100.0 if not pd.isna(adx.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _calculate_market_efficiency(self, returns: pd.Series) -> float:
        """Calculate market efficiency using variance ratio test"""
        try:
            if len(returns) < 50:
                return 0.5
            
            # Calculate variance ratio
            k = 2  # lag
            n = len(returns)
            
            # Variance of k-period returns
            k_returns = returns.rolling(window=k).sum().dropna()
            var_k = k_returns.var()
            
            # Variance of 1-period returns
            var_1 = returns.var()
            
            # Variance ratio
            if var_1 > 0:
                variance_ratio = var_k / (k * var_1)
                
                # Convert to efficiency score (1 = perfectly efficient/random)
                efficiency = abs(1 - variance_ratio)
                return max(0.0, min(1.0, 1 - efficiency))
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating market efficiency: {e}")
            return 0.5
    
    async def detect_anomalies(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """Detect statistical anomalies in price and volume data"""
        try:
            anomalies = {
                'price_anomalies': [],
                'volume_anomalies': [],
                'return_anomalies': []
            }
            
            # Price anomalies (using z-score)
            price_zscore = zscore(data['close'])
            price_anomaly_threshold = 2.5
            
            for i, z in enumerate(price_zscore):
                if abs(z) > price_anomaly_threshold:
                    anomalies['price_anomalies'].append({
                        'index': i,
                        'timestamp': data.index[i].isoformat() if hasattr(data.index[i], 'isoformat') else str(data.index[i]),
                        'price': float(data['close'].iloc[i]),
                        'zscore': float(z),
                        'severity': 'high' if abs(z) > 3 else 'medium'
                    })
            
            # Volume anomalies
            volume_zscore = zscore(data['volume'])
            volume_anomaly_threshold = 2.0
            
            for i, z in enumerate(volume_zscore):
                if abs(z) > volume_anomaly_threshold:
                    anomalies['volume_anomalies'].append({
                        'index': i,
                        'timestamp': data.index[i].isoformat() if hasattr(data.index[i], 'isoformat') else str(data.index[i]),
                        'volume': int(data['volume'].iloc[i]),
                        'zscore': float(z),
                        'severity': 'high' if abs(z) > 3 else 'medium'
                    })
            
            # Return anomalies
            returns = data['close'].pct_change().dropna()
            return_zscore = zscore(returns)
            return_anomaly_threshold = 2.0
            
            for i, z in enumerate(return_zscore):
                if abs(z) > return_anomaly_threshold:
                    anomalies['return_anomalies'].append({
                        'index': i + 1,  # +1 because returns start from index 1
                        'timestamp': data.index[i + 1].isoformat() if hasattr(data.index[i + 1], 'isoformat') else str(data.index[i + 1]),
                        'return': float(returns.iloc[i]),
                        'zscore': float(z),
                        'severity': 'high' if abs(z) > 3 else 'medium'
                    })
            
            return {
                'success': True,
                'anomalies': anomalies,
                'summary': {
                    'total_price_anomalies': len(anomalies['price_anomalies']),
                    'total_volume_anomalies': len(anomalies['volume_anomalies']),
                    'total_return_anomalies': len(anomalies['return_anomalies'])
                }
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def start(self):
        """Start the statistical agent"""
        logger.info("Starting statistical agent...")
        logger.info(f"Active strategies: {list(self.active_strategies.keys())}")
        
        # Initialize any required resources
        return {
            'success': True,
            'message': 'Statistical agent started successfully',
            'active_strategies': list(self.active_strategies.keys())
        }
        
    async def emergency_stop(self):
        """Emergency stop for the statistical agent"""
        logger.info("Initiating emergency stop for statistical agent...")
        
        # Clean up resources
        self.active_strategies.clear()
        
        return {
            'success': True,
            'message': 'Statistical agent stopped successfully'
        }