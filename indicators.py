import numpy as np
import pandas as pd

class TechnicalIndicators:
    """
    Class for calculating various technical indicators for stock analysis
    """
    def __init__(self, data):
        """
        Initialize with stock data dataframe
        
        Args:
            data (pd.DataFrame): DataFrame containing at least 'Close' price data
        """
        self.data = data.copy()
        
    def add_all_indicators(self):
        """Add all available indicators to the dataframe"""
        self.add_sma()
        self.add_ema()
        self.add_macd()
        self.add_rsi()
        self.add_bollinger_bands()
        self.add_atr()
        self.add_stochastic_oscillator()
        self.add_obv()
        return self.data
        
    def add_sma(self, periods=[20, 50, 200]):
        """
        Add Simple Moving Average for given periods
        
        Args:
            periods (list): List of periods to calculate SMA for
        """
        for period in periods:
            self.data[f'SMA_{period}'] = self.data['Close'].rolling(window=period).mean()
        return self.data
        
    def add_ema(self, periods=[12, 26, 50]):
        """
        Add Exponential Moving Average for given periods
        
        Args:
            periods (list): List of periods to calculate EMA for
        """
        for period in periods:
            self.data[f'EMA_{period}'] = self.data['Close'].ewm(span=period, adjust=False).mean()
        return self.data
    
    def add_macd(self, fast=12, slow=26, signal=9):
        """
        Add MACD (Moving Average Convergence Divergence) indicator
        
        Args:
            fast (int): Fast period
            slow (int): Slow period
            signal (int): Signal period
        """
        # Calculate EMA
        ema_fast = self.data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line and signal line
        self.data['MACD_line'] = ema_fast - ema_slow
        self.data['MACD_signal'] = self.data['MACD_line'].ewm(span=signal, adjust=False).mean()
        self.data['MACD_histogram'] = self.data['MACD_line'] - self.data['MACD_signal']
        
        return self.data
    
    def add_rsi(self, period=14):
        """
        Add Relative Strength Index (RSI) indicator
        
        Args:
            period (int): RSI calculation period
        """
        delta = self.data['Close'].diff()
        
        # Make two series: one for gains and one for losses
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        # Calculate average gain and loss over the period
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI (Relative Strength Index)
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        return self.data
    
    def add_bollinger_bands(self, period=20, std_dev=2):
        """
        Add Bollinger Bands indicator
        
        Args:
            period (int): Period for moving average
            std_dev (int): Number of standard deviations for bands
        """
        self.data['BB_middle'] = self.data['Close'].rolling(window=period).mean()
        self.data['BB_std'] = self.data['Close'].rolling(window=period).std()
        
        self.data['BB_upper'] = self.data['BB_middle'] + (self.data['BB_std'] * std_dev)
        self.data['BB_lower'] = self.data['BB_middle'] - (self.data['BB_std'] * std_dev)
        
        # Calculate %B (where price is in relation to the bands)
        self.data['%B'] = (self.data['Close'] - self.data['BB_lower']) / (self.data['BB_upper'] - self.data['BB_lower'])
        
        return self.data
    
    def add_atr(self, period=14):
        """
        Add Average True Range (ATR) indicator
        
        Args:
            period (int): Period for ATR calculation
        """
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        self.data['ATR'] = true_range.rolling(period).mean()
        return self.data
    
    def add_stochastic_oscillator(self, k_period=14, d_period=3):
        """
        Add Stochastic Oscillator
        
        Args:
            k_period (int): K period
            d_period (int): D period
        """
        # Calculate %K
        lowest_low = self.data['Low'].rolling(window=k_period).min()
        highest_high = self.data['High'].rolling(window=k_period).max()
        
        self.data['%K'] = 100 * ((self.data['Close'] - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (3-day SMA of %K)
        self.data['%D'] = self.data['%K'].rolling(window=d_period).mean()
        
        return self.data
    
    def add_obv(self):
        """Add On-Balance Volume (OBV) indicator"""
        obv = [0]
        
        for i in range(1, len(self.data)):
            if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                obv.append(obv[-1] + self.data['Volume'].iloc[i])
            elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i-1]:
                obv.append(obv[-1] - self.data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
                
        self.data['OBV'] = obv
        return self.data
    
    def get_indicator_values(self, indicator_name):
        """
        Return the values for a specific indicator
        
        Args:
            indicator_name (str): Name of the indicator
            
        Returns:
            pandas.Series or pandas.DataFrame: The indicator values
        """
        if indicator_name not in self.data.columns:
            raise ValueError(f"Indicator {indicator_name} not found in data")
        
        return self.data[indicator_name]