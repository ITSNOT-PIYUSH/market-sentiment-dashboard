import pandas as pd 
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from indicators import TechnicalIndicators


class StockChart:
    def __init__(self, data):
        self.data = data.copy()  # Make a copy to avoid modifying the original data
        # Create 3 rows for price, indicators, and volume
        self.fig = make_subplots(rows=3, cols=1, 
                                vertical_spacing=0.1, 
                                shared_xaxes=True,
                                row_heights=[0.5, 0.3, 0.2])
        self.indicators = TechnicalIndicators(self.data)
        self.use_candlestick = False  # Default to line chart

    def set_chart_type(self, use_candlestick=False):
        """Set whether to use candlestick charts or line charts"""
        self.use_candlestick = use_candlestick

    def add_price_chart(self):
        if self.use_candlestick:
            # Add candlestick chart
            self.fig.add_trace(go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='Price',
                increasing_line_color='#2B8308',  # Green for increasing candles
                decreasing_line_color='#9C1F0B'   # Red for decreasing candles
            ), row=1, col=1)
        else:
            # Original line charts
            self.fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Open'], name='Open Price', marker_color='#1F77B4'), row=1, col=1)
            self.fig.add_trace(go.Scatter(x=self.data.index, y=self.data['High'], name='High Price', marker_color='#9467BD'), row=1, col=1)
            self.fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Low'], name='Low Price', marker_color='#D62728'), row=1, col=1)
            self.fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Close'], name='Close Price', marker_color='#76B900'), row=1, col=1)
        
    def add_moving_averages(self, periods=[20, 50, 200]):
        """Add simple moving averages to price chart"""
        colors = ['orange', 'purple', 'cyan']
        
        # Update the data with calculated SMAs
        self.data = self.indicators.add_sma(periods)
        
        # Now add each SMA to the chart
        for i, period in enumerate(periods):
            # Check if the SMA column exists
            sma_column = f'SMA_{period}'
            if sma_column in self.data.columns:
                col = colors[i % len(colors)]
                self.fig.add_trace(
                    go.Scatter(
                        x=self.data.index, 
                        y=self.data[sma_column], 
                        name=f'SMA {period}', 
                        marker_color=col,
                        line=dict(width=1)
                    ), 
                    row=1, col=1
                )
    
    def add_bollinger_bands(self):
        """Add Bollinger Bands to price chart"""
        # Update the data with Bollinger bands
        self.data = self.indicators.add_bollinger_bands()
        
        # Add middle band
        self.fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['BB_middle'], 
                name='BB Middle', 
                marker_color='grey',
                line=dict(width=1, dash='dot')
            ), 
            row=1, col=1
        )
        
        # Add upper and lower bands
        self.fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['BB_upper'], 
                name='BB Upper', 
                marker_color='grey',
                line=dict(width=1),
                opacity=0.3
            ), 
            row=1, col=1
        )
        
        self.fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['BB_lower'], 
                name='BB Lower', 
                marker_color='grey',
                line=dict(width=1),
                opacity=0.3,
                fill='tonexty',  # Fill area between upper and lower bands
                fillcolor='rgba(128, 128, 128, 0.1)'
            ), 
            row=1, col=1
        )
    
    def add_rsi(self):
        """Add RSI indicator"""
        # Update the data with RSI
        self.data = self.indicators.add_rsi()
        
        self.fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['RSI'], 
                name='RSI', 
                marker_color='#17BECF'
            ), 
            row=2, col=1
        )
        
        # Add overbought/oversold lines
        self.fig.add_hline(y=30, line_dash='dash', line_color='limegreen', line_width=1, row=2, col=1)
        self.fig.add_hline(y=70, line_dash='dash', line_color='red', line_width=1, row=2, col=1)
        self.fig.update_yaxes(title_text='RSI', row=2, col=1)

    def add_macd(self):
        """Add MACD indicator"""
        # Update the data with MACD
        self.data = self.indicators.add_macd()
        
        # Add MACD line and signal line
        self.fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['MACD_line'], 
                name='MACD', 
                marker_color='blue'
            ), 
            row=2, col=1
        )
        
        self.fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['MACD_signal'], 
                name='Signal', 
                marker_color='red'
            ), 
            row=2, col=1
        )
        
        # Add histogram
        colors = ['green' if val >= 0 else 'red' for val in self.data['MACD_histogram']]
        self.fig.add_trace(
            go.Bar(
                x=self.data.index, 
                y=self.data['MACD_histogram'], 
                name='Histogram',
                marker_color=colors,
                opacity=0.6
            ), 
            row=2, col=1
        )
        
        self.fig.update_yaxes(title_text='MACD', row=2, col=1)
    
    def add_stochastic(self):
        """Add Stochastic Oscillator"""
        # Update the data with Stochastic
        self.data = self.indicators.add_stochastic_oscillator()
        
        self.fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['%K'], 
                name='%K', 
                marker_color='blue'
            ), 
            row=2, col=1
        )
        
        self.fig.add_trace(
            go.Scatter(
                x=self.data.index, 
                y=self.data['%D'], 
                name='%D', 
                marker_color='red'
            ), 
            row=2, col=1
        )
        
        # Add overbought/oversold lines
        self.fig.add_hline(y=20, line_dash='dash', line_color='limegreen', line_width=1, row=2, col=1)
        self.fig.add_hline(y=80, line_dash='dash', line_color='red', line_width=1, row=2, col=1)
        self.fig.update_yaxes(title_text='Stochastic', row=2, col=1)

    def add_volume_chart(self):
        colors = ['#9C1F0B' if row['Open'] - row['Close'] >= 0 else '#2B8308' for index, row in self.data.iterrows()]
        self.fig.add_trace(go.Bar(x=self.data.index, y=self.data['Volume'], showlegend=False, marker_color=colors), row=3, col=1)
        self.fig.update_yaxes(title_text='Volume', row=3, col=1)

    def render_chart(self, title="Historical Price and Indicators"):
        self.fig.update_layout(title=title, height=800, margin=dict(l=0, r=10, b=10, t=30))
        st.plotly_chart(self.fig, use_container_width=True)

class Plots:
    def __init__(self, data):
        self.data = data
        self.use_candlestick = False

    def set_chart_type(self, use_candlestick=False):
        """Set whether to use candlestick charts or line charts"""
        self.use_candlestick = use_candlestick

    def plot_predictions(self, predictions, future_predictions, title='Stock Price Prediction'):
        predicted_dates = self.data.index[-len(predictions):]  
        future_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=len(future_predictions), freq='B')
        predictions = [float(val) for val in predictions if pd.notna(val)]
        future_predictions = [float(val) for val in future_predictions if pd.notna(val)]

        fig = make_subplots(rows=1, cols=1)
        
        # Add historical data (as candlestick or line chart)
        if self.use_candlestick:
            fig.add_trace(go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='Actual Stock Prices',
                increasing_line_color='#2B8308',  # Green for increasing candles
                decreasing_line_color='#9C1F0B'   # Red for decreasing candles
            ))
        else:
            fig.add_trace(go.Scatter(
                x=self.data.index, 
                y=self.data['Close'], 
                mode='lines', 
                name='Actual Stock Prices', 
                marker_color='blue'
            ))
        
        # Add predictions and forecasts (always as lines)
        fig.add_trace(go.Scatter(
            x=predicted_dates, 
            y=predictions, 
            mode='lines', 
            name='LSTM Predicted Prices', 
            marker_color='red', 
            line=dict(dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_predictions, 
            mode='lines', 
            name='Future Predictions', 
            marker_color='green', 
            line=dict(dash='dot')
        ))

        fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', legend_title='Legend', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    def plot_comparison(self, stocks_data, ticker_names, use_candlestick=False):
        """
        Plot comparison of multiple stocks
        
        Args:
            stocks_data: Dictionary with ticker as key and dataframe as value
            ticker_names: List of ticker names to include in comparison
            use_candlestick: Whether to use candlestick charts (only applies to single stock)
        """
        # If using candlestick and only one ticker, show detailed view
        if use_candlestick and len(ticker_names) == 1:
            ticker = ticker_names[0]
            if ticker in stocks_data:
                df = stocks_data[ticker]
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=ticker,
                    increasing_line_color='#2B8308',  # Green
                    decreasing_line_color='#9C1F0B'   # Red
                )])
                
                fig.update_layout(
                    title=f'{ticker} Candlestick Chart',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                return
        
        # For multiple stocks or when not using candlesticks, use line chart
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple']  # Colors for different stocks
        
        for i, ticker in enumerate(ticker_names):
            if ticker in stocks_data:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=stocks_data[ticker].index, 
                    y=stocks_data[ticker]['Close'],
                    mode='lines',
                    name=f'{ticker} Close Price',
                    marker_color=color
                ))
        
        fig.update_layout(
            title='Stock Price Comparison',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Stocks',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def plot_multiple_metrics(self, stocks_data, ticker_names, metric='Close'):
        """
        Plot specific metric for multiple stocks with normalized values for better comparison
        
        Args:
            stocks_data: Dictionary with ticker as key and dataframe as value
            ticker_names: List of ticker names to include in comparison
            metric: The column name to plot (default: 'Close')
        """
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Normalize data to percentage change from first day for fair comparison
        for i, ticker in enumerate(ticker_names):
            if ticker in stocks_data and metric in stocks_data[ticker].columns:
                data = stocks_data[ticker][metric]
                normalized_data = (data / data.iloc[0]) * 100  # Convert to percentage of initial value
                
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=stocks_data[ticker].index,
                    y=normalized_data,
                    mode='lines',
                    name=f'{ticker} ({metric})',
                    marker_color=color
                ))
        
        fig.update_layout(
            title=f'Normalized {metric} Price Comparison (Base: 100%)',
            xaxis_title='Date',
            yaxis_title='Normalized Price (%)',
            legend_title='Stocks',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)



