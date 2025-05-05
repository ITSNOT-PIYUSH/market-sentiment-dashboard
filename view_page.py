from stock_data_loader import StockDataLoader
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plots import Plots, StockChart
from news_fetcher import NewsFetcher
from sentiment_analysis import StockSentimentAnalyzer

class StockDashboard:
    def __init__(self):
        self.popular_tickers = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX', 'TSLA', 'AMD', 'INTC']
        self.period_map = {'all': 'max','1m': '1mo', '6m': '6mo', '1y': '1y'}
        self.ticker = None
        self.available_indicators = {
            "Moving Averages": "moving_averages",
            "Bollinger Bands": "bollinger_bands",
            "RSI": "rsi",
            "MACD": "macd",
            "Stochastic": "stochastic"
        }
        self.selected_indicators = []
        # Initialize sentiment analysis components
        self.sentiment_analyzer = StockSentimentAnalyzer()
        self.news_fetcher = NewsFetcher()
        self.news_lookback_days = 7  # Default to 7 days of news
        self.show_sentiment = True  # Default to showing sentiment analysis
        self.use_candlestick = False  # Default to line charts

    def validate_ticker(self, ticker):
        """Validate if ticker exists in yfinance"""
        if not ticker:
            return False
        try:
            # Attempt to fetch a small amount of data to validate ticker
            data = yf.download(ticker, period="1d", progress=False)
            return not data.empty
        except Exception:
            return False

    def render_sidebar(self):
        st.sidebar.header("Choose your filter:")
        
        # Option to select from popular tickers or enter custom ticker
        ticker_input_method = st.sidebar.radio("Select ticker input method", 
                                              ["Choose from popular tickers", "Enter custom ticker"])
        
        if ticker_input_method == "Choose from popular tickers":
            self.ticker = st.sidebar.selectbox('Choose Ticker', options=self.popular_tickers, help='Select a ticker')
        else:
            custom_ticker = st.sidebar.text_input('Enter any ticker symbol (e.g., TSLA, AAPL)', 
                                               value='', 
                                               help="Enter any valid stock ticker symbol from yfinance")
            
            # Just use the custom ticker directly if provided, validation happens in load_data
            if custom_ticker:
                self.ticker = custom_ticker.strip().upper()
            else:
                # Default to first popular ticker if none provided
                self.ticker = self.popular_tickers[0]
        
        # Make sure selected_range is set before it might be used elsewhere
        self.selected_range = st.sidebar.selectbox('Select Period', options=list(self.period_map.keys()), index=3)  # Default to '1y'

        # Display Options Section
        st.sidebar.header("Display Options")
        # Add candlestick toggle option
        self.use_candlestick = st.sidebar.checkbox(
            "Use Candlestick Chart", 
            value=False, 
            help="Display price data as candlestick charts instead of line charts"
        )
        
        # Technical Indicators Section        
        st.sidebar.header("Technical Indicators")
        st.sidebar.write("Select indicators to display:")
        
        # Create checkboxes for each indicator
        self.selected_indicators = []
        for indicator_name, indicator_value in self.available_indicators.items():
            if st.sidebar.checkbox(indicator_name, value=(indicator_name=="Moving Averages")):
                self.selected_indicators.append(indicator_value)
        
        # Sentiment Analysis Section
        st.sidebar.header("Sentiment Analysis")
        self.show_sentiment = st.sidebar.checkbox("Show Sentiment Analysis", value=True)
        
        if self.show_sentiment:
            # Select news lookback period
            self.news_lookback_days = st.sidebar.slider(
                "News lookback period (days)", 
                min_value=1, 
                max_value=30, 
                value=7,
                help="Number of days to look back for news articles"
            )

    def load_data(self):
        if not self.ticker:
            st.error("No ticker selected. Please select a ticker to continue.")
            return False
            
        try:
            # Validate the ticker first
            with st.spinner(f"Validating ticker symbol {self.ticker}..."):
                is_valid = self.validate_ticker(self.ticker)
                if not is_valid:
                    st.error(f"Invalid ticker symbol: {self.ticker}. Please enter a valid ticker symbol.")
                    return False
            
            with st.spinner(f"Loading data for {self.ticker}..."):
                self.yf_data = yf.Ticker(self.ticker)
                self.df_history = self.yf_data.history(period=self.period_map[self.selected_range])
                
                # Safety check in case data is empty
                if self.df_history.empty:
                    st.error(f"No data available for ticker {self.ticker}.")
                    return False
                    
                self.current_price = self.yf_data.info.get('currentPrice', 'N/A')
                self.previous_close = self.yf_data.info.get('previousClose', 'N/A')
                st.success(f"Successfully loaded data for {self.ticker}")
                return True
        except Exception as e:
            st.error(f"Error loading data for {self.ticker}: {str(e)}")
            return False
          
    def display_header(self):
        try:
            company_name = self.yf_data.info.get('shortName', self.ticker)
            symbol = self.yf_data.info.get('symbol', self.ticker)
            st.subheader(f'{company_name} ({symbol}) 💰')
            st.divider()
            if self.current_price != 'N/A' and self.previous_close != 'N/A':
                price_change = self.current_price - self.previous_close
                price_change_ratio = (abs(price_change) / self.previous_close * 100)
                price_change_direction = "+" if price_change > 0 else "-"
                st.metric(label='Current Price', value=f"{self.current_price:.2f}",
                          delta=f"{price_change:.2f} ({price_change_direction}{price_change_ratio:.2f}%)")
        except Exception as e:
            st.warning(f"Limited data available for {self.ticker}. Some information may not be displayed.")
            st.subheader(f'{self.ticker} 💰')

    def load_news(self):
        """Load news articles for the selected ticker"""
        if not self.ticker:
            return []
        
        try:
            with st.spinner(f"Fetching news for {self.ticker}..."):
                # Use the news fetcher to get news
                news = self.news_fetcher.get_news(
                    ticker=self.ticker, 
                    days_lookback=self.news_lookback_days,
                    use_dummy_if_empty=False
                )
                
                # Filter news that has content to analyze
                filtered_news = []
                for article in news if news else []:
                    if 'title' in article or 'summary' in article:
                        filtered_news.append(article)
                
                return filtered_news
                
        except Exception as e:
            st.error(f"Error loading news for {self.ticker}: {str(e)}")
            return []

    def analyze_news_sentiment(self, news):
        """Analyze sentiment of news articles"""
        if not news:
            return None
        
        # Extract text content for sentiment analysis
        texts = []
        for article in news:
            if 'title' in article:
                texts.append(article['title'])
            if 'summary' in article:
                texts.append(article['summary'])
        
        # Analyze sentiment
        return self.sentiment_analyzer.analyze_sentiment(texts)

    def display_sentiment_header(self, sentiment_results):
        """Display sentiment analysis results header"""
        if not sentiment_results:
            return
            
        sentiment = sentiment_results['sentiment']
        scores = sentiment_results['scores']
        
        st.subheader("News Sentiment Analysis")
        
        # Create columns for metrics
        col1, col2 = st.columns(2)
        
        # Display overall sentiment with appropriate color
        color = self.sentiment_analyzer.get_sentiment_color(sentiment)
        with col1:
            st.markdown(f"""
            <div style="background-color: {color}; padding: 10px; border-radius: 5px;">
                <h3 style="color: white; text-align: center;">Overall Sentiment: {sentiment.upper()}</h3>
            </div>
            """, unsafe_allow_html=True)
            
        # Display sentiment scores
        with col2:
            chart_data = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Score': [scores['positive'], scores['neutral'], scores['negative']]
            })
            
            # Custom colors
            colors = ['green', 'grey', 'red']
            
            fig = go.Figure(data=[go.Bar(
                x=chart_data['Sentiment'],
                y=chart_data['Score'],
                marker_color=colors
            )])
            
            fig.update_layout(
                title='Sentiment Scores',
                xaxis_title='Sentiment',
                yaxis_title='Score',
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def display_news_with_sentiment(self, news):
        """Display news articles with sentiment indicators"""
        if not news:
            st.info(f"No recent news found for {self.ticker} in the past {self.news_lookback_days} days.")
            return
            
        st.subheader(f"Recent News for {self.ticker}")
        
        # Sort news by date if available
        news.sort(key=lambda x: x.get('providerPublishTime', 0), reverse=True)
        
        for article in news:
            # Extract article details
            title = article.get('title', 'No Title')
            publisher = article.get('publisher', 'Unknown Publisher')
            summary = article.get('summary', 'No summary available.')
            link = article.get('link', '#')
            
            # Analyze individual article sentiment
            sentiment_results = self.sentiment_analyzer.analyze_sentiment([title, summary])
            sentiment = sentiment_results['sentiment']
            color = self.sentiment_analyzer.get_sentiment_color(sentiment)
            
            # Format publish time
            publish_time = article.get('providerPublishTime', None)
            time_str = "Unknown date"
            if publish_time:
                dt_object = datetime.fromtimestamp(publish_time)
                time_str = dt_object.strftime("%Y-%m-%d %H:%M")
            
            # Display article with sentiment color indicator
            st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding-left: 10px; margin-bottom: 20px;">
                <h4><a href="{link}" target="_blank">{title}</a></h4>
                <p><strong>Publisher:</strong> {publisher} | <strong>Published:</strong> {time_str} | <strong>Sentiment:</strong> <span style="color:{color};">{sentiment.upper()}</span></p>
                <p>{summary}</p>
            </div>
            """, unsafe_allow_html=True)

    def plot_data(self):
        if hasattr(self, 'df_history') and not self.df_history.empty:
            chart = StockChart(self.df_history)
            # Set chart type based on user preference
            chart.set_chart_type(self.use_candlestick)
            # Add price chart (either candlestick or line based on the setting)
            chart.add_price_chart()
            
            # Add selected technical indicators
            for indicator in self.selected_indicators:
                if indicator == "moving_averages":
                    chart.add_moving_averages()
                elif indicator == "bollinger_bands":
                    chart.add_bollinger_bands()
                elif indicator == "rsi":
                    chart.add_rsi()
                elif indicator == "macd":
                    chart.add_macd()
                elif indicator == "stochastic":
                    chart.add_stochastic()
            
            chart.add_volume_chart()
            chart.render_chart(title=f"{self.ticker} - Price with Technical Indicators")
            
            # Display indicator descriptions if any are selected
            if self.selected_indicators:
                self.display_indicator_descriptions()
        else:
            st.error("No data available to plot.")
    
    def display_indicator_descriptions(self):
        st.subheader("About the Selected Indicators")
        
        indicator_descriptions = {
            "moving_averages": {
                "title": "Moving Averages",
                "description": "A moving average smooths price data to form a trend following indicator. "
                              "It helps identify trend direction and potential support/resistance levels. "
                              "Common periods shown are 20 days (short-term), 50 days (mid-term), and 200 days (long-term)."
            },
            "bollinger_bands": {
                "title": "Bollinger Bands",
                "description": "Bollinger Bands consist of a middle band (20-day SMA) and two outer bands "
                              "placed 2 standard deviations above and below the middle band. They help identify "
                              "volatility and potential overbought/oversold conditions."
            },
            "rsi": {
                "title": "Relative Strength Index (RSI)",
                "description": "RSI measures the speed and change of price movements, oscillating between 0 and 100. "
                              "Traditionally, RSI values over 70 indicate overbought conditions, while values below 30 "
                              "suggest oversold conditions."
            },
            "macd": {
                "title": "Moving Average Convergence Divergence (MACD)",
                "description": "MACD shows the relationship between two moving averages of a security's price. "
                              "It consists of the MACD line (12-day EMA minus 26-day EMA), the signal line (9-day EMA of MACD), "
                              "and a histogram showing their difference. It helps identify momentum, trend direction, and potential reversals."
            },
            "stochastic": {
                "title": "Stochastic Oscillator",
                "description": "The Stochastic Oscillator compares a security's closing price to its price range over a period of time. "
                              "%K represents the current price position relative to the high-low range, while %D is a 3-day moving average of %K. "
                              "Values above 80 suggest overbought conditions, while values below 20 suggest oversold conditions."
            }
        }
        
        for indicator in self.selected_indicators:
            info = indicator_descriptions.get(indicator, None)
            if info:
                st.write(f"**{info['title']}**")
                st.write(info['description'])
                st.divider()

    def display_sentiment_analysis(self):
        """Perform and display sentiment analysis"""
        # Add loading placeholder for model loading
        loading_placeholder = st.empty()
        loading_placeholder.info("Loading transformer model for sentiment analysis... This may take a few seconds.")
        
        news = self.load_news()
        sentiment_results = self.analyze_news_sentiment(news)
        
        # Clear loading message
        loading_placeholder.empty()
        
        if sentiment_results:
            self.display_sentiment_header(sentiment_results)
            st.divider()
            self.display_news_with_sentiment(news)
        else:
            st.warning(f"Unable to perform sentiment analysis for {self.ticker}. No news articles found.")

    def run(self):
        st.write("--------------------------------------------")
        st.write(f'<div style="font-size:50px">📈 Stock Analysis Dashboard', unsafe_allow_html=True)
        self.render_sidebar()
        if self.load_data():
            self.display_header()
            
            # Create tabs for Technical Analysis and Sentiment Analysis
            tab1, tab2 = st.tabs(["Technical Analysis", "News Sentiment"])
            
            # Technical Analysis Tab
            with tab1:
                self.plot_data()
            
            # Sentiment Analysis Tab - Only load when this tab is selected
            with tab2:
                if self.show_sentiment:
                    self.display_sentiment_analysis()
                else:
                    st.info("Sentiment analysis is disabled. Enable it in the sidebar to view news sentiment.")


