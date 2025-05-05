import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sentiment_analysis import StockSentimentAnalyzer
from news_fetcher import NewsFetcher

class StockSentimentPage:
    def __init__(self):
        self.popular_tickers = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX', 'TSLA', 'AMD', 'INTC']
        self.ticker = None
        self.sentiment_analyzer = StockSentimentAnalyzer()
        self.news_fetcher = NewsFetcher()
        self.news_lookback_days = 7  # Default to 7 days of news

    def render_sidebar(self):
        st.sidebar.header("Choose your stock:")
        
        # Option to select from popular tickers or enter custom ticker
        ticker_input_method = st.sidebar.radio("Select ticker input method", 
                                              ["Choose from popular tickers", "Enter custom ticker"],
                                              key="sentiment_ticker_method")
        
        if ticker_input_method == "Choose from popular tickers":
            self.ticker = st.sidebar.selectbox('Choose Ticker', 
                                             options=self.popular_tickers, 
                                             help='Select a ticker',
                                             key="sentiment_popular_ticker")
        else:
            custom_ticker = st.sidebar.text_input('Enter any ticker symbol (e.g., TSLA, AAPL)', 
                                               value='', 
                                               help="Enter any valid stock ticker symbol from yfinance",
                                               key="sentiment_custom_ticker")
            
            if custom_ticker:
                self.ticker = custom_ticker.strip().upper()
            else:
                self.ticker = self.popular_tickers[0]
        
        # Select news lookback period
        self.news_lookback_days = st.sidebar.slider(
            "News lookback period (days)", 
            min_value=1, 
            max_value=30, 
            value=7,
            help="Number of days to look back for news articles"
        )

    def load_news(self):
        """Load news articles for the selected ticker"""
        if not self.ticker:
            st.error("No ticker selected. Please select a ticker to continue.")
            return []
        
        try:
            st.sidebar.info(f"Fetching news for {self.ticker}...")
            
            # Use the news fetcher to get news
            news = self.news_fetcher.get_news(
                ticker=self.ticker, 
                days_lookback=self.news_lookback_days,
                use_dummy_if_empty=False  # Never use dummy news
            )
            
            if news:
                st.sidebar.success(f"Found {len(news)} news articles for {self.ticker}.")
            else:
                st.sidebar.error(f"No news found for {self.ticker}.")
            
            # Filter news that has content to analyze
            filtered_news = []
            for article in news if news else []:
                if 'title' in article or 'summary' in article:
                    filtered_news.append(article)
            
            return filtered_news
            
        except Exception as e:
            st.error(f"Error loading news for {self.ticker}: {str(e)}")
            import traceback
            st.sidebar.error(f"Detailed error: {traceback.format_exc()}")
            return []

    def analyze_news_sentiment(self, news):
        """Analyze sentiment of news articles"""
        if not news:
            st.info(f"No recent news found for {self.ticker} in the past {self.news_lookback_days} days.")
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

    def run(self):
        st.write("--------------------------------------------")
        st.write(f'<div style="font-size:50px">🔍 Stock Sentiment Analysis', unsafe_allow_html=True)
        st.write(f'<div style="font-size:20px">Analyze news sentiment using transformer-based NLP models', unsafe_allow_html=True)
        
        self.render_sidebar()
        
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