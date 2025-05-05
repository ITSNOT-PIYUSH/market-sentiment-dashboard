import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta

class NewsFetcher:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Finnhub API key
        self.finnhub_api_key = "d0a9tahr01qnh1rh6ingd0a9tahr01qnh1rh6io0"
    
    def fetch_yahoo_finance_news(self, ticker, days_lookback=7):
        """Fetch news from Yahoo Finance using yfinance library"""
        try:
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news
            return news if news else []
        except Exception:
            return []
    
    def fetch_finnhub_news(self, ticker, days_lookback=7):
        """
        Fetch news using Finnhub API
        """
        try:
            # Calculate dates for the API query
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_lookback)
            
            # Format dates as required by Finnhub API (YYYY-MM-DD)
            from_date = start_date.strftime("%Y-%m-%d")
            to_date = end_date.strftime("%Y-%m-%d")
            
            # Finnhub API endpoint for company news
            url = f"https://finnhub.io/api/v1/company-news"
            
            # Parameters for the API call
            params = {
                'symbol': ticker,
                'from': from_date,
                'to': to_date,
                'token': self.finnhub_api_key
            }
            
            # Make the API request
            response = requests.get(url, params=params)
            
            # Check if request was successful
            if response.status_code == 200:
                finn_news = response.json()
                
                # Convert Finnhub news format to match the format we're using
                processed_news = []
                for article in finn_news:
                    processed_news.append({
                        'title': article.get('headline', ''),
                        'publisher': article.get('source', 'Finnhub'),
                        'summary': article.get('summary', ''),
                        'link': article.get('url', '#'),
                        'providerPublishTime': article.get('datetime', 0)
                    })
                
                return processed_news
            else:
                print(f"Finnhub API request failed with status code: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching news from Finnhub: {str(e)}")
            return []
    
    def fetch_marketaux_news(self, ticker, days_lookback=7):
        """
        Fetch news using MarketAux API (requires an API key)
        Note: This is a placeholder. To use it, you would need to sign up at marketaux.com
        """
        # This is a placeholder - for actual use, you'd need to add your API key
        # api_key = "YOUR_MARKETAUX_API_KEY"
        return []
    
    def get_news(self, ticker, days_lookback=7, use_dummy_if_empty=False):
        """
        Fetch news from multiple sources and combine results
        
        Args:
            ticker: Stock symbol
            days_lookback: Number of days to look back for news
            use_dummy_if_empty: Ignored parameter (kept for backward compatibility)
        
        Returns:
            List of news articles
        """
        # Try Finnhub first since we have an API key
        news = self.fetch_finnhub_news(ticker, days_lookback)
        
        # If no news found from Finnhub, try Yahoo Finance
        if not news:
            news = self.fetch_yahoo_finance_news(ticker, days_lookback)
        
        # If still no news, try MarketAux (placeholder, would require API key)
        if not news:
            news = self.fetch_marketaux_news(ticker, days_lookback)
            
        return news