import pandas as pd
import yfinance as yf

import warnings
warnings.filterwarnings("ignore")

class StockDataLoader:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def get_stock_data(self):
        stock = yf.Ticker(self.ticker)
        stock_data = stock.history(start=self.start_date, end=self.end_date)
        stock_data.reset_index(inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
        return stock_data