import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class PortfolioManager:
    def __init__(self):
        # Initialize portfolio from session state or create empty if doesn't exist
        if 'portfolio' not in st.session_state:
            st.session_state['portfolio'] = {
                'holdings': {},  # Dict with ticker as key and details as value
                'cash': 0,
                'transactions': []
            }
        self.portfolio = st.session_state['portfolio']
    
    def add_holding(self, ticker, shares, purchase_price, purchase_date):
        """Add a stock holding to the portfolio"""
        ticker = ticker.upper()
        
        # Validate ticker
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            name = info.get('shortName', ticker)
        except:
            return False, f"Could not validate ticker: {ticker}"
        
        # Add or update the holding
        if ticker in self.portfolio['holdings']:
            # Update existing holding with average cost calculation
            current = self.portfolio['holdings'][ticker]
            total_shares = current['shares'] + shares
            total_cost = (current['shares'] * current['purchase_price']) + (shares * purchase_price)
            avg_price = total_cost / total_shares if total_shares > 0 else 0
            
            self.portfolio['holdings'][ticker] = {
                'name': name,
                'shares': total_shares,
                'purchase_price': avg_price,
                'purchase_date': purchase_date
            }
        else:
            # Add new holding
            self.portfolio['holdings'][ticker] = {
                'name': name,
                'shares': shares,
                'purchase_price': purchase_price,
                'purchase_date': purchase_date
            }
        
        # Record the transaction
        self.portfolio['transactions'].append({
            'date': datetime.now().strftime("%Y-%m-%d"),
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': purchase_price,
            'total': shares * purchase_price
        })
        
        return True, f"Added {shares} shares of {ticker} to portfolio"
    
    def remove_holding(self, ticker, shares, sale_price=None):
        """Remove shares from a holding"""
        ticker = ticker.upper()
        
        if ticker not in self.portfolio['holdings']:
            return False, f"Ticker {ticker} not found in portfolio"
        
        current = self.portfolio['holdings'][ticker]
        
        if shares > current['shares']:
            return False, f"Cannot sell more shares than owned ({current['shares']} available)"
        
        # If no sale price provided, use current market price
        if sale_price is None:
            try:
                stock = yf.Ticker(ticker)
                quote = stock.history(period="1d")
                sale_price = quote['Close'].iloc[-1]
            except:
                return False, f"Could not get current price for {ticker}"
        
        # Update the holding
        current['shares'] -= shares
        
        # If shares is zero, remove the holding
        if current['shares'] <= 0:
            del self.portfolio['holdings'][ticker]
        
        # Record the transaction
        self.portfolio['transactions'].append({
            'date': datetime.now().strftime("%Y-%m-%d"),
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': sale_price,
            'total': shares * sale_price
        })
        
        # Add to cash balance
        self.portfolio['cash'] += shares * sale_price
        
        return True, f"Sold {shares} shares of {ticker}"
        
    def update_cash(self, amount):
        """Add or subtract cash from portfolio"""
        self.portfolio['cash'] += amount
        action = 'DEPOSIT' if amount > 0 else 'WITHDRAWAL'
        
        # Record the transaction
        self.portfolio['transactions'].append({
            'date': datetime.now().strftime("%Y-%m-%d"),
            'ticker': 'CASH',
            'action': action,
            'shares': 0,
            'price': 0,
            'total': amount
        })
        
        return True, f"{action} of ${abs(amount):.2f} processed"
    
    def get_portfolio_value(self):
        """Calculate current value of the portfolio"""
        total_value = self.portfolio['cash']
        holdings_value = 0
        
        # Get current prices for all holdings
        for ticker, details in self.portfolio['holdings'].items():
            try:
                stock = yf.Ticker(ticker)
                quote = stock.history(period="1d")
                current_price = quote['Close'].iloc[-1]
                value = details['shares'] * current_price
                holdings_value += value
            except:
                # If error, use the purchase price as fallback
                value = details['shares'] * details['purchase_price']
                holdings_value += value
        
        total_value += holdings_value
        return total_value, holdings_value
    
    def get_holdings_data(self):
        """Get detailed data about current holdings with performance metrics"""
        holdings_data = []
        
        for ticker, details in self.portfolio['holdings'].items():
            try:
                stock = yf.Ticker(ticker)
                quote = stock.history(period="1d")
                current_price = quote['Close'].iloc[-1]
                
                cost_basis = details['shares'] * details['purchase_price']
                current_value = details['shares'] * current_price
                gain_loss = current_value - cost_basis
                gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
                
                holdings_data.append({
                    'ticker': ticker,
                    'name': details['name'],
                    'shares': details['shares'],
                    'purchase_price': details['purchase_price'],
                    'current_price': current_price,
                    'cost_basis': cost_basis,
                    'current_value': current_value,
                    'gain_loss': gain_loss,
                    'gain_loss_pct': gain_loss_pct,
                    'purchase_date': details['purchase_date']
                })
            except Exception as e:
                # Add with error flag
                holdings_data.append({
                    'ticker': ticker,
                    'name': details['name'],
                    'shares': details['shares'],
                    'purchase_price': details['purchase_price'],
                    'current_price': details['purchase_price'],  # Use purchase price as fallback
                    'cost_basis': details['shares'] * details['purchase_price'],
                    'current_value': details['shares'] * details['purchase_price'],
                    'gain_loss': 0,
                    'gain_loss_pct': 0,
                    'purchase_date': details['purchase_date'],
                    'error': str(e)
                })
        
        return holdings_data
    
    def get_performance_history(self, period="1y"):
        """Get historical performance of portfolio"""
        if not self.portfolio['holdings']:
            return pd.DataFrame()
            
        tickers = list(self.portfolio['holdings'].keys())
        end_date = datetime.now()
        
        if period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "3y":
            start_date = end_date - timedelta(days=3*365)
        elif period == "5y":
            start_date = end_date - timedelta(days=5*365)
        else:
            start_date = end_date - timedelta(days=365)  # Default to 1y
            
        # Get historical data for all stocks
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        
        # Handle single ticker case properly
        if len(tickers) == 1:
            # If only one ticker, yfinance returns a Series instead of DataFrame
            # Convert it to DataFrame with the ticker as column name
            ticker = tickers[0]
            if isinstance(data, pd.Series):
                data = pd.DataFrame({ticker: data})
            
        # Calculate weighted portfolio value over time
        portfolio_values = pd.Series(index=data.index)
        
        for ticker, details in self.portfolio['holdings'].items():
            if ticker in data.columns:
                # Multiply stock price by number of shares
                portfolio_values = portfolio_values.add(
                    data[ticker] * details['shares'], 
                    fill_value=0
                )
            
        # Add cash component
        portfolio_values = portfolio_values + self.portfolio['cash']
        
        # Convert to DataFrame
        performance_df = pd.DataFrame({
            'date': portfolio_values.index,
            'value': portfolio_values.values
        })
        
        # Calculate daily returns
        if not performance_df.empty:
            performance_df['daily_return'] = performance_df['value'].pct_change()
            performance_df['cumulative_return'] = (1 + performance_df['daily_return']).cumprod() - 1
            
        return performance_df
    
    def get_sector_allocation(self):
        """Get sector allocation of the portfolio"""
        sectors = {}
        total_value = 0
        
        for ticker, details in self.portfolio['holdings'].items():
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                sector = info.get('sector', 'Unknown')
                
                # Get current price
                quote = stock.history(period="1d")
                current_price = quote['Close'].iloc[-1]
                value = details['shares'] * current_price
                
                if sector in sectors:
                    sectors[sector] += value
                else:
                    sectors[sector] = value
                    
                total_value += value
            except:
                # Use purchase price as fallback
                value = details['shares'] * details['purchase_price']
                if 'Unknown' in sectors:
                    sectors['Unknown'] += value
                else:
                    sectors['Unknown'] = value
                total_value += value
        
        # Calculate percentages
        sector_allocation = []
        for sector, value in sectors.items():
            percentage = (value / total_value) * 100 if total_value > 0 else 0
            sector_allocation.append({
                'sector': sector,
                'value': value,
                'percentage': percentage
            })
        
        return sector_allocation
    
    def plot_portfolio_allocation(self):
        """Create a pie chart of portfolio allocation by ticker"""
        holdings_data = self.get_holdings_data()
        
        if not holdings_data:
            return None
        
        # Create a DataFrame for plotting
        df = pd.DataFrame(holdings_data)
        
        # Create pie chart
        fig = px.pie(
            df, 
            values='current_value', 
            names='ticker', 
            title='Portfolio Allocation by Stock',
            hover_data=['name', 'shares', 'current_price', 'gain_loss_pct'],
            labels={
                'current_value': 'Value ($)',
                'ticker': 'Stock',
                'name': 'Company',
                'shares': 'Shares',
                'current_price': 'Price ($)',
                'gain_loss_pct': 'Gain/Loss (%)'
            }
        )
        
        # Update trace colors based on performance
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#FFF', width=1))
        )
        
        fig.update_layout(
            height=500,
            legend_title_text='Stocks'
        )
        
        return fig
    
    def plot_sector_allocation(self):
        """Create a pie chart of portfolio allocation by sector"""
        sector_data = self.get_sector_allocation()
        
        if not sector_data:
            return None
        
        # Create a DataFrame for plotting
        df = pd.DataFrame(sector_data)
        
        # Create pie chart
        fig = px.pie(
            df, 
            values='value', 
            names='sector', 
            title='Portfolio Allocation by Sector',
            labels={
                'value': 'Value ($)',
                'sector': 'Sector',
                'percentage': 'Percentage (%)'
            }
        )
        
        # Update trace
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#FFF', width=1))
        )
        
        fig.update_layout(
            height=500,
            legend_title_text='Sectors'
        )
        
        return fig
    
    def plot_performance_history(self, period="1y"):
        """Plot historical performance of the portfolio"""
        performance_df = self.get_performance_history(period)
        
        if performance_df.empty:
            return None
        
        # Create the line chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=performance_df['date'], 
                y=performance_df['value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(width=2, color='blue')
            )
        )
        
        # Format the chart
        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Value ($)',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def export_portfolio_data(self):
        """Export portfolio data to CSV format"""
        holdings_data = self.get_holdings_data()
        transactions = self.portfolio['transactions']
        
        holdings_df = pd.DataFrame(holdings_data)
        transactions_df = pd.DataFrame(transactions)
        
        return holdings_df, transactions_df