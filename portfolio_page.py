import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from portfolio import PortfolioManager

class PortfolioPage:
    def __init__(self):
        self.portfolio_manager = PortfolioManager()
        
    def render_header(self):
        """Render the portfolio dashboard header"""
        st.write("--------------------------------------------")
        st.write(f'<div style="font-size:50px">💼 Portfolio Management', unsafe_allow_html=True)
        st.divider()
        
        # Portfolio summary
        total_value, holdings_value = self.portfolio_manager.get_portfolio_value()
        cash = self.portfolio_manager.portfolio['cash']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Portfolio Value",
                value=f"${total_value:,.2f}",
            )
            
        with col2:
            st.metric(
                label="Holdings Value",
                value=f"${holdings_value:,.2f}",
                delta=f"{(holdings_value/total_value)*100:.1f}% of portfolio" if total_value > 0 else "0% of portfolio"
            )
            
        with col3:
            st.metric(
                label="Cash Balance",
                value=f"${cash:,.2f}",
                delta=f"{(cash/total_value)*100:.1f}% of portfolio" if total_value > 0 else "0% of portfolio"
            )
    
    def render_portfolio_management(self):
        """Render the section to manage portfolio holdings"""
        st.subheader("Manage Portfolio")
        
        tabs = st.tabs(["Add Stock", "Remove Stock", "Manage Cash"])
        
        # Tab 1: Add Stock
        with tabs[0]:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                ticker_input = st.text_input("Stock Symbol", key="add_ticker", help="Enter a valid stock ticker (e.g., AAPL)")
                
            with col2:
                shares = st.number_input("Number of Shares", min_value=0.01, step=0.01, value=1.0, key="add_shares")
                
            with col3:
                price = st.number_input("Purchase Price ($)", min_value=0.01, step=0.01, value=100.0, key="add_price")
            
            with col4:
                purchase_date = st.date_input("Purchase Date", value=datetime.now().date(), key="purchase_date")
                
            if st.button("Add to Portfolio", key="add_btn"):
                if ticker_input and shares > 0 and price > 0:
                    success, message = self.portfolio_manager.add_holding(
                        ticker_input,
                        shares,
                        price,
                        purchase_date.strftime("%Y-%m-%d")
                    )
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter a valid ticker, number of shares, and purchase price.")
                    
        # Tab 2: Remove Stock
        with tabs[1]:
            holdings_data = self.portfolio_manager.get_holdings_data()
            
            if holdings_data:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    tickers = [holding['ticker'] for holding in holdings_data]
                    selected_ticker = st.selectbox("Select Stock to Sell", options=tickers, key="sell_ticker")
                
                with col2:
                    # Find the selected stock details
                    selected_holding = next((h for h in holdings_data if h['ticker'] == selected_ticker), None)
                    max_shares = selected_holding['shares'] if selected_holding else 0
                    shares_to_sell = st.number_input(
                        "Shares to Sell", 
                        min_value=0.01, 
                        max_value=float(max_shares), 
                        step=0.01,
                        value=float(max_shares),
                        key="sell_shares"
                    )
                
                with col3:
                    # Default to current market price
                    current_price = selected_holding['current_price'] if selected_holding else 0
                    sale_price = st.number_input(
                        "Sale Price ($)", 
                        min_value=0.01, 
                        step=0.01,
                        value=float(current_price),
                        key="sell_price"
                    )
                
                if st.button("Sell Stock", key="sell_btn"):
                    if selected_ticker and shares_to_sell > 0:
                        success, message = self.portfolio_manager.remove_holding(
                            selected_ticker,
                            shares_to_sell,
                            sale_price
                        )
                        
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                    else:
                        st.warning("Please select a stock and enter a valid number of shares to sell.")
            else:
                st.info("No stocks in your portfolio. Add some stocks first.")
        
        # Tab 3: Manage Cash
        with tabs[2]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Deposit Cash")
                deposit_amount = st.number_input(
                    "Amount to Deposit ($)",
                    min_value=0.01,
                    step=10.0,
                    value=1000.0,
                    key="deposit_amount"
                )
                
                if st.button("Deposit", key="deposit_btn"):
                    success, message = self.portfolio_manager.update_cash(deposit_amount)
                    st.success(message)
            
            with col2:
                st.subheader("Withdraw Cash")
                current_cash = self.portfolio_manager.portfolio['cash']
                withdraw_amount = st.number_input(
                    "Amount to Withdraw ($)",
                    min_value=0.01,
                    max_value=float(current_cash) if current_cash > 0 else 0.01,
                    step=10.0,
                    value=min(1000.0, float(current_cash)) if current_cash > 0 else 0.01,
                    key="withdraw_amount"
                )
                
                if st.button("Withdraw", key="withdraw_btn"):
                    if withdraw_amount <= current_cash:
                        success, message = self.portfolio_manager.update_cash(-withdraw_amount)
                        st.success(message)
                    else:
                        st.error(f"Insufficient funds. Current cash balance: ${current_cash:.2f}")

    def render_holdings_table(self):
        """Render the current holdings table with performance data"""
        st.subheader("Current Holdings")
        
        holdings_data = self.portfolio_manager.get_holdings_data()
        
        if holdings_data:
            # Convert to DataFrame for display
            df = pd.DataFrame(holdings_data)
            
            # Format the DataFrame for display
            df['purchase_price'] = df['purchase_price'].map('${:,.2f}'.format)
            df['current_price'] = df['current_price'].map('${:,.2f}'.format)
            df['cost_basis'] = df['cost_basis'].map('${:,.2f}'.format)
            df['current_value'] = df['current_value'].map('${:,.2f}'.format)
            
            # Format gain/loss with color
            def format_gain_loss(row):
                gain_loss = row['gain_loss']
                color = 'green' if gain_loss >= 0 else 'red'
                return f"<span style='color:{color}'>${gain_loss:,.2f}</span>"
            
            def format_gain_loss_pct(row):
                gain_loss_pct = row['gain_loss_pct']
                color = 'green' if gain_loss_pct >= 0 else 'red'
                return f"<span style='color:{color}'>{gain_loss_pct:.2f}%</span>"
            
            # Apply the formatting
            df['gain_loss_formatted'] = df.apply(format_gain_loss, axis=1)
            df['gain_loss_pct_formatted'] = df.apply(format_gain_loss_pct, axis=1)
            
            # Select columns to display
            display_df = df[[
                'ticker', 'name', 'shares', 'purchase_price', 'current_price', 
                'cost_basis', 'current_value', 'gain_loss_formatted', 'gain_loss_pct_formatted', 'purchase_date'
            ]].rename(columns={
                'ticker': 'Symbol',
                'name': 'Company Name',
                'shares': 'Shares',
                'purchase_price': 'Purchase Price',
                'current_price': 'Current Price',
                'cost_basis': 'Cost Basis',
                'current_value': 'Market Value',
                'gain_loss_formatted': 'Gain/Loss ($)',
                'gain_loss_pct_formatted': 'Gain/Loss (%)',
                'purchase_date': 'Purchase Date'
            })
            
            # Display the table with HTML formatting
            st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            # Add export button
            if st.button("Export Holdings to CSV"):
                holdings_df, _ = self.portfolio_manager.export_portfolio_data()
                st.download_button(
                    label="Download CSV",
                    data=holdings_df.to_csv(index=False),
                    file_name="portfolio_holdings.csv",
                    mime="text/csv"
                )
        else:
            st.info("No stocks in your portfolio. Add some stocks to see performance.")

    def render_performance_analysis(self):
        """Render portfolio performance analysis"""
        st.subheader("Portfolio Performance Analysis")
        
        # Check if there are holdings
        if not self.portfolio_manager.portfolio['holdings']:
            st.info("Add stocks to your portfolio to view performance analysis.")
            return
        
        # Performance period selector
        col1, col2 = st.columns([1, 3])
        
        with col1:
            period = st.selectbox(
                "Performance Period",
                options=["1mo", "3mo", "6mo", "1y", "3y", "5y"],
                index=3,  # Default to 1y
                key="performance_period"
            )
        
        # Plot performance history
        performance_chart = self.portfolio_manager.plot_performance_history(period)
        if performance_chart:
            st.plotly_chart(performance_chart, use_container_width=True)
        else:
            st.warning("Insufficient data to display performance chart.")
        
        # Show allocation charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot portfolio allocation by stock
            allocation_chart = self.portfolio_manager.plot_portfolio_allocation()
            if allocation_chart:
                st.plotly_chart(allocation_chart, use_container_width=True)
        
        with col2:
            # Plot sector allocation
            sector_chart = self.portfolio_manager.plot_sector_allocation()
            if sector_chart:
                st.plotly_chart(sector_chart, use_container_width=True)

    def render_transaction_history(self):
        """Render transaction history"""
        st.subheader("Transaction History")
        
        transactions = self.portfolio_manager.portfolio['transactions']
        
        if transactions:
            # Create DataFrame for display
            df = pd.DataFrame(transactions)
            
            # Format the columns
            df['price'] = df['price'].map('${:,.2f}'.format)
            df['total'] = df['total'].map('${:,.2f}'.format)
            
            # Apply conditional formatting for action
            def format_action(action):
                if action == 'BUY':
                    return f"<span style='color:blue'>{action}</span>"
                elif action == 'SELL':
                    return f"<span style='color:green'>{action}</span>"
                elif action == 'DEPOSIT':
                    return f"<span style='color:purple'>{action}</span>"
                elif action == 'WITHDRAWAL':
                    return f"<span style='color:orange'>{action}</span>"
                return action
            
            df['action_formatted'] = df['action'].apply(format_action)
            
            # Select columns to display and rename
            display_df = df[['date', 'ticker', 'action_formatted', 'shares', 'price', 'total']].rename(columns={
                'date': 'Date',
                'ticker': 'Symbol',
                'action_formatted': 'Action',
                'shares': 'Shares',
                'price': 'Price',
                'total': 'Total'
            })
            
            # Sort by most recent first
            display_df = display_df.sort_values(by='Date', ascending=False)
            
            # Display the table with HTML formatting
            st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            # Add export button
            if st.button("Export Transactions to CSV"):
                _, transactions_df = self.portfolio_manager.export_portfolio_data()
                st.download_button(
                    label="Download CSV",
                    data=transactions_df.to_csv(index=False),
                    file_name="portfolio_transactions.csv",
                    mime="text/csv"
                )
        else:
            st.info("No transactions recorded yet.")
    
    def run(self):
        """Main method to run the portfolio page"""
        self.render_header()
        
        # Create tabs for different portfolio sections
        tabs = st.tabs([
            "Portfolio Overview", 
            "Manage Holdings", 
            "Performance Analysis",
            "Transaction History"
        ])
        
        # Tab 1: Portfolio Overview
        with tabs[0]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Current holdings table
                self.render_holdings_table()
            
            with col2:
                # Add performance summary
                if self.portfolio_manager.portfolio['holdings']:
                    st.subheader("Portfolio Allocation")
                    allocation_chart = self.portfolio_manager.plot_portfolio_allocation()
                    if allocation_chart:
                        st.plotly_chart(allocation_chart, use_container_width=True)
                else:
                    st.info("Add stocks to your portfolio to see allocation chart.")
        
        # Tab 2: Manage Holdings
        with tabs[1]:
            self.render_portfolio_management()
        
        # Tab 3: Performance Analysis
        with tabs[2]:
            self.render_performance_analysis()
            
        # Tab 4: Transaction History
        with tabs[3]:
            self.render_transaction_history()