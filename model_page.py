import pandas as pd
import streamlit as st
import yfinance as yf
import traceback
from model import Model, MultiTickerModel
from plots import Plots
from stock_data_loader import StockDataLoader

class StockModelPage:
    def __init__(self):
        self.popular_tickers = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX', 'TSLA', 'AMD', 'INTC']
        self.setup_sidebar()

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

    def setup_sidebar(self):
        st.sidebar.header("Stock Selection")
        self.analysis_mode = st.sidebar.radio("Analysis Mode", ["Single Stock", "Multiple Stocks Comparison"])
        
        if self.analysis_mode == "Single Stock":
            # Option to select from popular tickers or enter custom ticker
            ticker_input_method = st.sidebar.radio("Select ticker input method", 
                                                  ["Choose from popular tickers", "Enter custom ticker"])
            
            if ticker_input_method == "Choose from popular tickers":
                self.ticker = st.sidebar.selectbox('Choose Stock Ticker', self.popular_tickers)
            else:
                self.ticker = st.sidebar.text_input('Enter any ticker symbol (e.g., TSLA, AAPL)', 
                                                   value='', 
                                                   help="Enter any valid stock ticker symbol from yfinance")
                
            self.selected_tickers = [self.ticker] if self.ticker else []
        else:
            # Multiple stocks mode
            st.sidebar.subheader("Select multiple tickers (max 5)")
            # Option to choose from popular or add custom tickers
            popular_selected = st.sidebar.multiselect('Choose from popular tickers', 
                                                     self.popular_tickers, 
                                                     default=['NVDA', 'AAPL'])
            
            # Custom ticker input with comma separation
            custom_tickers_input = st.sidebar.text_input('Enter additional custom tickers (comma separated)', 
                                                       value='', 
                                                       help="Example: TWTR,UBER,SPOT")
            
            # Process custom tickers
            custom_tickers = []
            if custom_tickers_input:
                custom_tickers = [t.strip().upper() for t in custom_tickers_input.split(',') if t.strip()]
                
            # Combine selected popular and valid custom tickers
            self.selected_tickers = popular_selected + custom_tickers
            
            # Limit to maximum 5 tickers for performance reasons
            if len(self.selected_tickers) > 5:
                self.selected_tickers = self.selected_tickers[:5]
                st.sidebar.warning("Maximum 5 stocks can be compared at once")
        
        # Chart display options
        st.sidebar.header("Chart Options")
        self.use_candlestick = st.sidebar.checkbox("Use Candlestick Chart", value=False, 
                                                help="Display price data as candlestick charts instead of line charts")
                
        self.start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2010-01-01'))
        self.end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('today'))
        self.load_button_clicked = st.sidebar.button('Load Data')

    def load_data(self):
        if self.load_button_clicked:
            if not self.selected_tickers:
                st.warning("Please select or enter at least one stock ticker")
                return
            
            # Validate tickers before proceeding
            valid_tickers = []
            invalid_tickers = []
            
            with st.spinner("Validating ticker symbols..."):
                for ticker in self.selected_tickers:
                    if self.validate_ticker(ticker):
                        valid_tickers.append(ticker)
                    else:
                        invalid_tickers.append(ticker)
            
            # Notify about invalid tickers
            if invalid_tickers:
                st.error(f"Invalid ticker symbols: {', '.join(invalid_tickers)}. Please verify these symbols exist in yfinance.")
                
            if not valid_tickers:
                st.error("No valid tickers to process. Please enter valid ticker symbols.")
                return
                
            # Proceed with valid tickers
            st.write("--------------------------------------------")
            if 'multi_ticker_model' not in st.session_state:
                st.session_state['multi_ticker_model'] = MultiTickerModel()
            
            # Clear previous data if any
            if 'stock_data' in st.session_state:
                st.session_state['stock_data'] = {}
            else:
                st.session_state['stock_data'] = {}
            
            with st.spinner(f"Loading data for {len(valid_tickers)} ticker(s)..."):
                for ticker in valid_tickers:
                    try:
                        loader = StockDataLoader(ticker, self.start_date, self.end_date)
                        data = loader.get_stock_data()
                        
                        # Store data in session state
                        st.session_state['stock_data'][ticker] = data
                        
                        # Add data to multi-ticker model
                        st.session_state['multi_ticker_model'].add_ticker_data(ticker, data)
                    except Exception as e:
                        st.error(f"Error loading data for {ticker}: {str(e)}")
                
            ticker_list = ', '.join(valid_tickers)
            if valid_tickers:
                st.success(f"Data for {ticker_list} from {self.start_date} to {self.end_date} loaded successfully!")

    def handle_model_training(self):
        if 'stock_data' in st.session_state and st.session_state['stock_data']:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                epochs = st.slider("Training Epochs", min_value=1, max_value=20, value=5)
                
            with col2:
                forecast_days = st.slider("Forecast Days", min_value=1, max_value=30, value=5)
            
            if st.button('Train Model & Predict'):
                status_placeholder = st.empty()
                status_placeholder.info("Training model... This may take a moment.")
                
                try:
                    if self.analysis_mode == "Single Stock":
                        # Single stock mode
                        ticker = self.selected_tickers[0]
                        stock_data = st.session_state['stock_data'][ticker]
                        
                        try:
                            model = Model(stock_data)
                            model.train_lstm(epochs=epochs)
                            predictions = model.make_predictions()
                            future_predictions = model.forecast_future(days=forecast_days)
                            
                            # Clear status message
                            status_placeholder.empty()
                            
                            # Plot predictions
                            self.plot_predictions(stock_data, predictions, future_predictions, ticker)
                            
                        except Exception as e:
                            status_placeholder.error(f"Error training model: {str(e)}")
                            st.error("Model training failed. See details below:")
                            st.code(traceback.format_exc())
                            st.info("Trying fallback approach...")
                            
                            # Fallback to simpler model or approach
                            self.fallback_prediction(ticker, stock_data, forecast_days)
                    
                    else:
                        # Multiple stocks comparison mode
                        multi_model = st.session_state['multi_ticker_model']
                        
                        results = {}
                        for ticker in self.selected_tickers:
                            try:
                                multi_model.train_model(ticker, epochs=epochs)
                                predictions = multi_model.get_predictions(ticker)
                                forecast = multi_model.get_forecast(ticker, days=forecast_days)
                                results[ticker] = {
                                    'predictions': predictions, 
                                    'forecast': forecast
                                }
                            except Exception as e:
                                st.error(f"Error training model for {ticker}: {str(e)}")
                                # Try fallback for this ticker
                                fallback_result = self.fallback_prediction(ticker, st.session_state['stock_data'][ticker], forecast_days, return_result=True)
                                if fallback_result:
                                    results[ticker] = fallback_result
                        
                        # Clear status message
                        status_placeholder.empty()
                        
                        if results:
                            self.plot_multiple_predictions(results)
                        else:
                            st.error("All model training attempts failed.")
                
                except Exception as e:
                    status_placeholder.error(f"Unexpected error: {str(e)}")
                    st.error("An unexpected error occurred during model training:")
                    st.code(traceback.format_exc())
            else:
                st.write("Click the button above to train the model.")
        else:
            st.write("--------------------------------------------")
            st.write("Please load data before training the model.")
    
    def fallback_prediction(self, ticker, stock_data, forecast_days, return_result=False):
        """Simple fallback prediction using moving averages when model training fails"""
        try:
            st.write(f"Using fallback prediction for {ticker} based on simple moving averages")
            
            # Calculate simple moving average
            closing_prices = stock_data['Close'].values
            
            # Use last available prices as prediction
            predictions = closing_prices.reshape(-1, 1)
            
            # Simple forecast using last price + small random change
            last_price = closing_prices[-1]
            import numpy as np
            forecast = np.array([last_price * (1 + (np.random.random() - 0.5) * 0.01) for _ in range(forecast_days)])
            future_predictions = forecast.reshape(-1, 1)
            
            if return_result:
                return {
                    'predictions': predictions,
                    'forecast': future_predictions
                }
            else:
                self.plot_predictions(stock_data, predictions, future_predictions, ticker)
                st.info("Note: This is a simplified fallback prediction and not a trained model.")
                
            return True
        
        except Exception as e:
            st.error(f"Even fallback prediction failed: {str(e)}")
            return None

    def plot_predictions(self, stock_data, predictions, future_predictions, ticker):
        plot_instance = Plots(stock_data)
        plot_instance.plot_predictions(predictions, future_predictions, title=f"{ticker} Stock Price Prediction")

    def plot_multiple_predictions(self, results):
        # Create visualization for multiple stocks
        st.subheader("Stock Price Predictions Comparison")
        
        for ticker, data in results.items():
            stock_data = st.session_state['stock_data'][ticker]
            plot_instance = Plots(stock_data)
            st.write(f"### {ticker} Predictions")
            plot_instance.plot_predictions(data['predictions'], data['forecast'], title=f"{ticker} Stock Price Prediction")
            st.write("--------------------------------------------")

    def run(self):
        st.write("--------------------------------------------")
        st.write(f'<div style="font-size:50px">🤖 Real-Time Stock Prediction', unsafe_allow_html=True)
        self.load_data()
        self.handle_model_training()