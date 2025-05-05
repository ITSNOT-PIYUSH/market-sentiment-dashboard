---
title: Stock Predict Lstm
emoji: 👁
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Stock Analyzer

A comprehensive stock analysis and portfolio management tool built with Python and Streamlit.

## Features

### Stock Analysis
- Real-time stock price data visualization
- Technical indicators (Moving Averages, Bollinger Bands, RSI, MACD, Stochastic)
- Option to switch between line charts and candlestick charts
- Historical price data with custom date ranges

### News Sentiment Analysis
- Fetch recent news for selected stocks
- AI-powered sentiment analysis using FinBERT
- Visual sentiment score breakdown
- Color-coded news articles based on sentiment

### Portfolio Management
- Track your stock holdings and performance
- Add/remove stocks from your portfolio
- Monitor gains and losses
- View portfolio allocation by stock and sector
- Performance history visualization
- Transaction tracking

### Stock Prediction
- LSTM-based price prediction model
- Future price forecasting
- Multi-ticker comparison

## Technology Stack

- **Frontend**: Streamlit
- **Data Sources**: yfinance
- **Data Visualization**: Plotly
- **Machine Learning**: TensorFlow, Keras
- **Sentiment Analysis**: Transformers (ProsusAI/finbert)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-analyzer.git
cd stock-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Usage

1. Select a page from the sidebar: View Page, Model Page, or Portfolio Management
2. Choose a stock ticker or enter a custom one
3. Explore different analysis features like technical indicators, sentiment analysis, and price prediction
4. Manage your portfolio by adding stocks and tracking performance

## License

MIT

---
*Note: This project is for educational purposes only and should not be used for financial advice.*
