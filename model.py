import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
# Force TensorFlow to use CPU by setting environment variables before importing TF
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings("ignore")

# Additional configuration to ensure CPU usage
try:
    # Disable all GPUs explicitly
    tf.config.set_visible_devices([], 'GPU')
    # Make sure TensorFlow doesn't use GPU memory
    tf.keras.backend.clear_session()
    print("GPU disabled, using CPU for model training.")
except Exception as e:
    print(f"Note: {e}")
    print("Proceeding with CPU operations.")

class Model:
    def __init__(self, data, ticker=None):
        self.data = data
        self.ticker = ticker
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def prepare_data(self, look_back=1):
        scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))
        def create_dataset(dataset):
            X, Y = [], []
            for i in range(len(dataset) - look_back):
                a = dataset[i:(i + look_back), 0]
                X.append(a)
                Y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(Y)
        
        X, Y = create_dataset(scaled_data)
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        return X, Y

    def train_lstm(self, epochs=5, batch_size=1):
        X, Y = self.prepare_data()
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(1, 1)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)

    def make_predictions(self):
        X, _ = self.prepare_data()
        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions

    def forecast_future(self, days=5):
        last_value = self.data['Close'].values[-1:].reshape(-1, 1)
        last_scaled = self.scaler.transform(last_value)
        future_predictions = []
        for _ in range(days):
            prediction = self.model.predict(last_scaled.reshape(1, 1, 1))[0]
            future_predictions.append(prediction)
            last_scaled = prediction  
        future_predictions = self.scaler.inverse_transform(future_predictions)
        return future_predictions
        
class MultiTickerModel:
    def __init__(self):
        self.models = {}
        self.data = {}
        
    def add_ticker_data(self, ticker, data):
        """Add stock data for a specific ticker"""
        self.data[ticker] = data
        self.models[ticker] = Model(data, ticker)
        
    def train_all_models(self, epochs=5, batch_size=1):
        """Train models for all added tickers"""
        for ticker, model in self.models.items():
            model.train_lstm(epochs=epochs, batch_size=batch_size)
            
    def train_model(self, ticker, epochs=5, batch_size=1):
        """Train model for a specific ticker"""
        if ticker in self.models:
            self.models[ticker].train_lstm(epochs=epochs, batch_size=batch_size)
            return True
        return False
    
    def get_predictions(self, ticker):
        """Get predictions for a specific ticker"""
        if ticker in self.models:
            return self.models[ticker].make_predictions()
        return None
    
    def get_forecast(self, ticker, days=5):
        """Get future forecast for a specific ticker"""
        if ticker in self.models:
            return self.models[ticker].forecast_future(days=days)
        return None
    
    def compare_tickers(self, tickers, days=5):
        """Compare forecasts for multiple tickers"""
        results = {}
        for ticker in tickers:
            if ticker in self.models:
                predictions = self.models[ticker].make_predictions()
                forecast = self.models[ticker].forecast_future(days=days)
                results[ticker] = {
                    'predictions': predictions, 
                    'forecast': forecast
                }
        return results