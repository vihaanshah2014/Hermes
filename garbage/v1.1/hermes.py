import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

class HERMES:
    def __init__(self, company='SPY', start_date='2016-01-01', end_date='2024-01-01', epochs=2):
        """
        Initialize the HERMES model with company ticker, start and end date for data, and training epochs.
        """
        self.company = company
        self.start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')
        self.epochs = epochs
        self.model = self.build_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self):
        """
        Load historical data from Yahoo Finance.
        """
        return yf.download(self.company, start=self.start_date, end=self.end_date)

    def prepare_data(self, data):
        """
        Prepare data for training, including scaling, and creating training dataset.
        """
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        prediction_days = 60
        x_train, y_train = [], []

        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        return np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)), y_train

    def build_model(self):
        """
        Build and compile the LSTM model.
        """
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, x_train, y_train):
        """
        Train the LSTM model.
        """
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=32)

    def backtest_model(self, test_start, test_end):
        """
        Backtest the model with new data and return predictions.
        """
        test_data = yf.download(self.company, start=test_start, end=test_end)
        total_dataset = pd.concat((self.load_data()['Close'], test_data['Close']), axis=0)
        model_inputs = total_dataset[len(total_dataset) - len(test_data) - 60:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = self.scaler.transform(model_inputs)

        x_test = []
        for x in range(60, len(model_inputs)):
            x_test.append(model_inputs[x - 60:x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predicted_prices = self.model.predict(x_test)
        return self.scaler.inverse_transform(predicted_prices), test_data['Close'].values

    def plot_results(self, actual_prices, predicted_prices):
        """
        Plot the actual vs predicted prices.
        """
        plt.plot(actual_prices, color="black", label="Actual Price")
        plt.plot(predicted_prices, color="green", label="Predicted Price")
        plt.title(f"{self.company} Share Price")
        plt.xlabel('Time')
        plt.ylabel(f"{self.company} Share Price")
        plt.legend()
        plt.show()

# Example usage
hermes = HERMES()
data = hermes.load_data()
x_train, y_train = hermes.prepare_data(data)
hermes.train_model(x_train, y_train)

# Define your backtesting dates
test_start = '2023-11-01'
test_end = '2024-02-21'
predicted_prices, actual_prices = hermes.backtest_model(test_start, test_end)
hermes.plot_results(actual_prices, predicted_prices)
