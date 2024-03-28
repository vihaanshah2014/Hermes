import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf
import datetime as dt

from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def load_data(company, prediction_days):
    """
    Load historical data from Yahoo Finance, adjusting the start date based on the prediction_days parameter.
    """
    max_look_back_years = 8
    end = dt.datetime.now()
    start = max(end - relativedelta(years=max_look_back_years), end - relativedelta(days=prediction_days * 50))
    return yf.download(company, start=start, end=end)


def prepare_data(data, prediction_days):
    """
    Prepare data for model training, including scaling and creating training datasets.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    return np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)), y_train, scaler


def build_model(input_shape):
    """
    Build and compile the LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, x_train, y_train, epochs):
    """
    Train the LSTM model.
    """
    model.fit(x_train, y_train, epochs=epochs, batch_size=32)


def test_model(model, company, scaler, prediction_days, test_start, test_end):
    """
    Test the model and predict prices using the test dataset.
    """
    test_data = yf.download(company, start=test_start, end=test_end)
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    return model.predict(np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))), test_data['Close'].values


def plot_results(actual_prices, predicted_prices, company):
    """
    Plot the actual vs. predicted prices.
    """
    plt.plot(actual_prices, color="black", label="Actual Price")
    plt.plot(predicted_prices, color="green", label="Predicted Price")
    plt.title(f"{company} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()


# Main execution flow
company = 'BTC-USD'
prediction_days = 60
epochs = 2  # Can be parameterized for expansion

data = load_data(company, prediction_days)
x_train, y_train, scaler = prepare_data(data, prediction_days)
model = build_model((x_train.shape[1], 1))
train_model(model, x_train, y_train, epochs)

# Set up testing dates based on the recent data available
test_start = dt.datetime(2010, 1, 1)
test_end = dt.datetime.now()

predicted_prices, actual_prices = test_model(model, company, scaler, prediction_days, test_start, test_end)
plot_results(actual_prices, predicted_prices, company)
