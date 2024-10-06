import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import datetime
from scipy import stats

def fetch_data(tickers, start_date, end_date):
    """
    Fetches adjusted close prices for given tickers between start_date and end_date.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_volatility(price_data, window=20):
    """
    Calculates rolling volatility (standard deviation) for each ticker.
    """
    volatility = price_data.pct_change().rolling(window=window).std() * np.sqrt(252)  # Annualized volatility
    return volatility

def prepare_dataset(volatility, target):
    """
    Prepares the dataset for modeling.
    """
    # Drop rows with any NaN values
    dataset = pd.concat([volatility, target], axis=1).dropna()
    X = dataset.iloc[:, :-1].values  # Features: volatilities
    y = dataset.iloc[:, -1].values   # Target: BTC next-day price
    return X, y

def train_model(X, y):
    """
    Trains a Random Forest Regressor and evaluates its performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    print(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")

    # Calculate accuracy as percentage of predictions within 5% of actual value
    train_accuracy = np.mean(np.abs((train_predictions - y_train) / y_train) <= 0.05) * 100
    test_accuracy = np.mean(np.abs((test_predictions - y_test) / y_test) <= 0.05) * 100
    print(f"Train Accuracy (within 5%): {train_accuracy:.2f}%")
    print(f"Test Accuracy (within 5%): {test_accuracy:.2f}%")

    return model, test_accuracy

def predict_next_day(model, latest_volatility):
    """
    Predicts the next day's BTC price using the latest volatility data.
    """
    prediction = model.predict([latest_volatility])
    return prediction[0]

def analyze_trend(price_data, window=30):
    """
    Analyzes the trend of BTC prices over the given window.
    """
    btc_prices = price_data['BTC-USD'].dropna()
    if len(btc_prices) < window:
        return "Not enough data for trend analysis"

    recent_prices = btc_prices[-window:]
    x = np.arange(len(recent_prices))
    slope, _, _, _, _ = stats.linregress(x, recent_prices)

    if slope > 0:
        trend = "upward"
    elif slope < 0:
        trend = "downward"
    else:
        trend = "neutral"

    percent_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100

    return f"BTC has shown a {trend} trend over the last {window} days, with a {percent_change:.2f}% change."

def main():
    # Define tickers
    btc_ticker = 'BTC-USD'
    spy_ticker = 'SPY'
    short_spy_ticker = 'SH'  # SH is ProShares Short S&P500 ETF

    tickers = [btc_ticker, spy_ticker, short_spy_ticker]

    # Define date range
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=365) 

    # Fetch data
    print("Fetching data...")
    price_data = fetch_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    # Calculate volatility
    print("Calculating volatility...")
    volatility = calculate_volatility(price_data)

    # Prepare target: BTC next-day price
    print("Preparing target variable...")
    btc_price = price_data[btc_ticker].shift(-1)  # Next-day price
    target = btc_price.rename('BTC_Next_Day')

    # Prepare dataset
    print("Preparing dataset...")
    X, y = prepare_dataset(volatility, target)

    # Train model
    print("Training model...")
    model, accuracy = train_model(X, y)

    # Analyze trend
    print("Analyzing trend...")
    trend_analysis = analyze_trend(price_data)
    print(trend_analysis)

    # Prepare latest volatility data for prediction
    print("Preparing latest volatility data for prediction...")
    latest_volatility = volatility.iloc[-1].values
    if np.any(np.isnan(latest_volatility)):
        print("Latest volatility data contains NaN values. Cannot make a prediction.")
        return

    # Predict next day's BTC price
    print("Predicting next day's BTC price...")
    predicted_price = predict_next_day(model, latest_volatility)
    print(f"Predicted BTC Price for next day: ${predicted_price:.2f}")
    print(f"This prediction is based on a model with {accuracy:.2f}% accuracy on the test set.")

if __name__ == "__main__":
    main()