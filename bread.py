import yfinance as yf
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_portfolio(csv_file_path):
    portfolio_data = pd.read_csv(csv_file_path)
    portfolio_weights = portfolio_data[['Symbol', 'Percent Of Account']].dropna()
    portfolio_weights['Percent Of Account'] = portfolio_weights['Percent Of Account'].str.rstrip('%').astype(float) / 100
    if 'SPAXX' in portfolio_weights['Symbol'].values:
        spaxx_weight = portfolio_weights.loc[portfolio_weights['Symbol'] == 'SPAXX', 'Percent Of Account'].values[0]
        portfolio_weights = portfolio_weights[portfolio_weights['Symbol'] != 'SPAXX']
        new_row = pd.DataFrame({'Symbol': ['SPXL'], 'Percent Of Account': [spaxx_weight]})
        portfolio_weights = pd.concat([portfolio_weights, new_row], ignore_index=True)
    portfolio_dict = dict(zip(portfolio_weights['Symbol'], portfolio_weights['Percent Of Account']))
    return portfolio_dict

def fetch_stock_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    history = stock.history(period="1y")
    return stock_symbol, history['Close']

def fetch_sp500_data():
    spy = yf.Ticker('SPY')
    history = spy.history(period="1y")
    return history['Close']

def calculate_daily_weighted_average(portfolio, df):
    weighted_prices = pd.DataFrame()

    for stock, weight in portfolio.items():
        prices = df[df['Stock'] == stock]['Prices'].values[0]
        weighted_prices[stock] = prices * weight

    daily_avg_price = weighted_prices.sum(axis=1)
    return daily_avg_price

def plot_portfolio_vs_sp500(portfolio_avg_price, sp500_prices):
    plt.figure(figsize=(10, 6))
    plt.plot(sp500_prices.index, sp500_prices / sp500_prices.iloc[0], label="S&P 500", color='red')
    plt.plot(portfolio_avg_price.index, portfolio_avg_price / portfolio_avg_price.iloc[0], label="Portfolio Average Price", color='blue')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.title("Portfolio vs S&P 500 Performance (Last Year)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Growth/Drop)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

def calculate_moving_average(prices):
    return np.mean(prices)

def predict_risk(prices):
    avg_change = np.mean(np.diff(prices) / prices[:-1]) * 100
    volatility = np.std(np.diff(prices) / prices[:-1]) * 100
    return avg_change * volatility

def predict_future_price(prices, days_ahead):
    last_price = prices.iloc[-1]
    avg_change = np.mean(np.diff(prices) / prices[:-1])
    return last_price * (1 + avg_change) ** days_ahead

def analyze_stock(stock_symbol, weight):
    stock_symbol, prices = fetch_stock_data(stock_symbol)
    moving_avg = calculate_moving_average(prices)
    risk = predict_risk(prices)
    future_price = predict_future_price(prices, 30)
    weighted_risk = risk * weight
    weighted_future_price = future_price * weight
    return {
        'Stock': stock_symbol,
        'Moving Average': moving_avg,
        'Predicted Risk': risk,
        'Weighted Risk': weighted_risk,
        'Predicted Price (30 Days)': future_price,
        'Weighted Future Price': weighted_future_price,
        'Average Price': moving_avg
    }

def identify_pain_points(df):
    pain_points = df[(df['Predicted Risk'] > 10) | (df['Predicted Price (30 Days)'] < df['Moving Average'])]
    return pain_points

def calculate_weighted_average(df, portfolio_weights):
    weighted_avg_prices = []
    for stock in df['Stock']:
        stock_avg_price = df[df['Stock'] == stock]['Average Price'].values[0]
        weight = portfolio_weights[stock]
        weighted_avg_prices.append(stock_avg_price * weight)
    total_weighted_avg_price = sum(weighted_avg_prices)
    return total_weighted_avg_price

def analyze_portfolio_with_sp500(csv_file_path):
    portfolio = load_portfolio(csv_file_path)
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_stock_data, symbol): symbol for symbol in portfolio.keys()}
        for future in futures:
            stock_symbol, prices = future.result()
            results.append({'Stock': stock_symbol, 'Prices': prices})

    df = pd.DataFrame(results)

    sp500_prices = fetch_sp500_data()

    portfolio_avg_price = calculate_daily_weighted_average(portfolio, df)

    plot_portfolio_vs_sp500(portfolio_avg_price, sp500_prices)

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_stock, symbol, weight): symbol for symbol, weight in portfolio.items()}
        for future in futures:
            result = future.result()
            results.append(result)
    df = pd.DataFrame(results)

    pain_points = identify_pain_points(df)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.colheader_justify', 'left')

    # Show pain points
    if not pain_points.empty:
        print("Pain points identified:")
        print(pain_points[['Stock', 'Moving Average', 'Predicted Risk', 'Weighted Risk', 'Predicted Price (30 Days)', 'Weighted Future Price']].to_string(index=False))
    else:
        print("No pain points identified.")

csv_file_path = './Portfolio_Positions_Oct-03-2024.csv'
analyze_portfolio_with_sp500(csv_file_path)
