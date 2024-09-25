#basic graphs and volatility charts
import requests
import matplotlib.pyplot as plt
from datetime import datetime

def get_bitcoin_prices():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': '180',  # Last 6 months
        'interval': 'daily'  # Get daily prices
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        prices = data['prices']  # List of [timestamp, price]
        return prices
    else:
        print("Error fetching data")
        return []

def process_data(prices):
    dates = [datetime.fromtimestamp(price[0] / 1000.0) for price in prices]  # Convert ms to seconds
    close_prices = [price[1] for price in prices]
    
    # Calculate daily volatility (high - low)
    volatility = []
    for i in range(1, len(prices)):
        prev_price = prices[i-1][1]
        curr_price = prices[i][1]
        daily_volatility = abs(curr_price - prev_price)
        volatility.append(daily_volatility)
    
    # Add a 0 volatility for the first day to match the length of dates
    volatility.insert(0, 0)
    
    return dates, close_prices, volatility

def plot_bitcoin_prices_and_volatility(dates, close_prices, volatility):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot the closing prices as a blue line
    ax1.plot(dates, close_prices, label='Bitcoin Closing Price (USD)', color='blue', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create a second y-axis for volatility
    ax2 = ax1.twinx()
    ax2.fill_between(dates, volatility, alpha=0.3, color='red', label='Daily Volatility')
    ax2.set_ylabel('Daily Volatility (USD)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Bitcoin Closing Prices and Daily Volatility Over the Last 6 Months')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 1), bbox_transform=ax1.transAxes)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Fetch and process Bitcoin prices
    prices = get_bitcoin_prices()
    if prices:
        dates, close_prices, volatility = process_data(prices)
        plot_bitcoin_prices_and_volatility(dates, close_prices, volatility)

if __name__ == "__main__":
    main()