import requests
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Function to get Bitcoin price data from CoinGecko API
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

# Convert the timestamps to readable dates and separate prices
def process_data(prices):
    dates = [datetime.fromtimestamp(price[0] / 1000.0) for price in prices]  # Convert ms to seconds
    prices = [price[1] for price in prices]
    return dates, prices

# Function to plot Bitcoin prices
def plot_bitcoin_prices(dates, prices):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, prices, label='Bitcoin Price (USD)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Bitcoin Prices Over the Last 6 Months')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

# Main function
def main():
    # Fetch and process Bitcoin prices
    prices = get_bitcoin_prices()
    if prices:
        dates, prices = process_data(prices)
        plot_bitcoin_prices(dates, prices)

# Call the main function
if __name__ == "__main__":
    main()
