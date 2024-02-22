# H.E.R.M.E.S. - High-frequency Execution and Risk Management Engine for Multi-asset Strategies

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

#Load Data
company = 'SPY'

start = dt.datetime(2016,1,1)
end = dt.datetime(2024,1,1)

data = yf.download(company, start=start, end=end)

#Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days : x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units = 50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #Prediction of closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=2, batch_size=32)


# BackTesting Model to improve

#Load Test Data
test_start = dt.datetime(2023,11, 1)
test_end = dt.datetime(2024, 2, 21)

test_data = yf.download(company, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

#Making Prediction on Testing Data

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)



#Predict Next Day
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

# print(scaler.inverse_transform(real_data[-1]))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction : {prediction}")

# Assuming NEXT_DAY is the day after test_end
next_day = dt.datetime.now()  # Example date, replace with actual next day
next_day_data = yf.download(company, start=next_day, end=next_day + dt.timedelta(days=1))
actual_next_day_price = next_day_data['Close'].values[0] if not next_day_data.empty else None

# Check if we got the next day's data
if actual_next_day_price is not None:
    print(f"Actual next day price: {actual_next_day_price}")
    # Calculate the percentage difference between actual and predicted price
    prediction_accuracy = 100 - (abs(actual_next_day_price - prediction) / actual_next_day_price) * 100
    print(f"Prediction accuracy: {prediction_accuracy}%")
else:
    print("Could not retrieve next day's price. Please check the date and try again.")


#Adding plots at the end
#Plot The Tests
# plt.plot(actual_prices, color="black", label=f"Actual Price")
# plt.plot(predicted_prices, color="green", label=f"Predicted Price")
# plt.title(f"{company} Share Price")
# plt.xlabel('Time')
# plt.ylabel(f'{company} Share Price')
# plt.legend()
# plt.show()


def generate_monthly_urls(start, end):
    current = start
    urls = []
    while current <= end:
        month_str = current.strftime('%Y%m')
        url = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_real_long_term&field_tdr_date_value_month={month_str}'
        urls.append(url)
        current += relativedelta(months=1)
    return urls

def scrape_treasury_data_for_period(start, end):
    urls = generate_monthly_urls(start, end)
    all_data = []
    for url in urls:
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.findAll('table')
            if tables:
                data = pd.read_html(str(tables[0]))[0]
                data['Date'] = pd.to_datetime(data.iloc[:, 0], errors='coerce')
                all_data.append(data)
        except Exception as e:
            print(f"An error occurred while fetching {url}: {e}")
    return pd.concat(all_data).drop_duplicates().reset_index(drop=True)

# Adjust your start and end dates here to match your desired range for treasury rates
treasury_data = scrape_treasury_data_for_period(test_start, test_end)

treasury_data_filtered = treasury_data[['Date', 'LT Real Average (10> Yrs)']].dropna()
treasury_data_filtered['MA_Treasury_Rates'] = treasury_data_filtered['LT Real Average (10> Yrs)'].rolling(window=5).mean()

# Assuming 'LT Real Average (10> Yrs)' is the correct column based on the CSV data you provided
if treasury_data is not None:
    # Setup for subplot: 2 rows, 1 column
    fig, axs = plt.subplots(2, 1, figsize=(14, 14))

    # Plot Actual and Predicted Prices on the first subplot
    axs[0].plot(test_data.index, actual_prices, color="black", label="Actual Price")
    axs[0].plot(test_data.index, predicted_prices, color="green", label="Predicted Price")

    # Plot Treasury Rates as Red Dots on the same subplot for context

    
    for i, row in treasury_data_filtered.iterrows():
        if row['Date'] in test_data.index:
            stock_price = test_data.loc[row['Date']]['Close']
            axs[0].plot(row['Date'], stock_price, 'ro', label='Treasury Rate' if i == 0 else "")
            axs[0].text(row['Date'], stock_price, f"{row['LT Real Average (10> Yrs)']}%", verticalalignment='bottom', horizontalalignment='center', color='blue', fontsize=8)

    axs[0].set_title(f"{company} Share Price with Treasury Rates")
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price')
    axs[0].legend()

    # Plot Treasury Rates and Moving Average on the second subplot
    if not treasury_data_filtered.empty:
        axs[1].plot(treasury_data_filtered['Date'], treasury_data_filtered['LT Real Average (10> Yrs)'], 'ro-', label='Treasury Rate')
        axs[1].plot(treasury_data_filtered['Date'], treasury_data_filtered['MA_Treasury_Rates'], color='blue', linestyle='--', label='MA Treasury Rates')

    axs[1].set_title(f"Treasury Rates and Moving Average")
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Rate (%)')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
else:
    print("Failed to fetch treasury rates or no data available.")