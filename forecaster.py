def weighted_moving_average(sales_data, weights):
    print("Calculating the Weighted Moving Average...\n")

    # Reverse the sales data to start with the most recent month
    reversed_sales = sales_data[::-1]
    detailed_work = []

    # Calculate the weighted moving average
    weighted_values = [sales * weight for sales, weight in zip(sales_data, weights)]
    
    for sales, weight, weighted_value in zip(sales_data, weights, weighted_values):
        line = f"{weight} * {sales} (weight for month * sales for month) = {weighted_value}"
        detailed_work.append(line)
        print(line)
    
    forecast = sum(weighted_values)
    return forecast, detailed_work

# Sales data for the last 6 months of 2022
sales_data = [
    73820,   # October 2022
    33780,   # November 2022
    79830,   # December 2022
]

# Weights for the corresponding months
weights = [
    0.10,  # 3 months prior
    0.30,  # 2 months prior
    0.60,  # Previous month
]

# Calculate the forecast for January 2023
forecast, detailed_work = weighted_moving_average(sales_data, weights)

print(f"\nForecast for January 2023: {forecast:.2f}")
