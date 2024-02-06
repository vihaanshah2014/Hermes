import sys

def parse_input_data(input_data):
    lines = input_data.strip().split('\n')
    sales_data = {}
    forecast_data = {}
    actual_sales_flag = False
    forecast_flag = False

    for line in lines:
        if 'ACTUAL SALES' in line:
            actual_sales_flag = True
            forecast_flag = False
            continue
        
        if 'FORECAST' in line:
            actual_sales_flag = False
            forecast_flag = True
            continue

        if actual_sales_flag and line.strip():
            month, sales = line.rsplit(' ', 1)
            sales_data[month] = int(sales.replace(',', ''))
        
        if forecast_flag and line.strip():
            month, forecast = line.rsplit(' ', 1)
            if '??' not in forecast:
                forecast_data[month] = int(forecast.replace(',', ''))
            else:
                forecast_data[month] = forecast  # Keep the '???' for now

    return sales_data, forecast_data

def calculate_forecast(sales_data, method='MAD', params=None):
    # Placeholder for forecasting calculation, compute MAD or WMA
    return {'January 2023': '??,???'}

# Read from standard input
input_data = sys.stdin.read()

# Parse the input
sales_data, forecast_data = parse_input_data(input_data)

# You can add the logic for MAD and WMA here using the sales_data info
# Assume that method is passed as a string: 'MAD' or 'WMA', and based on the method,
# Use the appropriate formula. 'params' can be the weights for the WMA, for example.
forecast_data.update(calculate_forecast(sales_data, method='MAD'))

# Assuming we want to print the result to check
print(sales_data)
print(forecast_data)


