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

# # In the above code, `parse_input_data()` processes the input text to populate `sales
# _data` and `forecast_data` dictionaries with the actual sales and forecast data, re
# spectively. The `??,???` is maintained as-is until you implement a forecasting function 
# to compute the values. The function `calculate_forecast()` is currently a placeholder for 
# the actual forecasting method. You would need to replace `'??,???'` with the appropriate fore
# cast value after calculating MAD or WMA.

# # To call these functions, the code will expect standard input. To
#  test from a file, you could run the Python script and redirect the file content 
# to the script's standard input using a command like:

# bash
# python script.py < input.txt
