# Define the actual sales and forecast lists
actual_sales = [4035, 4335, 4410, 4170, 4020, 4035, 3900, 4143, 6456, 6942, 7056, 6672]
forecast = [3825, 3990, 4650, 4260, 3750, 4020, 4215, 4065, 6300, 6405, 7272, 6816]

# Calculate the absolute differences
absolute_differences = [abs(actual - forecast) for actual, forecast in zip(actual_sales, forecast)]

# Calculate the sum of absolute differences
sum_of_absolute_differences = sum(absolute_differences)

# Calculate Mean Absolute Deviation
mad = sum_of_absolute_differences / len(absolute_differences)

print(f"The Mean Absolute Deviation (MAD) is: {mad:.2f}")
