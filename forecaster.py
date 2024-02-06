# Define the actual sales and forecast lists
actual_sales = [4035, 4335, 4410, 4170, 4020, 4035, 3900, 4143, 6456, 6942, 7056, 6672]
forecast = [3825, 3990, 4650, 4260, 3750, 4020, 4215, 4065, 6300, 6405, 7272, 6816]

# Calculate the absolute differences
absolute_differences = []
for actual, forecast in zip(actual_sales, forecast):
    absolute_difference = abs(actual - forecast)
    absolute_differences.append(absolute_difference)

# Show the work for absolute differences
print("Absolute Differences for each month:")
for i, diff in enumerate(absolute_differences, 1):
    print(f"Month {i}: {diff}")

# Calculate the sum of absolute differences
sum_of_absolute_differences = sum(absolute_differences)

# Show the sum of the absolute differences
print(f"\nSum of Absolute Differences: {sum_of_absolute_differences}")

# Calculate Mean Absolute Deviation
mad = sum_of_absolute_differences / len(absolute_differences)

# Output the MAD with work shown
print(f"\nThe Mean Absolute Deviation (MAD) is: {mad:.2f}")
