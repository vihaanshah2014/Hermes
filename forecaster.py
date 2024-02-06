# Define the actual sales and forecast lists
actual_sales = [4035, 4335, 4410, 4170, 4020, 4035, 3900, 4143, 6456, 6942, 7056, 6672]
forecast = [4320, 4305, 4080, 4470, 4410, 4050, 3585, 4131, 6732, 6888, 6708, 7137]

# Calculate the absolute differences
absolute_differences = [abs(actual - forecast) for actual, forecast in zip(actual_sales, forecast)]

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
