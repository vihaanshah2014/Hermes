# Step 1: Input the raw sales data
sales_data = """
July 2022 87,840
August 2022 27,882
September 2022 135,411
October 2022 47,673
November 2022 124,014
December 2022 59,580
"""

print("Step 2: Splitting the data into lines for each month.")
lines = sales_data.strip().split("\n")
for line in lines:
    print(line)

print("\nStep 3 & 4: Extracting and converting sales numbers into integers.")
# Extract numbers and convert them to integers, removing commas
sales_values = []
for line in lines:
    # Extract the last element after splitting and remove commas
    number_str = line.split()[-1].replace(",", "")
    # Convert to integer
    number = int(number_str)
    sales_values.append(number)
    print(f"Converted {number_str} to {number}")

print("\nStep 5: Calculating the sum of all sales with a running total.")
total_sales = 0
for index, value in enumerate(sales_values):
    total_sales += value
    print(f"Adding {value}: Running sum is {total_sales}")

print(f"\nTotal sales for the year: {total_sales}")

print("\nStep 6: Calculating average sales.")
# Calculate average sales
average_sales = total_sales / 6
print(f"The average monthly sales for the year 2022 is: {average_sales:.2f}")
