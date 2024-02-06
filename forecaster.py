
# Raw data with actual sales
sales_data = """
January 2022 140,240
February 2022 116,554
March 2022 78,225
April 2022 63,053
May 2022 17,467
June 2022 122,405
July 2022 16,975
August 2022 41,862
September 2022 57,281
October 2022 73,815
November 2022 33,786
December 2022 79,837
"""

# Extract numbers and convert them to integers, removing commas
sales_values = [int(line.split()[-1].replace(",", "")) for line in sales_data.strip().split("\n")]

# Calculate average sales
average_sales = sum(sales_values) / 12

print("The average monthly sales for the year 2022 is: {:.2f}".format(average_sales))
