# Improved script to scrape the US Treasury website for Constant Maturity Treasury Rate.

import pandas as pd
import requests
from bs4 import BeautifulSoup

def scrape(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # This will raise an exception for HTTP errors
        return response
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

def main():
    url = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_real_long_term&field_tdr_date_value_month=202402'
    response = scrape(url)

    if response:
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.findAll('table')

        # Check if the expected table is found
        if tables:
            # Process the correct table, assuming the first one is what we want (adjust this as needed)
            data = pd.read_html(str(tables[0]))[0]  # Adjust the index as per the correct table
            
            # Example data transformation
            data['Date'] = pd.to_datetime(data.iloc[:, 0])  # Assuming the first column is Date
            
            # Save the data to CSV
            data.to_csv('treasury_yield_curve_rates.csv', index=False)
            print("Data saved successfully.")
        else:
            print("No tables found on the page.")
    else:
        print("Failed to retrieve the webpage.")

if __name__ == "__main__":
    main()
