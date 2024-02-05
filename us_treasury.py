#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: filetype=python

# This script scrapes the US Treasury website for Constant Maturity Treasury Rate.
# The Treasury rate will be used as a risk-free interest rate, which is crucial to the VIX calculator.
# Source: https://github.com/je-suis-tm/quant-trading/blob/master/VIX%20Calculator.py

import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib3

# Disable warnings for insecure HTTPS requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Scraping function
def scrape(url):
    with requests.Session() as session:
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36'})
        response = session.get(url, verify=False)
        
        if response.status_code != 200:
            print("Failed to retrieve the webpage")
            return None
        else:
            return response

# Main execution function
def main():    
    url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield'
    response = scrape(url)

    if response:
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.findAll('table')
        
        # Assuming that the yield curve rates are in the second table
        print('Number of tables found:', len(tables))

        if len(tables) > 1:
            yield_curve_table = tables[1]
            # ... further processing
        else:
            print("The expected table was not found on the page.")
            # Maybe handle the problem or exit


        yield_curve_table = tables[1]
        data = pd.read_html(str(yield_curve_table))[0]

        
        
        # Cleanse and transform the data if necessary
        data = data.melt(id_vars=['Date'], var_name='maturity')
        
        # Convert the 'Date' column to datetime format
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Save the data as CSV in the current working directory
        data.to_csv('treasury_yield_curve_rates.csv', index=False)
    else:
        print("Webpage response was not successful.")

if __name__ == "__main__":
    main()
