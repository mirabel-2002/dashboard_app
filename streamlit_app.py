import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import panel as pn
import folium
from geopy.geocoders import Nominatim

# Define the website URL
website = 'https://www.cars.com/shopping/results/?makes[]=&maximum_distance=all&models[]=&page=5&stock_type=all&zip='

# Initialize an empty list to store data from all pages
results = []

# Iterate over the desired pages
for page_num in range(1, 7): # Adjust the range as needed

    # Construct the URL for the current page
    web = website + f'&page={page_num}'

    # Send a GET request to the website
    response = requests.get(web)

    # Check if the request was successful
    if response.status_code == 200:

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all vehicle cards on the page
        vehicle_cards = soup.find_all('div', {'class': 'vehicle-card'})

        # Append the vehicle cards to the list
        results.extend(vehicle_cards)

    else:
        print(f"Failed to retrieve data from page {page_num}")

# Print the total number of data entries across all pages
total_entries = len(results)
st.write("Total number of data entries from all pages:", total_entries)

# Extracting data from results
name = []
location = []
dealer_name = []
rating = []
review_count = []
price = []

for result in results:
    # name
    try:
        name.append(result.find('h2').get_text())
    except:
        name.append('n/a')

    # mileage
    try:
        location.append(result.find('div', {'class':'miles-from'}).get_text())
    except:
        location.append('n/a')

    # dealer_name
    try:
        dealer_name.append(result.find('div', {'class':'dealer-name'}).get_text().strip())
    except:
        dealer_name.append('n/a')

    # rating
    try:
        rating.append(result.find('span', {'class':'sds-rating__count'}).get_text())
    except:
        rating.append('n/a')

    # review_count
    try:
        review_count.append(result.find('span', {'class':'sds-rating__link'}).get_text())
    except:
        review_count.append('n/a')
    
    # price 
    try:
        price.append(result.find('span', {'class': 'primary-price'}).get_text())
    except:
        price.append('n/a')

# Create a DataFrame
car_dealer = pd.DataFrame({'Name': name, 'Location': location, 'Dealer Name': dealer_name,
                           'Rating': rating, 'Review Count': review_count, 'Price': price})

# Display the DataFrame
st.write("Car Dealer Data:")
st.dataframe(car_dealer)
