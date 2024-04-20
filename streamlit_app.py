#!/usr/bin/env python
# coding: utf-8

# In[136]:


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


# In[137]:


website =  'https://www.cars.com/shopping/results/?makes[]=&maximum_distance=all&models[]=&page=5&stock_type=all&zip='


# In[138]:


# Initialize an empty list to store data from all pages
all_data = []

# Iterate over the desired pages
for page_num in range(1, 7):
    # Construct the URL for the current page
    web = website + f'&page={page_num}'


# In[139]:


#Get Request from the website varialbles

response = requests.get(website)


# In[140]:


response.status_code  # Check if the request was successful


# In[141]:


#Soup object - Parse the HTML content of the page

soup = BeautifulSoup(response.content, 'html.parser')
soup


# In[142]:


# Results of data available on each page 

data_entry = soup.find_all('div', {'class' : 'vehicle-card'})
total_count = len(data_entry)
                  #results
print(f"Pages {page_num}: {total_count} data entries per page.")



# In[143]:


# Iterate over the desired pages
for page_num in range(1, 7):  # Adjust the range as needed
    # Construct the URL for the current page
    web = website + f'&page={page_num}'
    
    # Send a GET request to the website
    response = requests.get(web)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all vehicle cards on the page
        total_count = soup.find_all('div', {'class': 'vehicle-card'})
        
        # Count the number of vehicle cards on the page
        results = len(total_count)
        
        # Print the number of data entries on the current page
        print(f"Page {page_num}: {results} data entries")
    else:
        print(f"Failed to retrieve data from page {page_num}")


# In[144]:


# Initialize an empty list to store data from all pages
results = []

# Iterate over the desired pages
for page_num in range(1, 7):  # Adjust the range as needed
    
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
print("Total number of data entries from all pages:", total_entries)


# In[145]:


results[0]


# In[146]:


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
    
    #price 
    try:
        price.append(result.find('span', {'class':'primary-price'}).get_text())
    except:
        price.append('n/a')    


# In[147]:


# dictionary
car_dealer = pd.DataFrame({'Name': name, 'Location':location, 'Dealer Name':dealer_name,
                                'Rating': rating, 'Review Count': review_count, 'Price': price})


# In[148]:


car_dealer


# In[150]:


car_dealer

# Convert the plot to an image
plt_image = plt.gcf()

# Display the plot in Streamlit
st.pyplot(plt_image)


# In[151]:


# step1             #### *************** Missing values treatement ************************ ####

                    # Cleaning the dataframe by removing any rows with missing values


complete_df = pd.DataFrame(car_dealer)                                                      


complete_df.replace(['n/a', 'N/A', 'Not Priced', 'Not priced'], pd.NA, inplace=True)        # Replace 'n/a', 'N/A', 'NaN', 'Not Priced', 
                                                                                            #and 'Not priced' with NaN


complete_df.dropna(inplace=True)                                                            # Drop rows with missing values

print(complete_df)


# In[152]:


cleanD = complete_df.copy()                                                       # Make a copy of the DataFrame


##Review Count
cleanD['Review Count'] = cleanD['Review Count'].replace('n/a', 0)                  # Replace 'n/a' with 0 
cleanD['Review Count'] = cleanD['Review Count'].str.extract('(\d+)') 

##Price
cleanD['Price'] = cleanD['Price'].str.replace(r'\$', '').str.replace(r',','')      # Remove '$' and commas

# Convert prices from dollars to rands
exchange_rate = 15.5
cleanD['Price in SA-Rands (ZAR)'] = (cleanD['Price'].astype(float) * exchange_rate).round(2)
cleanD.drop(columns=['Price'], inplace=True)    # Drop the original price column


###Location
cleanD['Location'] = cleanD['Location'].str.replace(r'\n\n|\n|,', '')              # Remove'\n\n', '\n', and commas

print(cleanD)                                                                      # Display the updated DataFrame


# In[153]:


cleanD['Price in SA-Rands (ZAR)'] = pd.to_numeric(cleanD['Price in SA-Rands (ZAR)'], errors='coerce').fillna(0)        # Convert Price column to numeric type
cleanD['Review Count'] = pd.to_numeric(cleanD['Review Count'], errors='coerce').fillna(0)                                # Convert Review Count column to numeric type
cleanD['Rating'] = pd.to_numeric(cleanD['Rating'], errors='coerce').fillna(0)                                            # Convert Rating column to numeric type
print(cleanD.dtypes)


# In[154]:


# Finding the five summary of the dataframe for each variable.

cleanD.describe()


# In[155]:


cleanD.describe()

# Convert the plot to an image
plt_image = plt.gcf()
# Display the plot in Streamlit
st.pyplot(plt_image)


# In[156]:


# Step 5                 ### ********* Identifying outliers ********** ###

most_expensive_index = cleanD['Price in SA-Rands (ZAR)'].idxmax()                   # Find the index of the most expensive car


cheapest_index = cleanD['Price in SA-Rands (ZAR)'].idxmin()                          # Find the index of the cheapest car

# Get information for the most expensive car
most_expensive_car_info = cleanD.loc[most_expensive_index, ['Name', 'Price in SA-Rands (ZAR)', 'Rating', 'Review Count', 'Location']]

# Get information for the cheapest car
cheapest_car_info = cleanD.loc[cheapest_index, ['Name', 'Price in SA-Rands (ZAR)', 'Rating', 'Review Count', 'Location']]

print("Most Expensive Car:")
print(most_expensive_car_info)
print("\nCheapest Car:")
print(cheapest_car_info)

# Convert the plot to an image
plt_image = plt.gcf()
# Display the plot in Streamlit
st.pyplot(plt_image)


# In[158]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'cleanD' is your DataFrame containing the car data
# Create a box plot of the price, review count, and rating
plt.figure(figsize=(10, 6))  # Set the size of the plot
sns.boxplot(data=cleanD['Price in SA-Rands (ZAR)'])

# Set the title and labels
plt.title('Box Plot of Car Price')
plt.ylabel('Value')
plt.xlabel('Features')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()
#st.pyplot()

# Convert the plot to an image
plt_image = plt.gcf()
# Display the plot in Streamlit
st.pyplot(plt_image)


# In[159]:


### List of all car dealers 
### Data visualization using a barchart for comparing the number of cars posted for sell by each car dealer ###

dealer_list = cleanD['Dealer Name']

print("Table of cars-dealer name:")     # Display the table
print(dealer_list)


# Set the plot size
plt.figure(figsize=(10, 6))

# Create the bar plot
ax = sns.countplot(data=cleanD, x="Dealer Name", order = cleanD['Dealer Name'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

plt.xlabel('Dealer Name')                           # Set labels and title
plt.ylabel('Number of cars sold')
plt.title('Number of Cars for sale by Dealer')

plt.show()                                          # Show the plot
#st.pyplot()

# Convert the plot to an image
plt_image = plt.gcf()
# Display the plot in Streamlit
st.pyplot(plt_image)


# In[160]:


#matplotlib.rcParams['figure.figsize'] = (20,5)
f,axes = plt.subplots(1,2, figsize=(20, 5))
sns.displot(data=cleanD, x='Review Count', color='orange', kde=True)
sns.violinplot( x=cleanD["Review Count"], color = 'orange', ax= axes[0])
sns.boxplot(x=cleanD["Review Count"], color='orange', ax=axes[1])
plt.title('Review Count distplot distribution')
axes[0].title.set_text('Review Count distribution Violin plot')
axes[1].title.set_text('Review Count distribution Box plot')

plt.show()

# Convert the plot to an image
plt_image = plt.gcf()
# Display the plot in Streamlit
st.pyplot(plt_image)


# In[133]:


# Step 4           ### ********** Bivariate analysis ********** ###
                   # Is there a correlation between price of the car and the other variables? 
cleanD.corr()


# In[134]:


cleanD.corr()

# Convert the plot to an image
plt_image = plt.gcf()
# Display the plot in Streamlit
st.pyplot(plt_image)


# In[71]:





# In[161]:


#Step 4       ### ********** Bivariate analysis ********** ###
              # Data visualization - using scatterplot to measure the relationships between two variables (price and rating or review). 

import numpy as np
from sklearn.linear_model import LinearRegression

# Assuming cleanD DataFrame is already defined

# Create a figure and axis object
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot of Rating vs Price
axes[0].scatter(cleanD['Rating'], cleanD['Price in SA-Rands (ZAR)'], color='blue', alpha=0.5)
axes[0].set_title('Rating vs Price')
axes[0].set_xlabel('Rating')
axes[0].set_ylabel('Price in SA-Rands (ZAR)')

# Fit linear regression model for Rating vs Price
X_rating = cleanD['Rating'].values.reshape(-1, 1)
y_price = cleanD['Price in SA-Rands (ZAR)'].values
reg_rating = LinearRegression().fit(X_rating, y_price)
y_pred_rating = reg_rating.predict(X_rating)
axes[0].plot(cleanD['Rating'], y_pred_rating, color='red')

# Scatter plot of Review Count vs Price
axes[1].scatter(cleanD['Review Count'], cleanD['Price in SA-Rands (ZAR)'], color='green', alpha=0.5)
axes[1].set_title('Review Count vs Price')
axes[1].set_xlabel('Review Count')
axes[1].set_ylabel('Price in SA-Rands (ZAR)')

# Fit linear regression model for Review Count vs Price
X_review_count = cleanD['Review Count'].values.reshape(-1, 1)
reg_review_count = LinearRegression().fit(X_review_count, y_price)
y_pred_review_count = reg_review_count.predict(X_review_count)
axes[1].plot(cleanD['Review Count'], y_pred_review_count, color='red')

# Adjust layout
plt.tight_layout()
#Show the plot
plt.show()

#st.pyplot()

# Convert the plot to an image
plt_image = plt.gcf()
# Display the plot in Streamlit
st.pyplot(plt_image)


# In[77]:


### Data visualization for comparison using a barchart or line) 

# Set the size of the plot
plt.figure(figsize=(15, 6))  # Adjust the width and height as needed

ax = sns.barplot(data=cleanD, x="Name", y='Price in SA-Rands (ZAR)')

# Rotate the y-axis labels vertically
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

# Show the plot
plt.show()

# Convert the plot to an image
plt_image = plt.gcf()
# Display the plot in Streamlit
st.pyplot(plt_image)
st.pyplot()


# In[162]:


#### Data visualization for compositions - A pie-chart reviewing the rating ####

rating_counts = cleanD['Rating'].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=140, shadow = True)
plt.title('Distribution of Ratings')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Convert the plot to an image
plt_image = plt.gcf()
# Display the plot in Streamlit
st.pyplot(plt_image)


# In[169]:


import folium
from geopy.geocoders import Nominatim
import pandas as pd
import re

# Load the data
data = pd.DataFrame(cleanD)

# Geocoding the locations
geolocator = Nominatim(user_agent="my_geocoder")

# Define a function to handle geocoding and extract coordinates
def get_coordinates(location):
    geo_result = geolocator.geocode(location)
    if geo_result:
        return (geo_result.latitude, geo_result.longitude)
    else:
        return None

# Apply the function to each location to get coordinates
data['Coordinates'] = data['Location'].apply(get_coordinates)

# Create a map centered around the USA
usa_map = folium.Map(location=[37.0902, -95.7129], zoom_start=4)

# Add markers for each dealership with additional information in the pop-up
for index, row in data.iterrows():
    # Remove <b> and </b> tags using regular expressions
    popup_content = f"<b>Dealer Name:</b> {row['Dealer Name']}<br>" \
                    f"<b>Car Name:</b> {row['Name']}<br>" \
                    f"<b>Price:</b> {row['Price in SA-Rands (ZAR)']}<br>" \
                    f"<b>Location:</b> {row['Location']}\n"  # Add \n to print each row in a new line
    
    clean_popup_content = re.sub(r'<.*?>', '', popup_content)
    
    # If coordinates are available, add marker to the map
    if row['Coordinates']:
        folium.Marker(location=row['Coordinates'], popup=folium.Popup(clean_popup_content, parse_html=True)).add_to(usa_map)

# Display the map
usa_map

# Convert the plot to an image
plt_image = plt.gcf()
Display the plot in Streamlit
st.pyplot(plt_image)


# In[170]:


# Variable creation 


# In[177]:


# Extract year from car name
cleanD['Year of Car Model'] = cleanD['Name'].str.extract(r'(\d{4})').astype(int)

# 1. Car Age
current_year = 2024
cleanD['Car_Age'] = current_year - cleanD['Year of Car Model']

# 2. Price Category
price_bins = [0, 500000, 1000000, 1500000, float('inf')]
price_labels = ['Low', 'Medium', 'High', 'Luxury']
cleanD['Price_Category'] = pd.cut(cleanD['Price in SA-Rands (ZAR)'], bins=price_bins, labels=price_labels)

# 3. Dealer Rating Ratio
max_rating = cleanD['Rating'].max()
cleanD['Dealer_Rating_Ratio'] = cleanD['Rating'] / max_rating

# 4. Location Rating Average
location_rating_avg = cleanD.groupby('Location')['Rating'].mean()
cleanD['Location_Rating_Average'] = cleanD['Location'].map(location_rating_avg)

# 5. Review Count Category
review_bins = [0, 100, 500, 1000, float('inf')]
review_labels = ['Low', 'Medium', 'High', 'Very High']
cleanD['Review_Count_Category'] = pd.cut(cleanD['Review Count'], bins=review_bins, labels=review_labels)


# Display the transformed data
print("Transformed Data:")
print(cleanD)


# In[186]:


# Plotting the line graph for number of cars for sale based on year of car model
plt.figure(figsize=(16, 6))  # Set a larger figure size to accommodate both plots side by side

# Subplot 1: Number of cars for sale based on the manufactured year
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
car_year_frequency = cleanD['Year of Car Model'].value_counts().sort_index()
plt.plot(car_year_frequency.index, car_year_frequency.values, marker='o', linestyle='-')
plt.title('Number of Cars for Sale based on Year of Car Model')
plt.xlabel('Year of Car Model')
plt.ylabel('Number of Cars for Sale')
plt.grid(True)  # Add gridlines for better visualization

# Subplot 2: Price of Cars over the Years
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.scatter(cleanD['Year of Car Model'], cleanD['Price in SA-Rands (ZAR)'], marker='o')
plt.title('Price of Cars based on the Year of Car Model')
plt.xlabel('Year of Car Model')
plt.ylabel('Price in SA-Rands (ZAR)')
plt.grid(True)  # Add gridlines for better visualization

plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.show()

# Convert the plot to an image
#plt_image = plt.gcf()
# Display the plot in Streamlit
#st.pyplot(plt_image)


plt_path = "car_price_year_plot.png"
plt.savefig(plt_path)
st.image(plt_path)  # Display the plot in Streamlit


# In[185]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'cleanD' is your DataFrame with columns 'Price_Category'

# Calculate the frequency of price categories
price_category_frequency = cleanD['Price_Category'].value_counts()

# Create a dictionary mapping category labels to their corresponding frequencies
category_key = {'Low': 0, 'Medium': 0, 'High': 0, 'Luxury': 0}
for label, freq in price_category_frequency.items():
    category_key[label] = freq

# Plotting the bar graph
plt.figure(figsize=(8, 6))

# Plotting price category frequency with labels
plt.bar(category_key.keys(), category_key.values(), 
        color=['r', 'y', 'g', 'b'],
        label=['0 - 500000 ZAR', '500000 - 1000000 ZAR', '1000000 - 1500000 ZAR', '1500000 - 2000000 ZAR'])

plt.title('Number of Cars for Sale based on Price Categories')
plt.xlabel('Price Category key')
plt.ylabel('Numbers of Cars for Sale')
plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.legend(title='Price range')  # Display legend for the last plot

plt.show()

# Convert the plot to an image
plt_image = plt.gcf()
# Display the plot in Streamlit
st.pyplot(plt_image)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




