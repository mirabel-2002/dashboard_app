import dash
from dash import dcc, html
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Initialize Dash app
app = dash.Dash(__name__)

# Define the website URL
website = 'https://www.cars.com/shopping/results/?makes[]=&maximum_distance=all&models[]=&page=5&stock_type=all&zip='

# Initialize an empty list to store data from all pages
all_data = []

# Iterate over the desired pages
for page_num in range(1, 7):
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
        all_data.extend(vehicle_cards)
        
        # Count the number of vehicle cards on the page
        results = len(vehicle_cards)
        
        # Print the number of data entries on the current page
        print(f"Page {page_num}: {results} data entries")
    else:
        print(f"Failed to retrieve data from page {page_num}")

# Initialize lists to store scraped data
name = []
location = []
dealer_name = []
rating = []
review_count = []
price = []

# Iterate over each vehicle card and extract relevant information
for result in all_data:
    # Name
    try:
        name.append(result.find('h2').get_text())
    except:
        name.append('n/a')
    
    # Location
    try:
        location.append(result.find('div', {'class':'miles-from'}).get_text())
    except:
        location.append('n/a')
    
    # Dealer name
    try:
        dealer_name.append(result.find('div', {'class':'dealer-name'}).get_text().strip())
    except:
        dealer_name.append('n/a')
        
    # Rating
    try:
        rating.append(result.find('span', {'class':'sds-rating__count'}).get_text())
    except:
        rating.append('n/a')
    
    # Review count
    try:
        review_count.append(result.find('span', {'class':'sds-rating__link'}).get_text())
    except:
        review_count.append('n/a')
    
    # Price 
    try:
        price.append(result.find('span', {'class':'primary-price'}).get_text())
    except:
        price.append('n/a')

# Create a DataFrame from the scraped data
car_dealer = pd.DataFrame({'Name': name, 'Location':location, 'Dealer Name':dealer_name,
                            'Rating': rating, 'Review Count': review_count, 'Price': price})

# Dash layout
app.layout = html.Div([
    html.H1("Car Dealership Dashboard"),
    html.Div([
        html.H3("Data from Cars.com"),
        dcc.Graph(
            id='dealer-cars',
            figure={
                'data': [
                    {'x': car_dealer['Dealer Name'], 'type': 'histogram', 'name': 'Number of Cars'}
                ],
                'layout': {
                    'title': 'Number of Cars for Sale by Dealer',
                    'xaxis': {'title': 'Dealer Name'},
                    'yaxis': {'title': 'Number of Cars'},
                }
            }
        )
    ]),
    html.Div([
        html.H3("Car Dealer Information"),
        dcc.Graph(
            id='dealer-table',
            figure={
                'data': [
                    {
                        'type': 'table',
                        'header': dict(values=list(car_dealer.columns)),
                        'cells': dict(values=car_dealer.values.T)
                    }
                ]
            }
        )
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
