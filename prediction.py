import streamlit as st
from joblib import load
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import numpy as np

# Function to calculate Haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius = 6371  # Radius of the Earth in kilometers
    distance = radius * c
    return distance

# Define a function to geocode addresses
def geocode_address(address):
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None

# Function to calculate distance from house to ocean
def calculate_distance_to_ocean(house_coords, ocean_data):
    min_distance = float('inf')
    nearest_ocean = None
    for _, ocean_row in ocean_data.iterrows():
        ocean_coord = (ocean_row['Latitude'], ocean_row['Longitude'])
        distance = geodesic(house_coords, ocean_coord).miles
        if distance < min_distance:
            min_distance = distance
            nearest_ocean = ocean_coord
    return min_distance, nearest_ocean

# Function to determine ocean proximity based on distance
def determine_ocean_proximity(distance):
    if distance < 23.69:
        return 3  # NEAR BAY
    elif distance < 515.38:
        return 0  # <1H OCEAN
    elif distance < 846.99:
        return 1  # INLAND
    elif distance < 1319.59:
        return 4  # NEAR OCEAN
    else:
        return 2  # ISLAND

# Load the trained Random Forest model
randomForestModel = load('random_forest.joblib')

# for comparison
gaussianProcessModel = load('ocean_gaussian_process.joblib')
linearRegressionModel = load('linear_regression.joblib')

# Title of the Streamlit app
st.title('House Price Predictor')

# User input field for the address
house_address = st.text_input('Enter the address of the house:')

# Initialize geolocator
geolocator = Nominatim(user_agent="my_geocoder")

ocean_lat = 37.7749
ocean_lon = -122.4194

# Predict house price
if st.button('Predict Price'):
    # Geocode the address to get latitude and longitude
    latitude, longitude = geocode_address(house_address)
    if latitude and longitude:

        try:

            ocean_coords = (ocean_lat, ocean_lon)  # San Francisco, CA
            house_coords = (latitude, longitude)
            distance_to_ocean = geodesic(house_coords, ocean_coords).miles

            # Determine ocean proximity encoded based on distance
            ocean_proximity_encoded = determine_ocean_proximity(distance_to_ocean)

            # Combine features
            input_features = np.array([[longitude, latitude, ocean_proximity_encoded]])

            # Predict house prices using Random Forest
            predicted_price_rf = randomForestModel.predict(input_features)
            predicted_actual_price_rf = predicted_price_rf * 10000  # Convert to actual price

            # Predict house prices using Gaussian Process
            predicted_price_gp = gaussianProcessModel.predict(input_features)
            predicted_actual_price_gp = predicted_price_gp * 10000  # Convert to actual price

            # Predict house prices using Linear Regression
            predicted_price_lr = linearRegressionModel.predict(input_features)
            predicted_actual_price_lr = predicted_price_lr * 10000  # Convert to actual price

            # Display predictions and accuracy
            st.write(f"Distance to the ocean: {distance_to_ocean:.2f} miles")
            st.write("\n====================\n")
            st.write("Random Forest Regression:")
            st.write(f"Predicted house price: ${predicted_actual_price_rf[0]:,.2f}")

            st.write("Gaussian Process Regression:")
            st.write(f"Predicted house price: ${predicted_actual_price_gp[0]:,.2f}")

            st.write("Linear Regression:")
            st.write(f"Predicted house price: ${predicted_actual_price_lr[0]:,.2f}")
        except geopy.exc.GeocoderUnavailable:
            st.error("Geocoder service is currently unavailable. Please try again later.")
    else:
        st.error("Address not found. Please enter a valid address.")
