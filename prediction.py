import streamlit as st
from joblib import load
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import folium

# Load the trained Random Forest model
randomForestModel = load('random_forest.joblib')

# Title of the Streamlit app
st.title('House Price Predictor')

# User input field for the address
house_address = st.text_input('Enter the address of the house:')

# Ocean coordinates
ocean_lat = 37.7749
ocean_lon = -122.4194

# Initialize geolocator
geolocator = Nominatim(user_agent="my_geocoder")

# Predict house price
if st.button('Predict Price'):
    # Geocode the address to get latitude and longitude
    latitude, longitude = geocode_address(house_address)
    if latitude and longitude:
        # Calculate distance to the ocean
        distance_to_ocean = geodesic((latitude, longitude), (ocean_lat, ocean_lon)).miles
        st.write(f"Distance to the ocean: {distance_to_ocean:.2f} miles")

        # Display map with house location and ocean coordinates
        m = folium.Map(location=[latitude, longitude], zoom_start=10)
        folium.Marker([latitude, longitude], popup="House Location").add_to(m)
        folium.Marker([ocean_lat, ocean_lon], popup="Ocean").add_to(m)
        st.write(m)
        
        # Combine features
        ocean_proximity_encoded = determine_ocean_proximity(distance_to_ocean)
        input_features = np.array([[longitude, latitude, ocean_proximity_encoded]])

        # Predict house prices
        predicted_price = randomForestModel.predict(input_features)
        predicted_actual_price = predicted_price * 10000  # Convert to actual price
        st.write(f"Predicted house price: ${predicted_actual_price[0]:,.2f}")
    else:
        st.error("Address not found. Please enter a valid address.")
