import streamlit as st
from joblib import load
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import numpy as np
import pandas as pd

# Load the original dataset
original_data = pd.read_csv('cleaned_house_data.csv')  # Change the file path to your dataset

sample_data_ocean = house_data_ocean.sample(n=100, random_state=42)
X = sample_data_ocean.drop(columns=['median_house_value','ocean_proximity'])
y = sample_data_ocean['median_house_value'] / 10000
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


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

# Make predictions
y_pred_rf = randomForestModel.predict(X_test_scaled)

# Calculate metrics
mape_rf = MAPE(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Make predictions
y_pred_gp = gaussianProcessModel.predict(X_test_scaled)

# Calculate metrics
mape_gp = MAPE(y_test, y_pred_gp)
r2_gp = r2_score(y_test, y_pred_gp)
rmse_gp = np.sqrt(mean_squared_error(y_test, y_pred_gp))
mae_gp = mean_absolute_error(y_test, y_pred_gp)
mse_gp = mean_squared_error(y_test, y_pred_gp)

# Make predictions
y_pred_lr = linearRegressionModel.predict(X_test_scaled)

# Calculate metrics
mape_lr = MAPE(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)

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

        # Combine features for linear regression
        input_features_lr = np.array([[ocean_proximity_encoded]])

        # Predict house prices using Linear Regression
        predicted_price_lr = linearRegressionModel.predict(input_features_lr)
        predicted_actual_price_lr = predicted_price_lr * 10000  # Convert to actual price

        # Display predictions and accuracy
        st.write(f"Distance to the ocean: {distance_to_ocean:.2f} miles")
        st.write("\n====================\n")

        data = {
            "Algorithm": ["Random Forest", "Gaussian Process", "Linear Regression"],
            "Predicted Price": [
                predicted_actual_price_rf[0],
                predicted_actual_price_gp[0],
                predicted_actual_price_lr[0]
            ],
            "MAPE": [
                mape_rf,
                mape_gp,
                mape_lr
            ],
            "R-Squared": [
                r2_rf,
                r2_gp,
                r2_lr
            ],
            "RMSE": [
                rmse_rf,
                rmse_gp,
                rmse_lr
            ],
            "MAE": [
                mae_rf,
                mae_gp,
                mae_lr
            ],
            "MSE": [
                mse_rf,
                mse_gp,
                mse_lr
            ]
        }

        metrics_df = pd.DataFrame(data)

        # Display the metrics in a table
        st.write(metrics_df)
    else:
        st.error("Address not found. Please enter a valid address.")
