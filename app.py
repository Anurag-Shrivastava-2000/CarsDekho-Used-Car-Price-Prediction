import pickle
import xgboost
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import streamlit as st

cars_df = pd.read_csv('cardekho.csv')

with open('xgbregression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('oe.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('cat_features.pkl', 'rb') as f:
    cat_features = pickle.load(f)

# --- Streamlit UI ---
st.title("Car Selling Price Prediction")

st.header("Enter Car Details:")

year = st.number_input("Year of Car", min_value=1970, max_value=2025, value=2015)
km_driven = st.number_input("Kilometers Driven", min_value=0)
fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG'])
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
owner = st.selectbox("Owner Type", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.number_input("Mileage (km/ltr/kg)", min_value=0.0)
engine = st.number_input("Engine (CC)", min_value=624.0)
max_power = st.number_input("Max Power (bhp)", min_value=0.0)
seats = st.number_input("Seats", min_value=1)
car_model = st.text_input("Please enter car brand and model", placeholder="e.g. Maruti Swift")

car = pd.DataFrame({
    'year': [year],
    'km_driven': [km_driven],
    'fuel': [fuel],
    'seller_type': [seller_type],
    'transmission': [transmission],
    'owner': [owner],
    'mileage(km/ltr/kg)': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats],
    'car_model': [car_model]
})

car['car_model'] = car['car_model'].str.lower()

car[cat_features] = encoder.transform(car[cat_features])
X_input = scaler.transform(car)
# --- Predict ---
if st.button("Predict Selling Price"):
    y_pred = model.predict(X_input)
    st.success(f"Predicted Selling Price: ₹{y_pred[0]:,.0f}")
