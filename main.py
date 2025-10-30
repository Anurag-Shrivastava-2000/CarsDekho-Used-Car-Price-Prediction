import pickle
import pandas as pd
import numpy as np
import streamlit as st
import xgboost

# ---------------- Load Model and Encoders ----------------
cars_df = pd.read_csv('cardekho.csv')

with open('xgbregression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('oe.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('cat_features.pkl', 'rb') as f:
    cat_features = pickle.load(f)

# ---------------- Streamlit Page Setup ----------------
st.set_page_config(
    page_title="Car Price Predictor 🚗",
    page_icon="🚘",
    layout="centered"
)

# ---------------- Title Section ----------------
st.title("🚗 Car Selling Price Prediction App")
st.markdown("### Get an instant, AI-powered estimate for your car’s resale value!")
st.markdown("---")

# ---------------- Input Section ----------------
st.header("🔧 Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    car_model = st.text_input(
        "Car Name & Model",
        placeholder="e.g. Maruti Swift Dzire VDI"
    )
    year = st.number_input("Year of Manufacture", min_value=1970, max_value=2025, value=2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
    fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG'])

with col2:
    seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual', 'Trustmark Dealer'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    owner = st.selectbox(
        "Owner Type",
        ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']
    )
    mileage = st.number_input("Mileage (km/ltr/kg)", min_value=0.0, step=0.5)
    engine = st.number_input("Engine Capacity (CC)", min_value=624.0, step=10.0)
    max_power = st.number_input("Max Power (bhp)", min_value=0.0, step=1.0)
    seats = st.number_input("Seats", min_value=1, max_value=10, value=5)

st.markdown("---")

# ---------------- Prepare Input ----------------
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
    'car_model': [car_model.lower()]
})

# Encode and scale
car[cat_features] = encoder.transform(car[cat_features])
X_input = scaler.transform(car)

# ---------------- Predict Button ----------------
st.markdown("### 🚀 Click below to get your car's estimated price")

if st.button("🔮 Predict Selling Price"):
    y_pred = model.predict(X_input)
    price = y_pred[0]

    # Dynamic message based on price range
    if price < 200000:
        msg = "💡 This looks like an older or high-mileage car."
    elif price < 800000:
        msg = "✅ A good mid-range car — well maintained!"
    else:
        msg = "🔥 Premium car detected! Expect strong resale value."

    # Display results
    st.success(f"💰 **Estimated Selling Price:** ₹{price:,.0f}")
    st.markdown(msg)

    # Display summary card
    with st.expander("🧾 View Entered Details"):
        st.info(
            f"**Car Model:** {car_model}\n"
            f"**Year:** {year}\n"
            f"**Fuel:** {fuel}\n"
            f"**Transmission:** {transmission}\n"
            f"**Owner:** {owner}\n"
            f"**KM Driven:** {km_driven:,}\n"
            f"**Mileage:** {mileage} km/l\n"
            f"**Engine:** {engine} CC\n"
            f"**Power:** {max_power} bhp\n"
            f"**Seats:** {seats}"
        )

    st.balloons()
else:
    st.info("👆 Fill in the details and click **Predict Selling Price** to continue.")
