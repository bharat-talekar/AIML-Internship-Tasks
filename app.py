import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("LR_model.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")

# Title
st.title("🚗 Ford Car Price Predictor")

st.write("Enter the car details below.")

# Inputs
model_name = st.text_input("Model Name", "Fiesta")

year = st.number_input("Manufacturing Year", 1996, 2025, 2018)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic", "Semi-Auto"]
)

mileage = st.number_input("Mileage", 0, 300000, 30000)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)

tax = st.number_input("Road Tax", 0, 600, 150)

mpg = st.number_input("MPG", 0.0, 100.0, 55.0)

engine_size = st.number_input(
    "Engine Size",
    0.8,
    5.0,
    1.5
)

# Predict Button
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "model": [model_name],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuel_type],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size]
    })

    # One Hot Encoding
    input_data = pd.get_dummies(input_data)

    # Match training columns
    input_data = input_data.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    # Scale numerical columns
    numerical_columns = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize"
    ]

    input_data[numerical_columns] = scaler.transform(
        input_data[numerical_columns]
    )

    # Prediction
    prediction = model.predict(input_data)

    st.success(
        f"Estimated Selling Price: £{prediction[0]:,.2f}"
    )
    #cd D:\internship\codes
    #-m streamlit run app.py