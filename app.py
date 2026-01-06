import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import joblib

# Load model & encoder
model = joblib.load("car_model_knn.pkl")
encoder = joblib.load("encoders.pkl")

st.title("Car Evaluation Prediction (KNN)")

buying = st.selectbox("Buying Price", encoder.categories_[0])
maint = st.selectbox("Maintenance Price", encoder.categories_[1])
doors = st.selectbox("Number of Doors", encoder.categories_[2])
persons = st.selectbox("Number of Persons", encoder.categories_[3])
lug_boot = st.selectbox("Luggage Boot Size", encoder.categories_[4])
safety = st.selectbox("Safety Level", encoder.categories_[5])

if st.button("Predict"):
    input_data = [[buying, maint, doors, persons, lug_boot, safety]]
    input_encoded = encoder.transform(input_data)
    prediction = model.predict(input_encoded)

    st.success(f"Prediction: {prediction[0]}")
