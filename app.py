import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('linear_regression_model.pkl')

st.title('🛒 Ecommerce Customer Spend Predictor')

st.write("Enter the customer information below:")

# Input fields
avg_session_length = st.number_input('Average Session Length')
time_on_app = st.number_input('Time on App')
time_on_website = st.number_input('Time on Website')
length_of_membership = st.number_input('Length of Membership')

# Predict button
if st.button('Predict'):
    input_data = np.array([[avg_session_length, time_on_app, time_on_website, length_of_membership]])
    prediction = model.predict(input_data)
    st.success(f'💰 Predicted Yearly Amount Spent: **${prediction[0]:.2f}**')
