import streamlit as st
import pickle
import pandas as pd
import os
import joblib

model_path = os.path.join("models", "churn_model.joblib")
with open(model_path, "rb") as f:
    model = joblib.load(f)

# st.title("📉 Telco Customer Churn Prediction")
st.title("Saudi Customer Churn Predictor | نظام التنبؤ بترك العملاء")

senior_citizen = st.selectbox("Senior Citizen (0=No, 1=Yes)", [0, 1], index=0)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, step=1, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, step=0.1, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, step=0.1, value=600.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
], index=0)
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=0)
tech_support = st.selectbox("Tech Support", ["Yes", "No"], index=1)
online_security = st.selectbox("Online Security", ["Yes", "No"], index=1)

input_data = pd.DataFrame({
    "SeniorCitizen": [senior_citizen],
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "Contract": [contract],
    "PaymentMethod": [payment_method],
    "InternetService": [internet_service],
    "TechSupport": [tech_support],  # Value for TechSupport
    "OnlineSecurity": [online_security],  # Value for OnlineSecurity
    "NewCustomer": [1 if tenure < 12 else 0]
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")
