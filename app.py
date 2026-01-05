import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("loan_model.pkl")

st.title("🏦 Loan Eligibility Predictor")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict"):
    # 🔹 Convert inputs to model format (ENCODING)
    input_data = pd.DataFrame({
        "Gender": [1 if gender == "Male" else 0],
        "Married": [1 if married == "Yes" else 0],
        "Dependents": [3 if dependents == "3+" else int(dependents)],
        "Education": [1 if education == "Graduate" else 0],
        "Self_Employed": [1 if self_employed == "Yes" else 0],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [
            2 if property_area == "Urban"
            else 1 if property_area == "Semiurban"
            else 0
        ]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 'Y' or prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
