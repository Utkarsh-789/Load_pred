import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Loan Approval Prediction")
st.write("Enter applicant details to predict loan approval status.")

# -------------------------------
# Load trained model
# -------------------------------
MODEL_PATH = "loan_model.pkl"   # change if needed

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please check the path.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -------------------------------
# Input section
# -------------------------------
st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
    loan_amount = st.number_input("Loan Amount", min_value=0, step=10)
    loan_term = st.number_input("Loan Term (months)", min_value=0, step=12)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Loan Status"):
    input_data = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area]
    })

    try:
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")

    except Exception as e:
        st.error("Prediction failed. Ensure preprocessing matches training.")
        st.write(e)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Loan Prediction System • Streamlit App")
