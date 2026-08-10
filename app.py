import streamlit as st
import numpy as np
import joblib

# Load the trained Linear Regression model
model = joblib.load('best_gradient_boosting_model.pkl')
# Streamlit app

st.title("🏥 Insurance Charges Prediction App")
st.subheader("Enter the following details:")

claim_amount = st.number_input("Claim Amount", min_value=0.0, format="%.2f")
no_of_past_hospitalization=st.number_input("Number of Past Hospitalizations", min_value=0)
past_consultations = st.number_input("Number of Past Consultations", min_value=0)
hospital_expenditure = st.number_input("Hospital Expenditure", min_value=0.0, format="%.2f")
annual_salary = st.number_input("Annual Salary", min_value=0.0,format="%.2f")
children = st.number_input("Number of Children", min_value=0,max_value=5)
smoker = st.selectbox("Is the person a smoker?", ["No", "Yes"])
sex=st.selectbox("Male/Female",['Male','Female'])

# Encode smoker (0 = No, 1 = Yes)
smoker_encoded = 1 if smoker == "Yes" else 0
sex_encoded= 1 if sex=="Female" else 0
# Predict button
if st.button("Predict Insurance Charges"):
    # Create input array
    input_data = np.array([[sex_encoded,smoker_encoded,children,no_of_past_hospitalization,claim_amount,past_consultations,hospital_expenditure,annual_salary]]) # predict for 1 person, 6 features

    # Make prediction
   # prediction = model.predict(input_data)[0]
    prediction1=model.predict(input_data)[0]

    #st.success(f"💰 Estimated Insurance Charges: ₹{prediction:,.2f}")
    st.success(f"💰 Estimated Insurance Charges: ₹{prediction1:,.2f}")
