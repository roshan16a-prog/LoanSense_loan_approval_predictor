import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and preprocessors
@st.cache_resource
def load_assets():
    return joblib.load('best_model_assets.pkl')

try:
    assets = load_assets()
    model = assets['model']
    scaler = assets['scaler']
    le_dict = assets['le_dict']
    le_target = assets['le_target']
    feature_names = assets['feature_names']
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Predictor")
st.markdown("Enter the application details below to check for loan eligibility.")

# Create input form
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=None)
        coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=None)
        age = st.number_input("Age", min_value=18, max_value=100, value=None)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=None)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=None)
        existing_loans = st.number_input("Number of Existing Loans", min_value=0, max_value=10, value=None)

    with col2:
        dti_ratio = st.number_input("DTI Ratio (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=None, step=0.01)
        savings = st.number_input("Savings ($)", min_value=0, value=None)
        collateral_value = st.number_input("Collateral Value ($)", min_value=0, value=None)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=None)
        loan_term = st.number_input("Loan Term (Months)", min_value=1, value=None)

    st.subheader("Categorical Information")
    c_col1, c_col2 = st.columns(2)
    
    with c_col1:
        employment_status = st.selectbox("Employment Status", options=["Salaried", "Self-employed", "Unemployed", "Contract"], index=None, placeholder="Select Status")
        marital_status = st.selectbox("Marital Status", options=["Married", "Single"], index=None, placeholder="Select Status")
        loan_purpose = st.selectbox("Loan Purpose", options=["Home", "Business", "Personal", "Education", "Car"], index=None, placeholder="Select Purpose")

    with c_col2:
        property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"], index=None, placeholder="Select Area")
        education_level = st.selectbox("Education Level", options=["Graduate", "Not Graduate"], index=None, placeholder="Select Level")
        gender = st.selectbox("Gender", options=["Male", "Female"], index=None, placeholder="Select Gender")
        employer_category = st.selectbox("Employer Category", options=["Private", "Government", "MNC", "Business", "Unemployed"], index=None, placeholder="Select Category")

    submit = st.form_submit_button("Predict Approval Status")

if submit:
    # Check if any input is None
    form_values = [
        applicant_income, coapplicant_income, age, dependents, credit_score,
        existing_loans, dti_ratio, savings, collateral_value, loan_amount,
        loan_term, employment_status, marital_status, loan_purpose,
        property_area, education_level, gender, employer_category
    ]
    
    if None in form_values:
        st.warning("Please fill in all the fields before predicting.")
    else:
        # Prepare input data
        input_data = {
            'Applicant_Income': applicant_income,
            'Coapplicant_Income': coapplicant_income,
            'Employment_Status': employment_status,
            'Age': age,
            'Marital_Status': marital_status,
            'Dependents': dependents,
            'Credit_Score': credit_score,
            'Existing_Loans': existing_loans,
            'DTI_Ratio': dti_ratio,
            'Savings': savings,
            'Collateral_Value': collateral_value,
            'Loan_Amount': loan_amount,
            'Loan_Term': loan_term,
            'Loan_Purpose': loan_purpose,
            'Property_Area': property_area,
            'Education_Level': education_level,
            'Gender': gender,
            'Employer_Category': employer_category
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for col, le in le_dict.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except ValueError:
                    # Handle unseen values by picking the most frequent/fallback
                    input_df[col] = 0 
                    
        # Feature Engineering (as per notebook/training)
        if 'DTI_Ratio' in input_df.columns:
             input_df['DTI_Ratio_sq'] = input_df['DTI_Ratio'] ** 2
        if 'Credit_Score' in input_df.columns:
             input_df['Credit_Score_sq'] = input_df['Credit_Score'] ** 2

        # Reorder columns to match model training
        input_df = input_df[feature_names]
        
        # Scale numerical features
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Decode target
        result = le_target.inverse_transform([prediction])[0]
        
        # Display result
        st.divider()
        if result == 'Yes':
            st.success(f"### Congrats🎉 LOAN APPROVED ✅")
            st.write(f"Confidence: {probability[1]:.2%}")
        else:
            st.error(f"###  Sorry! LOAN REJECTED ❌")
            st.write(f"Confidence: {probability[0]:.2%}")
