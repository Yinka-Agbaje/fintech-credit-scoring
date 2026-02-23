import streamlit as st
import joblib
import numpy as np

# Load the predictive model and baseline data profile
model = joblib.load('lightgbm_credit_model.joblib')
base_profile = joblib.load('base_profile.joblib')

# Convert risk probability into a standard 300-850 credit score
def prob_to_credit_score(probability):
    min_score = 300
    max_score = 850
    score = min_score + ((1 - probability) * (max_score - min_score))
    return int(np.round(score))

# Configure the browser tab and page layout
st.set_page_config(page_title="Fintech Risk Engine", layout="wide")

st.title("💳 Fintech Credit Risk Engine")
st.markdown("Predict loan default risk using Gradient Boosting.")

# Build the input sidebar
st.sidebar.header("Applicant Financial Data")
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, value=50000)
loan_amount = st.sidebar.number_input("Requested Loan Amount ($)", min_value=1000, value=10000)
years_employed = st.sidebar.number_input("Years Employed", min_value=0.0, value=5.0)

# Execute the assessment only when the button is clicked
if st.sidebar.button("Run Risk Assessment"):
    applicant = base_profile.copy()
    
    applicant['AMT_INCOME_TOTAL'] = income
    applicant['AMT_CREDIT'] = loan_amount
    applicant['YEARS_EMPLOYED'] = years_employed
    
    applicant['CREDIT_INCOME_RATIO'] = loan_amount / income
    applicant['ANNUITY_INCOME_RATIO'] = applicant['AMT_ANNUITY'] / income
    
    prob = model.predict_proba(applicant)[0][1]
    score = prob_to_credit_score(prob)
    
    # Display the metrics cleanly in the center of the screen
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Default Probability", value=f"{prob:.1%}")
    with col2:
        st.metric(label="Assigned Credit Score", value=score)
    
    st.divider()
    
    # Output the final banking decision
    if score > 650:
        st.success("System Decision: **APPROVED** - Low Risk Profile")
    else:
        st.error("System Decision: **DECLINED** - High Risk Profile")