# 💳 Fintech Credit Risk Scoring Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI_Deployment-red?style=for-the-badge&logo=streamlit&logoColor=white)

## 📌 Executive Summary
An end-to-end Machine Learning pipeline designed to predict loan default risk. This project transitions raw financial data into a deployable gradient boosting model, wrapped in an interactive web interface for real-time stakeholder decision-making.

## 💼 Business Value
* **Automated Risk Assessment:** Translates complex predictive probabilities into an intuitive 300-850 FICO-style Credit Score.
* **Feature Engineering:** Leverages custom financial ratios (e.g., Debt-to-Income) alongside external credit bureau data to identify high-risk applicants.
* **Interactive Deployment:** Provides a real-time UI for stakeholders to input applicant data and receive instant approval or decline decisions.

## 🛠️ System Architecture & Tech Stack
* **Data Processing & Modeling:** `Pandas`, `NumPy`, `Scikit-Learn`, `LightGBM`
* **Deployment & Serialization:** `Streamlit`, `Joblib`
* **Version Control:** `Git`, `GitHub`

## 🚀 How to Run Locally

1. Clone the repository:
```bash
git clone [https://github.com/Yinka-Agbaje/fintech-credit-scoring.git](https://github.com/Yinka-Agbaje/fintech-credit-scoring.git)
```

2. Install the required dependencies:
```bash
pip install -r requirement.txt
```

3. Launch the Streamlit application:
```bash
streamlit run app.py
```

## 📂 Repository Structure
* `Credit_Score_Model.ipynb`: Research environment (data cleaning, feature engineering, model training).
* `app.py`: Production web application script.
* `lightgbm_credit_model.joblib`: Serialized machine learning model.
* `base_profile.joblib`: Baseline applicant data template for the UI.