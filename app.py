import streamlit as st
import numpy as np
import joblib

# Load model, threshold, and scaler
model = joblib.load("best_svc_model.pkl")
threshold = joblib.load("best_threshold.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Student Depression Prediction")

st.write("Enter student details below to predict depression risk:")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=40, value=20)
sleep = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"])
diet = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy", "Others"])
suicidal = st.selectbox("Ever had suicidal thoughts?", ["No", "Yes"])
family_history = st.selectbox("Family History of Mental Illness?", ["No", "Yes"])
academic_pressure = st.slider("Academic Pressure", 0, 10, 5)
work_pressure = st.slider("Work Pressure", 0, 10, 5)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5)
study_satisfaction = st.slider("Study Satisfaction", 0, 10, 5)
job_satisfaction = st.slider("Job Satisfaction", 0, 10, 5)
hours = st.number_input("Work/Study Hours", min_value=0, max_value=24, value=6)
financial_stress = st.slider("Financial Stress", 0, 10, 5)

# Encoding categorical features (consistent with training)
gender_val = 1 if gender == "Female" else 0
suicidal_val = 1 if suicidal == "Yes" else 0
history_val = 1 if family_history == "Yes" else 0

sleep_map = {
    "Less than 5 hours": 0,
    "5-6 hours": 1,
    "7-8 hours": 2,
    "More than 8 hours": 3,
    "Others": 4
}
diet_map = {
    "Unhealthy": 0,
    "Moderate": 1,
    "Healthy": 2,
    "Others": 3
}

sleep_val = sleep_map[sleep]
diet_val = diet_map[diet]

# Feature vector (must match training X order)
features = np.array([[ 
    gender_val, age, academic_pressure, work_pressure, cgpa, study_satisfaction,
    job_satisfaction, hours, financial_stress, suicidal_val, history_val,
    sleep_val, diet_val
]])

# Prediction button
if st.button("Predict"):
    features_std = scaler.transform(features)
    proba = model.predict_proba(features_std)[:, 1]
    prediction = (proba >= threshold).astype(int)[0]

    if prediction == 1:
        st.error("⚠️ High risk of Depression")
    else:
        st.success("✅ No Depression Detected")
