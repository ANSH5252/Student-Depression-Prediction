import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load("student_depression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please run the final cell in your Jupyter notebook to save them first.")
    st.stop()

st.set_page_config(page_title="Student Depression Prediction", page_icon="🧠", layout="centered")
st.title("🧠 Student Depression Prediction")
st.write("This app uses the machine learning to predict depression risk based on the inputs below.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal & Academic Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=17, max_value=60, value=21)
    academic_pressure = st.slider("Academic Pressure (Scale 1-5)", 1, 5, 3)
    work_pressure = st.slider("Work Pressure (Scale 0-5)", 0, 5, 0)
    cgpa = st.slider("CGPA (Scale 0-10)", 0.0, 10.0, 7.5)

with col2:
    st.subheader("Satisfaction & Lifestyle")
    study_satisfaction = st.slider("Study Satisfaction (Scale 0-5)", 0, 5, 3)
    job_satisfaction = st.slider("Job Satisfaction (Scale 0-4)", 0, 4, 0)
    sleep_duration = st.selectbox("Average Sleep Duration", ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours', 'Others'])
    dietary_habits = st.selectbox("Dietary Habits", ['Unhealthy', 'Moderate', 'Healthy', 'Others'])

st.subheader("Health & Financials")
col3, col4 = st.columns(2)

with col3:
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
    family_history = st.selectbox("Family History of Mental Illness?", ["No", "Yes"])

with col4:
    work_study_hours = st.slider("Work/Study Hours per Day", 0, 12, 8)
    financial_stress = st.slider("Financial Stress (Scale 1-5)", 1, 5, 2)

if st.button("Predict Depression Risk", type="primary"):
    
    input_data = {
        'Gender': gender, 'Age': age, 'Academic Pressure': academic_pressure,
        'Work Pressure': work_pressure, 'CGPA': cgpa, 'Study Satisfaction': study_satisfaction,
        'Job Satisfaction': job_satisfaction, 'Sleep Duration': sleep_duration,
        'Dietary Habits': dietary_habits,
        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
        'Work/Study Hours': work_study_hours, 'Financial Stress': financial_stress,
        'Family History of Mental Illness': family_history
    }
    input_df = pd.DataFrame([input_data])
    input_df['Gender'] = input_df['Gender'].map({'Male': 0, 'Female': 1})
    input_df['Have you ever had suicidal thoughts ?'] = input_df['Have you ever had suicidal thoughts ?'].map({'No': 0, 'Yes': 1})
    input_df['Family History of Mental Illness'] = input_df['Family History of Mental Illness'].map({'No': 0, 'Yes': 1})
    input_df['Sleep Duration'] = input_df['Sleep Duration'].replace({'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3, 'Others': 4})
    input_df['Dietary Habits'] = input_df['Dietary Habits'].replace({'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2, 'Others': 3})
    input_aligned = input_df.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_aligned)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    st.write("---")
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error(f"**High Risk of Depression Detected** (Model Confidence: {prediction_proba:.0%})")
        st.write("Based on the inputs, the model suggests a higher likelihood of depression. Please consider speaking with a mental health professional.")
    else:
        st.success(f"**Low Risk of Depression Detected** (Model Confidence: {1 - prediction_proba:.0%})")
        st.write("The model indicates a lower likelihood of depression. Continue to maintain a healthy lifestyle and monitor your well-being.")
    
    st.progress(prediction_proba)
    st.info("Disclaimer: This is an AI-powered prediction and is not a substitute for professional medical advice.", icon="ℹ️")