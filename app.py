import streamlit as st
import pandas as pd
import joblib

# Load trained model, scaler, threshold, and feature columns
model = joblib.load("best_svc_model.pkl")
scaler = joblib.load("scaler.pkl")
threshold = joblib.load("best_threshold.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # saved from training

st.title("🧠 Student Depression Prediction")

# --- Inputs ---
age = st.number_input("Age", min_value=10, max_value=100, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
study_hours = st.number_input("Study Hours", min_value=0, max_value=24, step=1)
social_media = st.number_input("Social Media Hours", min_value=0, max_value=24, step=1)
sleep = st.selectbox("Sleep Duration", ["<5", "5-6", "7-8", ">8"])
diet = st.selectbox("Dietary Habits", ["Poor", "Average", "Good"])
suicidal = st.selectbox("Suicidal Thoughts", ["No", "Yes"])
family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

# New categorical inputs
city = st.selectbox("City", ["CityA", "CityB", "CityC"])  # use your actual training cities
profession = st.selectbox("Profession", ["Student", "Part-time", "Full-time"])  # update list
degree = st.selectbox("Degree", ["B.Tech", "M.Tech", "MBA", "PhD"])  # update list

# --- Encode inputs into dataframe ---
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [1 if gender == "Male" else 0],
    "CGPA": [cgpa],
    "Study_Hours": [study_hours],
    "Social_Media_Hours": [social_media],
    "Sleep_Duration": [sleep],
    "Dietary_Habits": [diet],
    "Suicidal_Thoughts": [1 if suicidal == "Yes" else 0],
    "Family_History": [1 if family_history == "Yes" else 0],
    "City": [city],
    "Profession": [profession],
    "Degree": [degree]
})

# One-hot encode categorical like training
input_encoded = pd.get_dummies(input_data, columns=["City", "Profession", "Degree"], drop_first=True)

# Ensure same columns as training
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Scale numeric features
input_scaled = scaler.transform(input_encoded)

# Prediction
if st.button("Predict"):
    pred_proba = model.decision_function(input_scaled)
    prediction = (pred_proba > threshold).astype(int)
    
    if prediction[0] == 1:
        st.error("⚠️ High risk of Depression detected")
    else:
        st.success("✅ No significant signs of Depression detected")
