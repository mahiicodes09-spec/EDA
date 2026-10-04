import streamlit as st
import pandas as pd

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "student_gpa_model.pkl")



st.title("Student GPA Predictor")
st.write("Predict GPA based on daily student activities.")

study = st.number_input("Study Hours per Day", 0.0, 24.0, 5.0)
extra = st.number_input("Extracurricular Hours per Day", 0.0, 24.0, 2.0)
sleep = st.number_input("Sleep Hours per Day", 0.0, 24.0, 8.0)
social = st.number_input("Social Hours per Day", 0.0, 24.0, 2.0)
physical = st.number_input("Physical Activity Hours per Day", 0.0, 24.0, 2.0)

if st.button("Predict GPA"):
    input_data = pd.DataFrame([[
        study, extra, sleep, social, physical
    ]], columns=[
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day"
    ])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted GPA: {prediction:.2f}")