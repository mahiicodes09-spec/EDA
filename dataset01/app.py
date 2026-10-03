import streamlit as st
import pandas as pd
import joblib

# Loading the saved model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

bundle = joblib.load(BASE_DIR / "heart_model_bundle.pkl")

# Extracting the model from the bundle
model = bundle["model"]
scaler = bundle["scaler"]
feature_columns = bundle["feature_columns"]

# Making the user interface
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("Heart Disease Prediction")
st.write("Enter the details below to get a model prediction.")

st.subheader("Patient Details:")

# Numerical details
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=40
    )

with col2:
    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0,
        value=200
    )

with col3:
    max_hr = st.number_input(
        "Max Heart Rate",
        min_value=0,
        value=150
    )

col4, col5 = st.columns(2)

with col4:
    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        value=1.0,
        step=0.1
    )

with col5:
    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=0,
        value=120
    )

# Categorical details
st.subheader("Categorical Details")

col1, col2, col3 = st.columns(3)

with col1:
    sex = st.selectbox("Sex", ["M", "F"])

with col2:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["TA", "ATA", "NAP", "ASY"]
    )

with col3:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

col4, col5, col6 = st.columns(3)

with col4:
    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["Y", "N"]
    )

with col5:
    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

with col6:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar",
        [0, 1]
    )

predict_button = st.button(
    "Predict",
    type="primary",
    use_container_width=True
)

# Prediction and display
if predict_button:
    # Creating the input DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "MaxHR": [max_hr],
        "Oldpeak": [oldpeak],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingECG": [resting_ecg],
        "ExerciseAngina": [exercise_angina],
        "ST_Slope": [st_slope]
    })

    # Defining all possible categories
    categories = {
        "Sex": ["F", "M"],
        "ChestPainType": ["ASY", "ATA", "NAP", "TA"],
        "RestingECG": ["LVH", "Normal", "ST"],
        "ExerciseAngina": ["N", "Y"],
        "ST_Slope": ["Down", "Flat", "Up"]
    }

    # Converting categorical columns
    for col, values in categories.items():
        input_data[col] = pd.Categorical(
            input_data[col],
            categories=values
        )

    # Encoding categorical columns
    input_encoded = pd.get_dummies(
        input_data,
        columns=list(categories.keys()),
        drop_first=True,
        dtype=int
    )

    # Matching training columns
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scaling numerical columns
    num_cols = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak"
    ]

    input_encoded[num_cols] = scaler.transform(
        input_encoded[num_cols]
    )

    prediction = model.predict(input_encoded)
    probability = model.predict_proba(input_encoded)

    no_disease = probability[0][0] * 100
    disease = probability[0][1] * 100

    # Displaying results
    st.divider()
    st.subheader("Prediction Results")

    if prediction[0] == 1:
        st.warning("Model prediction: Heart disease")
    else:
        st.success("Model prediction: No heart disease")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "No heart disease",
            f"{no_disease:.2f}%"
        )

    with col2:
        st.metric(
            "Heart disease",
            f"{disease:.2f}%"
        )

    st.caption(
        "These are model-estimated probabilities, not medically "
        "validated diagnostic probabilities. This educational "
        "model should not be used for medical decisions."
    )
