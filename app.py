import streamlit as st
import pandas as pd
import joblib

# Load model and pipeline
model = joblib.load("modle.pkl")
pipeline = joblib.load("pipeline.pkl")

st.title("Heart Disease Prediction")

# Create input fields for all features in the same order as the training data
# You may need to adjust these based on your actual dataset
age = st.number_input("Age", min_value=1)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
bp = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
thalach = st.number_input("Max Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope of ST", ["Up", "Flat", "Down"])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Convert inputs into DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "ChestPainType": chest_pain,
    "RestingBP": bp,
    "Cholesterol": chol,
    "FastingBS": 1 if fbs == "Yes" else 0,
    "RestingECG": rest_ecg,
    "MaxHR": thalach,
    "ExerciseAngina": "Y" if exang == "Yes" else "N",
    "Oldpeak": oldpeak,
    "ST_Slope": slope,
    "Ca": ca,
    "Thal": thal
}])

# Predict
if st.button("Predict"):
    transformed = pipeline.transform(input_data)
    prediction = model.predict(transformed)

    if prediction[0] == 1:
        st.error("💔 Positive for Heart Disease")
    else:
        st.success("💖 Negative")
