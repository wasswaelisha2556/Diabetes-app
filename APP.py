import streamlit as st
import joblib
import numpy as np

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('diabetes_scaler.pkl')

st.title("Diabetes Risk Predictor")
st.write("Enter your health details below to check your diabetes risk.")

pregnancies = st.number_input("Pregnancies",min_value=0, max_value=20, value= 0)
glucose = st.number_input("Glucose",min_value=0, max_value=300, value= 100)
blood_pressure = st.number_input("Blood Pressure",min_value=0, max_value=200, value= 70)
skin_thickness = st.number_input("Skin Thickness",min_value=0, max_value=100, value= 20)
insulin = st.number_input("Insulin",min_value=0, max_value=900, value= 80)
bmi = st.number_input("BMI",min_value=0.0, max_value=70.0, value= 25.0)
dpf = st.number_input("Diabetes Pedigree Function",min_value=0.0, max_value=3.0, value= 0.5)
age = st.number_input("Age",min_value=1, max_value=120, value= 30)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("The model predicts: Diabetic")
    else:
        st.success("The model predicts: Not Diabetic")