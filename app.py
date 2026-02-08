import streamlit as st
import pickle
import pandas as pd

# Load model and scaler
with open("risk_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Healthcare Risk Prediction – 30-Day Readmission")

st.markdown("""
Enter patient information below to predict 30-day hospital readmission risk.
""")

# Input form
age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
diastolic_bp = st.number_input("Diastolic BP", 50, 120, 80)
heart_rate = st.number_input("Heart Rate", 40, 200, 75)
creatinine = st.number_input("Creatinine", 0.1, 15.0, 1.0)
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
copd = st.selectbox("COPD", ["No", "Yes"])
length_of_stay = st.number_input("Length of Stay (days)", 1, 30, 4)

# Map categorical inputs
sex = 1 if sex == "Male" else 0
diabetes = 1 if diabetes == "Yes" else 0
copd = 1 if copd == "Yes" else 0

input_dict = {
    "age": age,
    "sex": sex,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "heart_rate": heart_rate,
    "creatinine": creatinine,
    "diabetes": diabetes,
    "copd": copd,
    "length_of_stay": length_of_stay
}

if st.button("Predict Risk"):
    df = pd.DataFrame([input_dict])
    X_scaled = scaler.transform(df)
    risk_prob = model.predict_proba(X_scaled)[:,1][0]
    if risk_prob < 0.2:
        risk_group = "LOW"
    elif risk_prob < 0.5:
        risk_group = "MEDIUM"
    else:
        risk_group = "HIGH"
    
    st.subheader("Predicted Risk")
    st.write(f"**Risk Score:** {risk_prob:.2f}")
    st.write(f"**Risk Group:** {risk_group}")
