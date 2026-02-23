# ==================================
# IMPORTS
# ==================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# ==================================
# SESSION STATE INIT
# ==================================
if "step" not in st.session_state:
    st.session_state.step = "input"

# ==================================
# LOAD DATASET
# ==================================
@st.cache_data
def load_data():
    return pd.read_csv("pima_diabetes.csv")

df = load_data()

# ==================================
# TRAIN / LOAD MODEL
# ==================================
@st.cache_resource
def get_model():
    if os.path.exists("diabetes_model.pkl"):
        return joblib.load("diabetes_model.pkl")

    # ---- TRAIN MODEL IF NOT EXISTS ----
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "diabetes_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return model

model = get_model()
scaler = joblib.load("scaler.pkl")

# ==================================
# PAGE 1 : INPUT
# ==================================
if st.session_state.step == "input":

    st.title("🩺 Diabetes Risk Prediction System")
    st.markdown("### Enter Patient Details")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("👤 Name")
        age = st.number_input("Age", 1, 100, 25)

    with col2:
        gender = st.radio("Gender", ["Male", "Female", "Others"])

    st.subheader("🧬 Medical Details")

    col3, col4 = st.columns(2)

    with col3:
        glucose = st.number_input("Glucose (mg/dL)", 50, 300, 120)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

    with col4:
        bp = st.number_input("Blood Pressure", 40, 150, 80)
        pregnancies = st.number_input(
            "Pregnancies", 0, 10, 0
        ) if gender == "Female" else 0

    family = st.radio("Family History of Diabetes", ["No", "Yes"])
    family_val = 1 if family == "Yes" else 0

    if st.button("🔮 Predict Risk", use_container_width=True):

        user_df = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": 20,
            "Insulin": 80,
            "BMI": bmi,
            "DiabetesPedigreeFunction": 0.5,
            "Age": age
        }])

        user_scaled = scaler.transform(user_df)

        st.session_state.user_scaled = user_scaled
        st.session_state.meta = {
            "Name": name,
            "Gender": gender,
            "Age": age,
            "Glucose": glucose,
            "BMI": bmi,
            "BP": bp
        }

        st.session_state.step = "result"
        st.rerun()

# ==================================
# PAGE 2 : RESULT
# ==================================
if st.session_state.step == "result":

    st.title("📊 Prediction Result")

    prob = model.predict_proba(st.session_state.user_scaled)[0][1] * 100

    if prob < 35:
        risk = "LOW RISK"
        st.success(f"🟢 LOW RISK ({prob:.2f}%)")
    elif prob < 65:
        risk = "MODERATE RISK"
        st.warning(f"🟡 MODERATE RISK ({prob:.2f}%)")
    else:
        risk = "HIGH RISK"
        st.error(f"🔴 HIGH RISK ({prob:.2f}%)")

    # ---- PIE CHART ----
    fig, ax = plt.subplots()
    ax.pie([100 - prob, prob], labels=["Low", "High"], autopct="%1.0f%%")
    st.pyplot(fig)

    # ---- LOG SAVE ----
    log_file = "prediction_log.csv"
    log_row = pd.DataFrame([{
        "Name": st.session_state.meta["Name"],
        "Gender": st.session_state.meta["Gender"],
        "Age": st.session_state.meta["Age"],
        "RiskPercent": round(prob, 2),
        "RiskLevel": risk,
        "Timestamp": datetime.datetime.now()
    }])

    log_row.to_csv(
        log_file,
        mode="a",
        header=not os.path.exists(log_file),
        index=False
    )

    if st.button("⬅ Back"):
        st.session_state.step = "input"
        st.rerun()
