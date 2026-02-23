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

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# ==================================
# SESSION STATE INIT (VERY IMPORTANT)
# ==================================
if "step" not in st.session_state:
    st.session_state.step = "input"

# ==================================
# LOAD MODEL
# ==================================
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")  # make sure this file exists

model = load_model()

# ==================================
# PAGE 1 : INPUT PAGE
# ==================================
if st.session_state.step == "input":

    st.title("🩺 Diabetes Risk Prediction System")
    st.markdown("### Enter Patient Details")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("👤 Name")
        age = st.number_input(" Age", min_value=1, max_value=100, value=25)

    with col2:
        gender = st.radio("⚧ Gender", ["Male", "Female", "Others"])

    st.subheader("🧬 Medical Details")

    col3, col4 = st.columns(2)

    with col3:
        glucose = st.number_input("🧪 Glucose (mg/dL)", 50, 300, 120)
        bmi = st.number_input("⚖ BMI", 10.0, 60.0, 25.0)

    with col4:
        bp = st.number_input("💓 Blood Pressure (mmHg)", 40, 150, 80)
        pregnancies = st.number_input(
            "🤰 Pregnancies", 0, 10, 0
        ) if gender == "Female" else 0

    family = st.radio("👪 Family History of Diabetes", ["No", "Yes"])
    family_val = 1 if family == "Yes" else 0

    st.divider()

    if st.button("🔮 Predict Diabetes Risk", use_container_width=True):

        st.session_state.user_df = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "BMI": bmi,
            "Age": age,
            "FamilyHistory": family_val
        }])

        st.session_state.meta = {
            "Name": name,
            "Gender": gender,
            "Age": age,
            "Glucose": glucose,
            "BMI": bmi,
            "BP": bp,
            "FamilyHistory": family
        }

        st.session_state.step = "result"
        st.rerun()

# ==================================
# PAGE 2 : RESULT PAGE
# ==================================
if st.session_state.step == "result":

    st.title("📊 Prediction Result")

    user_df = st.session_state.user_df
    meta = st.session_state.meta

    # ---- PREDICTION ----
    prob = model.predict_proba(user_df)[0][1] * 100

    # ---- RISK CLASSIFICATION (FIXED LOGIC) ----
    if prob < 35:
        risk_label = "LOW RISK"
        st.success(f"🟢 LOW RISK ({prob:.2f}%)")
    elif prob < 65:
        risk_label = "MODERATE RISK"
        st.warning(f"🟡 MODERATE RISK ({prob:.2f}%)")
    else:
        risk_label = "HIGH RISK"
        st.error(f"🔴 HIGH RISK ({prob:.2f}%)")

    # ---- PIE CHART ----
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(
            [100 - prob, prob],
            labels=["Low Risk", "High Risk"],
            autopct="%1.0f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)
        plt.close()

    # ---- COMPARISON CHART ----
    st.subheader("📊 User vs Normal Comparison")

    compare_df = pd.DataFrame({
        "Normal": [100, 22, 80],
        "User": [meta["Glucose"], meta["BMI"], meta["BP"]]
    }, index=["Glucose", "BMI", "Blood Pressure"])

    st.bar_chart(compare_df)

    # ---- SAVE LOG ----
    log_file = "prediction_log.csv"

    log_row = pd.DataFrame([{
        "Name": meta["Name"],
        "Gender": meta["Gender"],
        "Age": meta["Age"],
        "Glucose": meta["Glucose"],
        "BMI": meta["BMI"],
        "FamilyHistory": meta["FamilyHistory"],
        "RiskPercent": round(prob, 2),
        "RiskLevel": risk_label,
        "Timestamp": datetime.datetime.now()
    }])

    log_row.to_csv(
        log_file,
        mode="a",
        header=not os.path.exists(log_file),
        index=False
    )

    # ---- GENDER ANALYTICS ----
    st.subheader("👥 Gender-wise Analytics")

    hist_df = pd.read_csv(log_file)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Gender Count**")
        st.bar_chart(hist_df["Gender"].value_counts())

    with col2:
        st.markdown("**Average Risk % by Gender**")
        st.bar_chart(hist_df.groupby("Gender")["RiskPercent"].mean())

    # ---- HISTORY ----
    with st.expander("📄 View Prediction History"):
        st.dataframe(hist_df)

    # ---- DOWNLOAD REPORT ----
    report = f"""
Diabetes Risk Assessment Report
-------------------------------
Name: {meta['Name']}
Gender: {meta['Gender']}
Age: {meta['Age']}
Glucose: {meta['Glucose']}
BMI: {meta['BMI']}
Blood Pressure: {meta['BP']}
Risk Probability: {prob:.2f}%
Risk Level: {risk_label}
"""

    st.download_button(
        "📥 Download Medical Report",
        report,
        file_name="diabetes_report.txt"
    )

    if st.button("⬅ Back to Entry Page"):
        st.session_state.step = "input"
        st.rerun()
