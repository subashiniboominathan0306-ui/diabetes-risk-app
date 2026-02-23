import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from sklearn.linear_model import LogisticRegression

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    layout="wide"
)

# ----------------------------------
# SESSION STATE INIT
# ----------------------------------
if "step" not in st.session_state:
    st.session_state.step = "input"

# ----------------------------------
# LOAD & TRAIN MODEL
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("pima_diabetes.csv")

df = load_data()
df = df.drop(columns=["Insulin", "SkinThickness", "DiabetesPedigreeFunction"])
df["FamilyHistory"] = np.random.randint(0, 2, len(df))

X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age", "FamilyHistory"]]
y = df["Outcome"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

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
        age = st.number_input("Age", 1, 100, 1)

    with col2:
        gender = st.radio("⚧ Gender", ["Male", "Female", "Others"])

    st.subheader("🧬 Medical Details")

    col3, col4 = st.columns(2)

    with col3:
        glucose = st.number_input("🧪 Glucose (mg/dL)", 50, 200, 50)
        bmi = st.number_input("⚖ BMI", 10.0, 60.0, 10.0)

    with col4:
        bp = st.number_input("💓 Blood Pressure", 40, 150, 40)
        pregnancies = st.number_input("🤰 Pregnancies", 0, 10, 0) if gender == "Female" else 0

    family = st.radio("👪 Family History of Diabetes", ["No", "Yes"])
    family_val = 1 if family == "Yes" else 0

    if gender == "Female":
        st.subheader("👩 Female-Specific Health Details")
        st.radio("Gestational Diabetes", ["No", "Yes"])
        st.radio("PCOS", ["No", "Yes"])
        st.selectbox("Menopause Status", ["Pre-menopause", "Post-menopause"])

    elif gender == "Male":
        st.subheader("👨 Male-Specific Health Details")
        st.number_input("Waist Circumference (cm)", 50, 150, 50)
        st.radio("Central Obesity", ["No", "Yes"])
        st.selectbox("Stress Level", ["Low", "Moderate", "High"])

    else:
        st.subheader("⚧ Inclusive Health Details")
        hormone = st.radio("On Hormone Therapy", ["No", "Yes"])
        if hormone == "Yes":
            st.number_input("Therapy Duration (years)", 0, 20, 1)

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

    prob = model.predict_proba(user_df)[0][1] * 100
    pred = model.predict(user_df)[0]

    if pred == 0:
        st.success(f"🟢 LOW RISK ({prob:.2f}%)")
    else:
        st.error(f"🔴 HIGH RISK ({prob:.2f}%)")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig, ax = plt.subplots(figsize=(2.4, 2.4))
        ax.pie(
            [100 - prob, prob],
            labels=["Low Risk", "High Risk"],
            autopct="%1.0f%%",
            startangle=90,
            colors=["green", "red"],
            textprops={'fontsize': 8}
        )
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("📊 User vs Normal Comparison")

    compare_df = pd.DataFrame({
        "Normal": [100, 22, 80],
        "User": [meta["Glucose"], meta["BMI"], meta["BP"]]
    }, index=["Glucose", "BMI", "Blood Pressure"])

    st.bar_chart(compare_df)

    log_file = "prediction_log.csv"

    log_row = pd.DataFrame([{
        "Name": meta["Name"],
        "Gender": meta["Gender"],
        "Age": meta["Age"],
        "Glucose": meta["Glucose"],
        "BMI": meta["BMI"],
        "FamilyHistory": meta["FamilyHistory"],
        "RiskPercent": round(prob, 2),
        "Timestamp": datetime.datetime.now()
    }])

    log_row.to_csv(
        log_file,
        mode="a",
        header=not os.path.exists(log_file),
        index=False
    )

    st.subheader("👥 Gender-wise Analytics")

    try:
        hist_df = pd.read_csv(log_file, engine="python", on_bad_lines="skip")
    except FileNotFoundError:
        st.warning("No prediction history available yet.")
        hist_df = pd.DataFrame()

    if not hist_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Gender Count**")
            st.bar_chart(hist_df["Gender"].value_counts())

        with col2:
            st.markdown("**Average Risk % by Gender**")
            st.bar_chart(hist_df.groupby("Gender")["RiskPercent"].mean())

        with st.expander("📄 View Prediction History"):
            st.dataframe(hist_df)

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
Status: {"High Risk" if pred else "Low Risk"}
"""

    st.download_button(
        "📥 Download Medical Report",
        report,
        file_name="diabetes_report.txt"
    )

    if st.button("⬅ Back to Entry Page"):
        st.session_state.step = "input"
        st.rerun()
