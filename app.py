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

    # -------- Medical Inputs --------
    st.subheader("🧬 Medical Details")

    col3, col4 = st.columns(2)

    with col3:
        glucose = st.number_input(
            "🧪 Glucose (mg/dL)", min_value=50, max_value=200, value=120
        )
        bmi = st.number_input(
            "⚖ BMI", min_value=10.0, max_value=60.0, value=25.0
        )

    with col4:
        bp = st.number_input(
            "💓 Blood Pressure (mmHg)", min_value=40, max_value=150, value=80
        )
        pregnancies = (
            st.number_input("🤰 Pregnancies", min_value=0, max_value=10, value=0)
            if gender == "Female"
            else 0
        )

    family = st.radio("👪 Family History of Diabetes", ["No", "Yes"])
    family_val = 1 if family == "Yes" else 0

    # -------- Gender-specific (UI only) --------
    if gender == "Female":
        st.subheader("👩 Female-Specific Health Details")
        st.radio("Gestational Diabetes", ["No", "Yes"])
        st.radio("PCOS", ["No", "Yes"])
        st.selectbox("Menopause Status", ["Pre-menopause", "Post-menopause"])

    elif gender == "Male":
        st.subheader("👨 Male-Specific Health Details")
        st.number_input("Waist Circumference (cm)", 50, 150, 85)
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

    # ---- Prediction Probability ----
    prob = model.predict_proba(user_df)[0][1] * 100

    # ---- Risk Classification ----
    if prob < 30:
        risk_label = "LOW RISK"
        st.success(f"🟢 LOW RISK ({prob:.2f}%)")
    elif prob < 60:
        risk_label = "MODERATE RISK"
        st.warning(f"🟡 MODERATE RISK ({prob:.2f}%)")
    else:
        risk_label = "HIGH RISK"
        st.error(f"🔴 HIGH RISK ({prob:.2f}%)")

    # -------- PIE CHART --------
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

    if os.path.exists(log_file):
        hist_df = pd.read_csv(log_file)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Gender Count**")
            st.bar_chart(hist_df["Gender"].value_counts())

        with col2:
            st.markdown("**Average Risk % by Gender**")
            st.bar_chart(hist_df.groupby("Gender")["RiskPercent"].mean())
    else:
        st.info("📌 Gender analytics will appear after multiple predictions.")

    # ---- HISTORY ----
    with st.expander("📄 View Prediction History"):
        if os.path.exists(log_file):
            st.dataframe(pd.read_csv(log_file))

    # ---- DOWNLOAD ----
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
