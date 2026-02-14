import pandas as pd
import time
import requests
import streamlit as st
import requests, json, numpy as np
from streamlit_option_menu import option_menu


API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="HealthAI Prediction Suite", layout="wide")

# Sidebar Menu
st.sidebar.title("HealthAI Tasks")
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        [
            "Home",
            "Risk Stratification",
            "Length of Stay Prediction",
            "Patient Segmentation",
            "Imaging Diagnosis",
            "Sequence Modeling",
            "Sentiment Analysis",
        ],
        icons=[
            "house", "activity", "clock", "people",
            "diagram-3", "chat-text", "image", "gear"
        ],
        menu_icon="cast",
        default_index=0
    )

menu = selected

# HOME PAGE
if menu == "Home":

    st.title("🏥 HealthAI Prediction Suite")
    st.markdown("### AI-Powered Clinical Decision Support Platform")

    st.divider()

    # Intro Section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Welcome to **HealthAI Prediction Suite**, an integrated healthcare AI platform designed to assist in:

        ✔ Patient risk prediction  
        ✔ Hospital resource planning  
        ✔ Disease diagnosis  
        ✔ Patient monitoring  
        ✔ Clinical analytics  

        Use the **sidebar** to navigate between modules.
        """)

    with col2:
        with st.container(border=True):
            st.subheader("System Status")

            try:
                r = requests.get(API_BASE_URL, timeout=3)
                st.success("API Server: Running")
            except:
                st.error("API Server: Offline")

            st.info("Model Services Ready")

    st.divider()

    # Features Section
    st.subheader("Available AI Modules")

    f1, f2, f3 = st.columns(3)

    with f1:
        with st.container(border=True):
            st.markdown("### ❤️ Risk Stratification")
            st.caption("Predict patient health risk level from clinical data.")

        with st.container(border=True):
            st.markdown("### ⏱ Length of Stay")
            st.caption("Estimate hospital stay duration.")

    with f2:
        with st.container(border=True):
            st.markdown("### 👥 Patient Segmentation")
            st.caption("Cluster patients based on clinical characteristics.")

        with st.container(border=True):
            st.markdown("### 🧠 Imaging Diagnosis")
            st.caption("AI-based medical image analysis.")

    with f3:
        with st.container(border=True):
            st.markdown("### 📈 Sequence Monitoring")
            st.caption("Real-time patient time-series risk prediction.")

        with st.container(border=True):
            st.markdown("### 💬 Sentiment Analysis")
            st.caption("Analyze patient feedback sentiment.")

    st.divider()

    # How to use section
    st.subheader("How to Use")
    st.markdown("""
    1. Select a module from the **left sidebar**
    2. Enter patient data or upload files
    3. Click prediction button
    4. View AI results instantly
    """)

    st.info("Designed for research, clinical decision support, and healthcare analytics.")


# 1. Risk Stratification
if menu == "Risk Stratification":
    st.title("Risk Stratification")

    # Demographic & Lifestyle
    with st.container(border=True):
        st.subheader("Demographic & Lifestyle Information")

        d1, d2 = st.columns(2)
        with d1:
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
            smoking_status = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with d2:
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            alcohol_use = st.selectbox("Alcohol Use", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    # Laboratory Information (3x3)
    with st.container(border=True):
        st.subheader("Laboratory Information")

        l1, l2, l3 = st.columns(3)

        with l1:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, step=0.1)
            platelet_count = st.number_input("Platelet Count", min_value=0.0, step=1000.0)

        with l2:
            total_leukocyte_count = st.number_input("Total Leukocyte Count", min_value=0.0, step=100.0)
            glucose_level = st.number_input("Glucose Level (mg/dL)", min_value=0.0, step=1.0)

        with l3:
            urea_level = st.number_input("Urea Level (mg/dL)", min_value=0.0, step=1.0)
            creatinine_level = st.number_input("Creatinine Level (mg/dL)", min_value=0.0, step=0.1)

        st.markdown("")

    predict_btn = st.button("Predict Risk")

    if predict_btn:
            try:
                payload = {
                    "features": [
                        age, gender, smoking_status, alcohol_use,
                        hemoglobin, total_leukocyte_count, platelet_count,
                        glucose_level, urea_level, creatinine_level
                    ]
                }

                response = requests.post(
                    f"{API_BASE_URL}/risk",
                    json=payload,
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    risk_map = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
                    st.success(
                        f"Predicted Risk Category: {risk_map.get(result['risk_class'], result['risk_class'])}"
                    )
                else:
                    st.error(f"API Error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("FastAPI server is not running.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# 2. Length of Stay Prediction
if menu == "Length of Stay Prediction":
    st.title("Length of Stay Prediction")

    # =========================
    # Demographic Information
    # =========================
    with st.expander("Demographic Information", expanded=True):
        d1, d2, d3 = st.columns(3)

        with d1:
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
        with d2:
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        with d3:
            residence_type = st.selectbox("Residence Type", [0, 1], format_func=lambda x: "Rural" if x == 0 else "Urban")

        admission_type = st.selectbox(
            "Admission Type",
            [0, 1],
            format_func=lambda x: "Elective" if x == 0 else "Emergency"
        )

    # =========================
    # Disease / Clinical Information
    # =========================
    with st.expander("Disease / Clinical Information", expanded=True):
        selected_diseases = st.multiselect(
            "Select Diagnosed Conditions",
            [
                "Stable Angina",
                "Complete Heart Block",
                "Heart Failure",
                "Coronary Artery Disease",
                "Hypertension",
                "Ventricular Tachycardia",
                "Atypical Chest Pain",
                "Acute Coronary Syndrome",
                "Diabetes Mellitus",
                "Urinary Tract Infection",
                "Prior Cardiomyopathy",
                "Raised Cardiac Enzymes",
            ]
        )

        disease_map = {
            "Stable Angina": "stable_angina",
            "Complete Heart Block": "complete_heart_block",
            "Heart Failure": "heart_failure",
            "Coronary Artery Disease": "coronary_artery_disease",
            "Hypertension": "hypertension",
            "Ventricular Tachycardia": "ventricular_tachycardia",
            "Atypical Chest Pain": "atypical_chest_pain",
            "Acute Coronary Syndrome": "acute_coronary_syndrome",
            "Diabetes Mellitus": "diabetes_mellitus",
            "Urinary Tract Infection": "urinary_tract_infection",
            "Prior Cardiomyopathy": "prior_cardiomyopathy",
            "Raised Cardiac Enzymes": "raised_cardiac_enzymes",
        }

        disease_features = {v: 0 for v in disease_map.values()}

        for disease in selected_diseases:
            disease_features[disease_map[disease]] = 1

        stable_angina = disease_features["stable_angina"]
        complete_heart_block = disease_features["complete_heart_block"]
        heart_failure = disease_features["heart_failure"]
        coronary_artery_disease = disease_features["coronary_artery_disease"]
        hypertension = disease_features["hypertension"]
        ventricular_tachycardia = disease_features["ventricular_tachycardia"]
        atypical_chest_pain = disease_features["atypical_chest_pain"]
        acute_coronary_syndrome = disease_features["acute_coronary_syndrome"]
        diabetes_mellitus = disease_features["diabetes_mellitus"]
        urinary_tract_infection = disease_features["urinary_tract_infection"]
        prior_cardiomyopathy = disease_features["prior_cardiomyopathy"]
        raised_cardiac_enzymes = disease_features["raised_cardiac_enzymes"]

    # =========================
    # Laboratory Information
    # =========================
    with st.expander("Laboratory Information", expanded=True):
        l1, l2, l3 = st.columns(3)

        with l1:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, step=0.1)
            total_leukocyte_count = st.number_input("Total Leukocyte Count", min_value=0.0, step=100.0)

        with l2:
            platelet_count = st.number_input("Platelet Count", min_value=0.0, step=1000.0)
            glucose_level = st.number_input("Glucose Level (mg/dL)", min_value=0.0, step=1.0)

        with l3:
            urea_level = st.number_input("Urea Level (mg/dL)", min_value=0.0, step=1.0)
            creatinine_level = st.number_input("Creatinine Level (mg/dL)", min_value=0.0, step=0.1)

    predict_btn = st.button("Predict Length of Stay")

    if predict_btn:
        try:
            payload = {
                "features": [
                    urea_level,
                    total_leukocyte_count,
                    platelet_count,
                    hemoglobin,
                    glucose_level,
                    age,
                    creatinine_level,
                    admission_type,
                    stable_angina,
                    complete_heart_block,
                    heart_failure,
                    coronary_artery_disease,
                    hypertension,
                    ventricular_tachycardia,
                    urinary_tract_infection,
                    diabetes_mellitus,
                    prior_cardiomyopathy,
                    raised_cardiac_enzymes,
                    acute_coronary_syndrome,
                    gender,
                    residence_type,
                    atypical_chest_pain,
                    ]
                    }
            


            response = requests.post(
                f"{API_BASE_URL}/los",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                st.success(
                    f"Predicted Length of Stay (days): {result['length_of_stay']}"
                )
            else:
                st.error(f"API Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("FastAPI server is not running.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# 3. Patient Segmentation
if menu == "Patient Segmentation":
    st.title("Patient Segmentation")

    left, right = st.columns([1, 3])

    with left:
        with st.container(border=True):
            st.subheader("Primary Feature")
            age = st.number_input("Age", min_value=0, max_value=120, step=1)
            st.text_input("", value="", disabled=True)

    with right:
        with st.container(border=True):
            st.subheader("Clinical Parameters")

            c1, c2, c3 = st.columns(3)
            with c1:
                hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, step=0.1)
            with c2:
                total_leukocyte_count = st.number_input("Total Leukocyte Count", min_value=0.0, step=100.0)
            with c3:
                platelet_count = st.number_input("Platelet Count", min_value=0.0, step=1000.0)

            c4, c5, c6 = st.columns(3)
            with c4:
                glucose_level = st.number_input("Glucose Level (mg/dL)", min_value=0.0, step=1.0)
            with c5:
                creatinine_level = st.number_input("Creatinine Level (mg/dL)", min_value=0.0, step=0.1)
            with c6:
                urea_level = st.number_input("Urea Level (mg/dL)", min_value=0.0, step=1.0)

    predict_btn = st.button("Predict Patient Segment")

    if predict_btn:
        try:
            payload = {
                "features": [
                    age,
                    hemoglobin,
                    total_leukocyte_count,
                    platelet_count,
                    glucose_level,
                    creatinine_level,
                    urea_level,
                ]
            }

            response = requests.post(
                f"{API_BASE_URL}/segment",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()

                cluster_map = {
                    0: ("Stable Metabolic Group", "success"),
                    1: ("Inflammatory–Metabolic Risk Group", "warning"),
                    2: ("Renal Dysfunction – High Risk Group", "error")
                }

                cluster_id = result["cluster"]
                cluster_name, ui_type = cluster_map.get(
                    cluster_id, ("Unknown Cluster", "info")
                )

                if ui_type == "success":
                    st.success(f"Predicted Patient Cluster: {cluster_name}")
                elif ui_type == "warning":
                    st.warning(f"Predicted Patient Cluster: {cluster_name}")
                elif ui_type == "error":
                    st.error(f"Predicted Patient Cluster: {cluster_name}")
                else:
                    st.info(f"Predicted Patient Cluster: {cluster_name}")

            else:
                st.error(f"API Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("FastAPI server is not running.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# 5. Imaging Diagnosis
if menu == "Imaging Diagnosis":
    st.title("Imaging Diagnosis")

    st.markdown(
        "Upload a chest image for automated diagnostic prediction. "
        "Images are resized to **224×224** and normalized as during training."
    )

    # Upload Box
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Upload Chest Image (JPG / PNG)",
            type=["jpg", "jpeg", "png"]
        )

    # Preview Box (Smaller)
    if uploaded_file is not None:
        with st.container(border=True):
            st.subheader("Image Preview")
            st.image(
                uploaded_file,
                caption="Uploaded Image (Preview)",
                width=250   # 👈 preview size control (Streamlit only)
            )

    predict_btn = st.button("Run Imaging Diagnosis")

    # =========================
    # Prediction
    # =========================
    if predict_btn and uploaded_file is not None:
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(
                f"{API_BASE_URL}/image",
                files=files,
                timeout=20
            )

            if response.status_code == 200:
                result = response.json()
                st.success("Imaging Diagnosis Completed")
                st.json(result)
            else:
                st.error(f"API Error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("FastAPI server is not running.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    elif predict_btn and uploaded_file is None:
        st.warning("Please upload an image before running diagnosis.")

# 6. Sequence Modeling
if menu == "Sequence Modeling":
    st.title("Live Patient Monitoring (Time Series Prediction)")

    uploaded_file = st.file_uploader(
        "Upload Time-Series Excel File",
        type=["xlsx"]
    )

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head(2))

        interval = 4

        start_btn = st.button("Start Monitoring")

        if start_btn:
            history = []
            monitor_box = st.empty()

            with st.spinner("Monitoring patient in real time..."):
                for i in range(len(df)):
                    history.append(df.iloc[i].values.tolist())

                    payload = {"sequence": history}

                    response = requests.post(
                        f"{API_BASE_URL}/sequence",
                        json=payload,
                        timeout=10
                    )

                    if response.status_code != 200:
                        st.error(f"API Error at timestep {i+1}")
                        break

                    result = response.json()
                    prediction = result["prediction"]

                    THRESHOLD = 0.5
                    is_risk = prediction >= THRESHOLD

                    with monitor_box.container():
                        st.subheader(f"Timestep {i+1}")

                        # VITALS BOX (TOP)
                        with st.container(border=True):
                            st.markdown("### Vitals")

                            # -------- Row 1 --------
                            r1 = st.columns(5)
                            vitals_r1 = [
                                ("Heart Rate", "heart_rate"),
                                ("Systolic BP", "systolic_bp"),
                                ("Diastolic BP", "diastolic_bp"),
                                ("Respiratory Rate", "respiratory_rate"),
                                ("SpO₂", "spo2"),
                            ]

                            for col, (label, key) in zip(r1, vitals_r1):
                                with col:
                                    with st.container(border=True):
                                        st.metric(label, round(df.loc[i, key], 2))

                            # -------- Row 2 --------
                            r2 = st.columns(4)
                            vitals_r2 = [
                                ("Temperature", "temperature"),
                                ("Creatinine", "creatinine"),
                                ("WBC", "wbc"),
                                ("Lactate", "lactate"),
                            ]

                            for col, (label, key) in zip(r2, vitals_r2):
                                with col:
                                    with st.container(border=True):
                                        st.metric(label, round(df.loc[i, key], 2))

                        # RISK & STATUS BOX
                        with st.container(border=True):
                            st.markdown("### Patient Status and  Risk Probability")

                            r3c1, r3c2 = st.columns(2)
                            with r3c1:
                                with st.container(border=True):
                                    if is_risk:
                                        st.error("\nPatient Status : Abnormal")
                                    else:
                                        st.success("\nPatient Status : Normal")
                            with r3c2:
                                with st.container(border=True):
                                    if is_risk:
                                        st.error(f"""**Risk Probability :** **{round(prediction, 2)}**""")
                                    else:
                                        st.success(f"""**Risk Probability :** **{round(prediction, 2)}**""")
                    time.sleep(interval)

            st.info("Patient monitoring completed.")

# 7. Sentiment Analysis
if menu == "Sentiment Analysis":
    st.title("Sentiment Analysis")

    st.markdown(
        "Analyze patient feedback sentiment using a deep learning NLP model. "
        "Supports **single text analysis** and **bulk Excel analysis**."
    )

    tab1, tab2 = st.tabs(["Single Analysis", "Bulk Analysis (Excel)"])


    # TAB 1: Single Sentiment Analysis

    with tab1:
        st.subheader("Single Feedback Analysis")

        with st.container(border=True):
            text_input = st.text_area(
                "Enter Patient Feedback",
                height=150,
                placeholder="The hospital staff were very supportive and caring..."
            )

        predict_btn = st.button("Analyze Sentiment")

        if predict_btn:
            if text_input.strip() == "":
                st.warning("Please enter feedback text before analysis.")
            else:
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/sentiment",
                        json={"text": text_input},
                        timeout=10
                    )

                    if response.status_code == 200:
                        result = response.json()

                        with st.container(border=True):
                            st.subheader("Sentiment Result")

                            if result["label"] == "Positive":
                                st.success("Sentiment: Positive")
                            else:
                                st.error("Sentiment: Negative")

                            st.metric(
                                label="Confidence Score",
                                value=round(result["probability"], 3)
                            )
                    else:
                        st.error(f"API Error: {response.status_code}")

                except requests.exceptions.ConnectionError:
                    st.error("FastAPI server is not running.")
                except Exception as e:
                    st.error(f"Sentiment analysis failed: {e}")


    # TAB 2: Bulk Sentiment Analysis (Excel)

    with tab2:
        st.subheader("Bulk Sentiment Analysis")

        st.markdown(
            """
            **Excel File Requirements**
            - File format: `.xlsx`
            - Must contain a column named **`text`**
            """
        )

        with st.container(border=True):
            uploaded_file = st.file_uploader(
                "Upload Excel File",
                type=["xlsx"]
            )

        if uploaded_file is not None:
            try:
                import pandas as pd
                import io

                df = pd.read_excel(uploaded_file)

                if "text" not in df.columns:
                    st.error("Excel file must contain a column named 'text'.")
                else:
                    with st.container(border=True):
                        st.subheader("Data Preview")
                        st.dataframe(df.head())

                    run_bulk_btn = st.button("Run Bulk Sentiment Analysis")

                    if run_bulk_btn:
                        sentiments = []
                        confidences = []

                        with st.spinner("Analyzing feedback sentiments..."):
                            for text in df["text"]:
                                response = requests.post(
                                    f"{API_BASE_URL}/sentiment",
                                    json={"text": str(text)},
                                    timeout=10
                                )

                                if response.status_code == 200:
                                    res = response.json()
                                    sentiments.append(res["label"])
                                    confidences.append(round(res["probability"], 3))
                                else:
                                    sentiments.append("Error")
                                    confidences.append(None)

                        df["sentiment"] = sentiments
                        df["confidence"] = confidences

                        with st.container(border=True):
                            st.success("Bulk sentiment analysis completed.")
                            st.dataframe(df)

                        # Download results
                        output = io.BytesIO()
                        df.to_excel(output, index=False)

                        st.download_button(
                            label="Download Results as Excel",
                            data=output.getvalue(),
                            file_name="sentiment_analysis_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

            except Exception as e:
                st.error(f"Failed to process Excel file: {e}")

