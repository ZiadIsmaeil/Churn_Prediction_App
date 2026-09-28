import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_model():

    # Project folder:
    # E:\customer_churn_prediction\Churn_Prediction_App

    project_root = Path(__file__).resolve().parents[1]

    model_path = project_root / "models" / "churn_model.pkl"
    threshold_path = project_root / "models" / "threshold.json"

    model = joblib.load(model_path)

    with open(threshold_path, "r") as f:
        threshold = json.load(f)["threshold"]

    return model, threshold


model, threshold = load_model()


st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    Predict whether a customer is likely to churn based on
    their demographic information, subscription details,
    services, and billing information.
    """
)

st.divider()

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )


with col2:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


with col3:

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


st.divider()

st.subheader("Additional Services")

col1, col2, col3 = st.columns(3)


with col1:

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with col2:

    device_protection = st.selectbox(
        "Device Protection",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


with col3:

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "Yes",
            "No",
            "No internet service"
        ]
    )


st.divider()

st.subheader("Billing")

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)


st.divider()


if st.button(
    "Predict Churn",
    type="primary",
    use_container_width=True
):

    customer_data = pd.DataFrame({
        "SeniorCitizen": [senior_citizen],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "gender": [gender],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method]
    })

    probability = model.predict_proba(
        customer_data
    )[0][1]

    prediction = int(
        probability >= threshold
    )

    st.divider()

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )


    with result_col2:

        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to churn"
            )

        else:

            st.success(
                "✅ Customer is unlikely to churn"
            )


    st.progress(
        float(probability)
    )

    st.caption(
        f"Prediction threshold: {threshold:.2f}"
    )