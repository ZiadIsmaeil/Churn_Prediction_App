from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

project_root = Path(__file__).resolve().parents[1]

model_path = project_root / "models" / "churn_model.pkl"
threshold_path = project_root / "models" / "threshold.json"
background_path = project_root / "models" / "X_train_background.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(model_path)


@st.cache_data
def load_threshold():
    with open(threshold_path, "r") as f:
        return json.load(f)["threshold"]


@st.cache_data
def load_background():
    return joblib.load(background_path)


model = load_model()
threshold = load_threshold()
background_data = load_background()


# ============================================================
# MODEL COMPONENTS
# ============================================================

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_original_feature(transformed_feature):

    feature = (
        transformed_feature
        .replace("num__", "")
        .replace("cat__", "")
    )

    if feature in numerical_features:
        return feature

    for base_feature in categorical_features:
        prefix = base_feature + "_"

        if feature.startswith(prefix):
            return base_feature

    return feature


def friendly_feature_name(feature):

    names = {
        "SeniorCitizen": "Senior Citizen",
        "tenure": "Tenure",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
        "gender": "Gender",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "PhoneService": "Phone Service",
        "MultipleLines": "Multiple Lines",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "Contract": "Contract",
        "PaperlessBilling": "Paperless Billing",
        "PaymentMethod": "Payment Method"
    }

    return names.get(feature, feature)


def format_customer_value(feature, value):

    if feature == "SeniorCitizen":
        return "Yes" if value == 1 else "No"

    if feature in ["MonthlyCharges", "TotalCharges"]:
        return f"${float(value):,.2f}"

    if feature == "tenure":
        return f"{int(value)} months"

    return str(value)


def calculate_shap(customer_data):

    background_transformed = preprocessor.transform(background_data)
    customer_transformed = preprocessor.transform(customer_data)

    background_dense = (
        background_transformed.toarray()
        if hasattr(background_transformed, "toarray")
        else background_transformed
    )

    customer_dense = (
        customer_transformed.toarray()
        if hasattr(customer_transformed, "toarray")
        else customer_transformed
    )

    feature_names = preprocessor.get_feature_names_out()

    background_sample = background_dense[:100]

    explainer = shap.LinearExplainer(
        classifier,
        background_sample
    )

    shap_result = explainer(customer_dense)

    return (
        shap_result,
        customer_dense,
        feature_names
    )


def group_shap_values(
    shap_result,
    feature_names,
    customer_data
):

    shap_values = shap_result.values[0]

    grouped = {}

    for feature_name, shap_value in zip(
        feature_names,
        shap_values
    ):

        original_feature = get_original_feature(feature_name)

        grouped.setdefault(
            original_feature,
            0
        )

        grouped[original_feature] += shap_value

    rows = []

    for feature, contribution in grouped.items():

        value = customer_data.iloc[0][feature]

        rows.append({
            "feature": feature,
            "display_name": friendly_feature_name(feature),
            "value": format_customer_value(feature, value),
            "contribution": contribution,
            "abs_contribution": abs(contribution)
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "abs_contribution",
        ascending=False
    )

    return result


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Customer Profile")

    st.caption(
        "Enter customer information to generate a churn prediction."
    )

    st.divider()

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0
    )

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

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No internet service", "No", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No internet service", "No", "Yes"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No internet service", "No", "Yes"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No internet service", "No", "Yes"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No internet service", "No", "Yes"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No internet service", "No", "Yes"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
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

    predict_button = st.button(
        "🔍 Analyze Customer",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# CUSTOMER DATA
# ============================================================

customer_data = pd.DataFrame([{
    "SeniorCitizen": senior_citizen,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "gender": gender,
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method
}])


# ============================================================
# HEADER
# ============================================================

st.title("Customer Churn Intelligence")

st.caption(
    "Machine Learning-powered customer churn prediction "
    "with model explainability."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🤖 Balanced Logistic Regression")

with col2:
    st.info("🔎 SHAP Explainability")

with col3:
    st.info(f"🎯 Decision Threshold: {threshold:.2f}")


st.divider()


# ============================================================
# PREDICTION
# ============================================================

probability = None
prediction = None
shap_result = None
grouped_shap = None


if predict_button:

    probability = model.predict_proba(
        customer_data
    )[0][1]

    prediction = int(
        probability >= threshold
    )

    shap_result, customer_dense, feature_names = calculate_shap(
        customer_data
    )

    grouped_shap = group_shap_values(
        shap_result,
        feature_names,
        customer_data
    )


# ============================================================
# TABS
# ============================================================

tab_prediction, tab_explainability, tab_model = st.tabs(
    [
        "📊 Prediction",
        "🔎 Explainability",
        "📈 Model Performance"
    ]
)


# ============================================================
# TAB 1 — PREDICTION
# ============================================================

with tab_prediction:

    if probability is None:

        st.subheader("Customer Churn Prediction")

        st.write(
            "Enter the customer information from the sidebar "
            "and click **Analyze Customer** to generate a prediction."
        )

        st.info(
            "The model estimates the probability that this customer "
            "will churn based on the provided characteristics."
        )

    else:

        st.subheader("Prediction Dashboard")

        probability_percent = probability * 100

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                "Churn Probability",
                f"{probability_percent:.1f}%"
            )

        with metric_col2:

            if prediction == 1:
                st.metric(
                    "Prediction",
                    "Likely to Churn"
                )
            else:
                st.metric(
                    "Prediction",
                    "Predicted Not to Churn"
                )

        with metric_col3:
            st.metric(
                "Decision Threshold",
                f"{threshold:.2f}"
            )

        st.write("")

        st.subheader("Churn Probability")

        st.progress(
            float(probability)
        )

        st.caption(
            f"Model probability: {probability_percent:.1f}%"
        )

        if prediction == 1:

            st.warning(
                "The model predicts that this customer is likely to churn."
            )

        else:

            st.success(
                "The model predicts that this customer is not likely to churn."
            )

        st.divider()

        st.subheader("Customer Profile")

        profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)

        with profile_col1:
            st.metric(
                "Tenure",
                f"{tenure} months"
            )

        with profile_col2:
            st.metric(
                "Monthly Charges",
                f"${monthly_charges:,.2f}"
            )

        with profile_col3:
            st.metric(
                "Contract",
                contract
            )

        with profile_col4:
            st.metric(
                "Internet Service",
                internet_service
            )

        st.write("")

        st.subheader("Key Factors")

        if grouped_shap is not None:

            top_factors = grouped_shap.head(6)

            for _, row in top_factors.iterrows():

                contribution = row["contribution"]

                if contribution > 0:

                    icon = "🔴"
                    direction = "Increased predicted churn risk"

                else:

                    icon = "🟢"
                    direction = "Reduced predicted churn risk"

                with st.container():

                    factor_col1, factor_col2, factor_col3 = st.columns(
                        [2.2, 2.0, 2.5]
                    )

                    with factor_col1:
                        st.markdown(
                            f"**{icon} {row['display_name']}**"
                        )

                    with factor_col2:
                        st.write(
                            f"Value: **{row['value']}**"
                        )

                    with factor_col3:
                        st.write(
                            f"{direction}"
                        )

                    st.caption(
                        f"Model contribution: {contribution:+.4f}"
                    )

                    st.divider()


# ============================================================
# TAB 2 — EXPLAINABILITY
# ============================================================

with tab_explainability:

    st.subheader("Model Explainability")

    if shap_result is None:

        st.info(
            "Run a prediction first to generate the SHAP explanation."
        )

    else:

        st.write(
            "SHAP explains how each feature contributed to this "
            "individual prediction relative to the model's background data."
        )

        st.divider()

        st.subheader("Top Contributing Features")

        explanation_table = grouped_shap.head(8).copy()

        explanation_table["Direction"] = explanation_table[
            "contribution"
        ].apply(
            lambda x:
            "Increases predicted churn"
            if x > 0
            else "Reduces predicted churn"
        )

        explanation_table["Contribution"] = explanation_table[
            "contribution"
        ].map(
            lambda x: f"{x:+.4f}"
        )

        display_table = explanation_table[
            [
                "display_name",
                "value",
                "Direction",
                "Contribution"
            ]
        ].rename(
            columns={
                "display_name": "Feature",
                "value": "Customer Value"
            }
        )

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        st.subheader("SHAP Waterfall")

        shap_waterfall = shap.Explanation(
            values=shap_result.values[0],
            base_values=shap_result.base_values[0],
            data=customer_dense[0],
            feature_names=feature_names
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        shap.plots.waterfall(
            shap_waterfall,
            max_display=10,
            show=False
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.divider()

        st.subheader("Technical Validation")

        model_output = classifier.decision_function(
            customer_dense
        )[0]

        base_value = np.array(
            shap_result.base_values
        ).flatten()[0]

        shap_sum = np.sum(
            shap_result.values
        )

        reconstructed_output = (
            base_value + shap_sum
        )

        difference = abs(
            model_output - reconstructed_output
        )

        validation_col1, validation_col2, validation_col3 = st.columns(3)

        with validation_col1:
            st.metric(
                "Model Output",
                f"{model_output:.6f}"
            )

        with validation_col2:
            st.metric(
                "SHAP Reconstruction",
                f"{reconstructed_output:.6f}"
            )

        with validation_col3:
            st.metric(
                "Difference",
                f"{difference:.10f}"
            )

        if difference < 1e-5:

            st.success(
                "✓ SHAP explanation is mathematically consistent "
                "with the model output."
            )

        else:

            st.warning(
                "SHAP reconstruction difference is above the expected tolerance."
            )

        st.caption(
            "SHAP contributions are measured in the model's "
            "log-odds/output space. They represent how features "
            "shift the model prediction relative to the background data; "
            "they should not be interpreted as causal effects."
        )


# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================

with tab_model:

    st.subheader("Model Performance")

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Score": [
            0.7409252669,
            0.5069930070,
            0.7795698925,
            0.6144067797,
            0.8398455797
        ]
    })

    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

    with metric_col1:
        st.metric(
            "Accuracy",
            "74.1%"
        )

    with metric_col2:
        st.metric(
            "Precision",
            "50.7%"
        )

    with metric_col3:
        st.metric(
            "Recall",
            "78.0%"
        )

    with metric_col4:
        st.metric(
            "F1 Score",
            "61.4%"
        )

    with metric_col5:
        st.metric(
            "ROC-AUC",
            "84.0%"
        )

    st.divider()

    st.subheader("Evaluation Summary")

    st.dataframe(
        metrics.style.format(
            {"Score": "{:.1%}"}
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Model Configuration")

    config_col1, config_col2 = st.columns(2)

    with config_col1:

        st.write("**Algorithm**")
        st.write("Balanced Logistic Regression")

        st.write("**Class Handling**")
        st.write("class_weight = balanced")

        st.write("**Decision Threshold**")
        st.write(f"{threshold:.2f}")

    with config_col2:

        st.write("**Preprocessing**")
        st.write("StandardScaler + OneHotEncoder")

        st.write("**Explainability**")
        st.write("SHAP LinearExplainer")

        st.write("**Dataset**")
        st.write("IBM Telco Customer Churn")

    st.divider()

    st.subheader("Why Recall Matters")

    st.write(
        "Customer churn is an imbalanced classification problem. "
        "The balanced model was selected to increase its ability to "
        "identify customers who may churn, resulting in higher recall "
        "while accepting a lower precision compared with the standard "
        "Logistic Regression model."
    )

    st.info(
        "The displayed threshold of 0.50 is the selected evaluation "
        "threshold from the current project experiments. A production "
        "deployment would normally tune the threshold according to "
        "business costs and intervention capacity."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Intelligence • Machine Learning + Explainable AI"
)