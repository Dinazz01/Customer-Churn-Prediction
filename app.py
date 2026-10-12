
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Page configuration
# ==========================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction System")

st.write(
    "Predict customer churn using an optimized "
    "Logistic Regression machine learning model."
)

# ==========================================
# Load trained model
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("customer_churn_model.pkl")

artifact = load_model()

model = artifact["model"]
threshold = artifact["threshold"]
feature_columns = artifact["feature_columns"]

# ==========================================
# Customer information
# ==========================================
st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Customer Age", 18, 100, 40
    )

    tenure_months = st.number_input(
        "Tenure (Months)", 0, 120, 12
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        0.0, 1000.0, 70.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        0.0, 100000.0, 840.0
    )

    contract_type = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "Unknown"]
    )

    region = st.selectbox(
        "Region",
        ["East", "North", "South", "West"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

with col2:
    support_calls = st.number_input(
        "Support Calls", 0, 50, 2
    )

    late_payments = st.number_input(
        "Late Payments", 0, 50, 0
    )

    num_services = st.number_input(
        "Number of Services", 0, 20, 3
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer",
            "Credit card",
            "Electronic check",
            "Mailed check",
            "Unknown"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    streaming = st.selectbox(
        "Streaming",
        ["Yes", "No"]
    )

    tech_support = st.selectbox(
        "Technical Support",
        ["Yes", "No", "Unknown"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "Unknown"]
    )

# ==========================================
# Create input DataFrame
# ==========================================
customer_data = {
    "age": age,
    "senior_citizen": senior_citizen,
    "region": region,
    "tenure_months": tenure_months,
    "contract_type": contract_type,
    "internet_service": internet_service,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "payment_method": payment_method,
    "paperless_billing": paperless_billing,
    "num_services": num_services,
    "support_calls": support_calls,
    "late_payments": late_payments,
    "streaming": streaming,
    "tech_support": tech_support,
    "device_protection": device_protection
}

input_df = pd.DataFrame([customer_data])

# Ensure all model features are present
missing_columns = (
    set(feature_columns) - set(input_df.columns)
)

if missing_columns:
    st.error(
        f"Missing model features: {sorted(missing_columns)}"
    )
    st.stop()

input_df = input_df[feature_columns]

# ==========================================
# Generate prediction
# ==========================================
if st.button("Predict Customer Churn"):

    probability = model.predict_proba(input_df)[0, 1]

    prediction = int(probability >= threshold)

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Classification Threshold",
            f"{threshold:.0%}"
        )

    st.progress(float(probability))

    if prediction == 1:
        st.error(
            "High Churn Risk: Customer is predicted "
            "to churn."
        )

        st.warning(
            "Recommended Action: Review customer "
            "satisfaction and consider retention offers."
        )

    else:
        st.success(
            "Low Churn Risk: Customer is predicted "
            "to stay."
        )

        st.info(
            "Recommended Action: Maintain customer "
            "engagement and service quality."
        )

    st.subheader("Customer Data")
    st.dataframe(input_df, use_container_width=True)
