import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and features
model = joblib.load("model/final_intrusion_model.pkl")
feature_names = joblib.load("model/feature_names.pkl")

st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Network Intrusion Detection System")

st.write("This system predicts whether network traffic is normal or an intrusion attack.")

st.divider()

# -------------------------
# Input Section
# -------------------------

st.subheader("Enter Network Traffic Features")

duration = st.number_input("Duration", min_value=0, value=0)

src_bytes = st.number_input("Source Bytes", min_value=0, value=0)

dst_bytes = st.number_input("Destination Bytes", min_value=0, value=0)

count = st.number_input("Connection Count", min_value=0, value=0)

srv_count = st.number_input("Service Count", min_value=0, value=0)

protocol = st.selectbox(
    "Protocol Type",
    ["tcp", "udp", "icmp"]
)

flag = st.selectbox(
    "Connection Flag",
    ["SF", "S0", "REJ", "RSTO", "SH"]
)

st.divider()

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Attack Type"):

    # Create empty feature dictionary
    input_dict = dict.fromkeys(feature_names, 0)

    # Fill numeric values
    input_dict["duration"] = duration
    input_dict["src_bytes"] = src_bytes
    input_dict["dst_bytes"] = dst_bytes
    input_dict["count"] = count
    input_dict["srv_count"] = srv_count

    # Handle protocol encoding
    protocol_feature = f"protocol_type_{protocol}"
    if protocol_feature in input_dict:
        input_dict[protocol_feature] = 1

    # Handle flag encoding
    flag_feature = f"flag_{flag}"
    if flag_feature in input_dict:
        input_dict[flag_feature] = 1

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)[0]

    # Confidence score
    probabilities = model.predict_proba(input_df)[0]

    confidence = np.max(probabilities) * 100

    st.divider()

    # Display result
    if prediction == "normal":
        st.success(f"Prediction: {prediction}")
    else:
        st.error(f"Prediction: {prediction}")

    st.info(f"Confidence: {confidence:.2f}%")

st.divider()

st.caption("Built using Machine Learning and Streamlit")