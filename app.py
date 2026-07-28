# ============================================================
# AI Powered Food Delivery Time Prediction
# Developed by Naveen Kumar
# ============================================================

import streamlit as st
import time

from utils import (
    load_model,
    validate_inputs,
    encode_inputs,
    create_dataframe,
    scale_data,
    predict_delivery_time,
    create_summary
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_resources():
    return load_model()

model, scaler, encoder = load_resources()

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
padding-left:3rem;
padding-right:3rem;
}

.title{
text-align:center;
font-size:45px;
font-weight:bold;
color:#1976D2;
margin-bottom:5px;
}

.subtitle{
text-align:center;
font-size:18px;
color:gray;
margin-bottom:25px;
}

.result-card{
background:linear-gradient(135deg,#11998e,#38ef7d);
padding:30px;
border-radius:20px;
color:white;
text-align:center;
box-shadow:0px 8px 20px rgba(0,0,0,0.2);
margin-top:20px;
margin-bottom:20px;
}

.result-title{
font-size:24px;
font-weight:bold;
}

.result-value{
font-size:45px;
font-weight:bold;
}

.stButton > button{

width:100%;
height:55px;

background:#1976D2;
color:white;

font-size:18px;
font-weight:bold;

border-radius:10px;

border:none;

transition:0.3s;

}

.stButton > button:hover{

background:#0D47A1;
color:white;

}

</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📌 About Project")

    st.write("""
This application predicts the **Estimated Delivery Time**
using a trained **Gradient Boosting Regressor** model.
""")

    st.divider()

    st.subheader("🤖 Model")

    st.success("Gradient Boosting Regressor")

    st.write("Problem Type : Regression")

    st.write("Input Features : 7")

    st.write("Output : Delivery Time")

    st.divider()

    st.subheader("📊 Features")

    st.write("""
📍 Distance

🌦 Weather

🚦 Traffic Level

🚗 Vehicle Type

🕒 Time of Day

👨‍🍳 Preparation Time

👨‍💼 Courier Experience
""")

    st.divider()

    st.subheader("👨‍💻 Developer")

    st.info("Naveen Kumar")

# ============================================================
# HEADER
# ============================================================

st.markdown(
"""
<div class='title'>
🚚 Food Delivery Time Prediction
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='subtitle'>
AI Powered Delivery Time Estimation using Machine Learning
</div>
""",
unsafe_allow_html=True
)

st.divider()

# ============================================================
# INPUT SECTION
# ============================================================

left,right = st.columns(2)

with left:

    distance = st.number_input(
        "📍 Distance (km)",
        min_value=0.0,
        step=0.1
    )

    preparation = st.number_input(
        "👨‍🍳 Preparation Time (Minutes)",
        min_value=0,
        step=1
    )

    experience = st.number_input(
        "👨‍💼 Courier Experience (Years)",
        min_value=0,
        step=1
    )

with right:

    weather = st.selectbox(
        "🌦 Weather",
        [
            "Clear",
            "Foggy",
            "Rainy",
            "Snowy",
            "Windy"
        ]
    )

    traffic = st.selectbox(
        "🚦 Traffic Level",
        [
            "High",
            "Medium",
            "Low"
        ]
    )

    vehicle = st.selectbox(
        "🚗 Vehicle Type",
        [
            "Bike",
            "Scooter",
            "Car"
        ]
    )

    time_of_day = st.selectbox(
        "🕒 Time of Day",
        [
            "Morning",
            "Afternoon",
            "Evening",
            "Night"
        ]
    )

st.divider()

predict = st.button(
    "🚚 Predict Delivery Time",
    use_container_width=True
)
# ============================================================
# PREDICTION SECTION
# ============================================================

if predict:

    # ------------------------------
    # Input Validation
    # ------------------------------

    is_valid, message = validate_inputs(
        distance,
        preparation,
        experience
    )

    if not is_valid:
        st.warning(message)
        st.stop()

    # ------------------------------
    # Loading Animation
    # ------------------------------

    with st.spinner("🤖 AI is predicting delivery time..."):

        time.sleep(1)

        # Save original values for summary

        weather_input = weather
        traffic_input = traffic
        vehicle_input = vehicle
        time_input = time_of_day

        # ------------------------------
        # Encode categorical variables
        # ------------------------------

        (
            weather_encoded,
            traffic_encoded,
            vehicle_encoded,
            time_encoded

        ) = encode_inputs(

            weather,
            traffic,
            vehicle,
            time_of_day,
            encoder

        )

        # ------------------------------
        # Create DataFrame
        # ------------------------------

        new_data = create_dataframe(

            distance,

            weather_encoded,

            traffic_encoded,

            time_encoded,

            vehicle_encoded,

            preparation,

            experience

        )

        # ------------------------------
        # Scale Data
        # ------------------------------

        scaled_data = scale_data(
            new_data,
            scaler
        )

        # ------------------------------
        # Predict
        # ------------------------------

        prediction = predict_delivery_time(
            model,
            scaled_data
        )

    # ============================================================
    # SUCCESS
    # ============================================================

    st.balloons()

    st.success("Prediction Completed Successfully!")

    # ============================================================
    # RESULT CARD
    # ============================================================

  

    st.markdown("## 🚚 Estimated Delivery Time")

    st.markdown(
    f"""
### ⏱ **{prediction:.2f} Minutes**
""")


    st.info(
        "Prediction generated using a trained Gradient Boosting Regressor model."
    )

    st.divider()
    # ============================================================
    # PREDICTION SUMMARY
    # ============================================================

    st.subheader("📋 Prediction Summary")

    summary = create_summary(
        distance,
        weather_input,
        traffic_input,
        vehicle_input,
        time_input,
        preparation,
        experience
    )

    st.table(summary)

    st.divider()

    # ============================================================
    # VIEW ENCODED DATA
    # ============================================================

    with st.expander("🔍 View Encoded Input Data"):

        encoded_df = create_dataframe(
            distance,
            weather_encoded,
            traffic_encoded,
            time_encoded,
            vehicle_encoded,
            preparation,
            experience
        )

        st.dataframe(
            encoded_df,
            use_container_width=True
        )

    st.divider()

    # ============================================================
    # QUICK INSIGHTS
    # ============================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="📍 Distance",
            value=f"{distance:.1f} km"
        )

    with col2:
        st.metric(
            label="👨‍🍳 Preparation",
            value=f"{preparation} min"
        )

    with col3:
        st.metric(
            label="👨‍💼 Experience",
            value=f"{experience} yrs"
        )

    st.divider()

    # ============================================================
    # DISCLAIMER
    # ============================================================

    st.caption(
        "⚠ This prediction is generated using a Machine Learning model "
        "and may vary depending on real-world delivery conditions."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;
                color:gray;
                font-size:16px;'>

    🚚 <b>AI Powered Food Delivery Time Prediction</b><br><br>

    🤖 Model : <b>Gradient Boosting Regressor</b><br>

    👨‍💻 Developed by <b>Naveen Kumar</b><br>

    © 2026 All Rights Reserved.

    </div>
    """,
    unsafe_allow_html=True
)