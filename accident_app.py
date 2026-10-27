import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf

st.set_page_config(
    page_title="Accident Severity Prediction",
    page_icon="🚗",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_files():
    model = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "accident_severity_model.keras")
    )
    preprocessor = joblib.load(
        os.path.join(BASE_DIR, "preprocessor.joblib")
    )
    label_encoder = joblib.load(
        os.path.join(BASE_DIR, "label_encoder.joblib")
    )
    return model, preprocessor, label_encoder

st.title("🚗 Accident Severity Prediction")
st.write(
    "Enter road, time and weather conditions to estimate "
    "the severity category using our trained machine-learning model."
)

try:
    model, preprocessor, label_encoder = load_files()
except Exception as e:
    st.error("Could not load the model files.")
    st.code(str(e))
    st.stop()

with st.form("prediction_form"):
    day = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"]
    )

    time = st.text_input("Time (HH:MM)", value="18:30")

    road = st.selectbox(
        "Road Type",
        ["Single carriageway", "Dual carriageway",
         "Roundabout", "One way street", "Slip road"]
    )

    speed = st.selectbox(
        "Speed Limit (mph)", [20, 30, 40, 50, 60, 70],
        index=3
    )

    light = st.selectbox(
        "Light Conditions",
        ["Daylight", "Darkness - lights lit",
         "Darkness - lights unlit",
         "Darkness - no lighting",
         "Darkness - lighting unknown"]
    )

    weather = st.selectbox(
        "Weather Conditions",
        ["Fine no high winds", "Raining no high winds",
         "Raining + high winds", "Snowing no high winds",
         "Fog or mist", "Other"]
    )

    surface = st.selectbox(
        "Road Surface Conditions",
        ["Dry", "Wet or damp", "Snow", "Frost or ice",
         "Flood over 3cm", "Data missing or out of range"]
    )

    area = st.selectbox(
        "Area",
        ["Urban", "Rural"]
    )

    submitted = st.form_submit_button("Predict Severity")

if submitted:
    try:
        if len(time.strip()) != 5 or time[2] != ":":
            st.error("Enter time in HH:MM format, for example 18:30.")
            st.stop()

        hour, minute = map(int, time.split(":"))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            st.error("Enter a valid 24-hour time, such as 09:15 or 18:30.")
            st.stop()

        sample = pd.DataFrame([{
            "Day_of_Week": day,
            "Time": time.strip(),
            "Road_Type": road,
            "Speed_limit": speed,
            "Light_Conditions": light,
            "Weather_Conditions": weather,
            "Road_Surface_Conditions": surface,
            "Urban_or_Rural_Area": area
        }])

        processed = preprocessor.transform(sample)

        if hasattr(processed, "toarray"):
            processed = processed.toarray()

        processed = np.asarray(processed, dtype=np.float32)

        probabilities = model.predict(processed, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = label_encoder.inverse_transform(
            [predicted_index]
        )[0]

        st.subheader("Prediction Result")
        st.success(f"Predicted severity category: {predicted_class}")

        st.write("Model output scores:")
        for name, score in zip(label_encoder.classes_, probabilities):
            st.write(f"**{name}:** {score:.1%}")
            st.progress(float(score))

        st.warning(
            "This is an experimental model trained on historical "
            "accident records. The scores are not calibrated real-world "
            "probabilities and do not predict whether an accident will occur."
        )

    except Exception as e:
        st.error("Prediction failed. Check the input categories and model.")
        st.code(str(e))
