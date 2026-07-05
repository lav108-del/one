"""
Handwritten Digit Recognition - Web App (Streamlit)
------------------------------------------------------
Draw a digit in the browser and the trained CNN predicts it live.
This is the web-deployable version of predict_gui.py.

Run locally:
    streamlit run app.py

Deploy for free:
    https://share.streamlit.io  (Streamlit Community Cloud)
"""

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from tensorflow import keras

MODEL_PATH = "digit_model.keras"
CANVAS_SIZE = 280
IMG_SIZE = 28

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️", layout="centered")


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


model = load_model()

st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) below and the neural network will predict it in real time.")

col1, col2 = st.columns([1.2, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    pred_placeholder = st.empty()
    st.subheader("Confidence")
    bar_placeholders = [st.empty() for _ in range(10)]

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")

    # canvas gives RGBA; convert to grayscale the same way MNIST expects
    gray = Image.fromarray(img).convert("L")

    if np.array(gray).sum() > 0:
        small = gray.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(small).astype("float32") / 255.0
        arr = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        probs = model.predict(arr, verbose=0)[0]
        pred = int(np.argmax(probs))

        pred_placeholder.markdown(f"## {pred}")
        for digit, ph in enumerate(bar_placeholders):
            ph.write(f"{digit}: ")
            ph.progress(float(probs[digit]))
    else:
        pred_placeholder.markdown("## —")
        for digit, ph in enumerate(bar_placeholders):
            ph.write(f"{digit}: ")
            ph.progress(0.0)

st.caption("Model: CNN trained on MNIST · Test accuracy: 99.10%")
