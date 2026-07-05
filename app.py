"""
Handwritten Digit Recognition - Web App (Streamlit)
------------------------------------------------------
Draw a digit in the browser and the trained CNN predicts it live.
Runs on pure NumPy (no TensorFlow) so it deploys reliably anywhere.

Run locally:
    streamlit run app.py

Deploy for free:
    https://share.streamlit.io  (Streamlit Community Cloud)
"""

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from numpy_model import DigitCNN

WEIGHTS_PATH = "model_weights.npz"
CANVAS_SIZE = 280
IMG_SIZE = 28


def preprocess_centered(gray_arr, pad=2):
    """
    Convert a raw drawn image into the same format MNIST digits use:
    cropped to the digit's bounding box, scaled to fit a 20x20 box,
    and centered in a 28x28 frame. This matters a lot for freehand
    drawings, since they're rarely centered/sized the way MNIST is.
    """
    arr = gray_arr.astype("float32")
    coords = np.argwhere(arr > 10)
    if coords.size == 0:
        return np.zeros((28, 28), dtype="float32")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    h_full, w_full = arr.shape
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(h_full, y1 + pad)
    x1 = min(w_full, x1 + pad)
    cropped = arr[y0:y1, x0:x1]

    h, w = cropped.shape
    if h > w:
        new_h = 20
        new_w = max(1, round(w * (20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, round(h * (20.0 / w)))

    cropped_img = Image.fromarray(cropped.astype("uint8")).resize((new_w, new_h), Image.LANCZOS)
    cropped_arr = np.array(cropped_img).astype("float32")

    canvas = np.zeros((28, 28), dtype="float32")
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = cropped_arr
    return canvas / 255.0

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️", layout="centered")


@st.cache_resource
def load_model():
    return DigitCNN(WEIGHTS_PATH)


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
    row_placeholders = [st.empty() for _ in range(10)]

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")

    # canvas gives RGBA; convert to grayscale the same way MNIST expects
    gray = Image.fromarray(img).convert("L")

    if np.array(gray).sum() > 0:
        arr = preprocess_centered(np.array(gray))

        probs = model.predict(arr)
        pred = int(np.argmax(probs))

        pred_placeholder.markdown(f"## {pred}")
        for digit, ph in enumerate(row_placeholders):
            with ph.container():
                c1, c2, c3 = st.columns([0.6, 3, 1])
                c1.write(f"**{digit}**")
                c2.progress(float(probs[digit]))
                c3.write(f"{probs[digit] * 100:.1f}%")
    else:
        pred_placeholder.markdown("## —")
        for digit, ph in enumerate(row_placeholders):
            with ph.container():
                c1, c2, c3 = st.columns([0.6, 3, 1])
                c1.write(f"**{digit}**")
                c2.progress(0.0)
                c3.write("0.0%")

st.caption("Model: CNN trained on MNIST (NumPy inference) · Test accuracy: 99.10%")


