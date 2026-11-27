"""Web demo.  Run from the project root:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src import config
from src.detect import annotate, load_face_detector

st.set_page_config(page_title="Face Mask Detection", page_icon="😷")
st.title("😷 Face Mask Detection")
st.write("Upload a photo or take one with your camera. Faces are marked green (mask) or red (no mask).")


@st.cache_resource
def get_models():
    return tf.keras.models.load_model(config.MODEL_PATH), load_face_detector()


try:
    model, detector = get_models()
except Exception as e:
    st.error(f"Model not found. Train it first with `python -m src.train`.\n\n{e}")
    st.stop()

source = st.radio("Input", ["Upload image", "Camera"], horizontal=True)
file = st.file_uploader("Image", type=["jpg", "jpeg", "png"]) if source == "Upload image" \
    else st.camera_input("Take a photo")

if file:
    rgb = np.array(Image.open(file).convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr, results = annotate(bgr, model, detector)
    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    if not results:
        st.info("No face detected. Try a clearer, front-facing photo.")
    for i, (label, conf, _) in enumerate(results, 1):
        (st.success if label == "Mask" else st.error)(f"Face {i}: {label} ({conf * 100:.1f}%)")
