import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import string

st.set_page_config(page_title="Sign Language Recognition", layout="centered")
st.title("🤟 Sign Language Recognition")

# Load model from same directory
model = tf.keras.models.load_model("m1_91_3.h5")

# Class labels: A-Z without 'J' and 'Z' (if 25 output classes)
class_labels = list(string.ascii_uppercase)
if model.output_shape[1] == 25:
    class_labels.remove('J')
    class_labels.remove('Z')

# Capture input from camera
img_file_buffer = st.camera_input("Take a picture of your hand ✋")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer).convert("RGB")
    img_np = np.array(img)

    # Preprocess
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (28, 28))
    reshaped = resized.reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Predict
    prediction = model.predict(reshaped)
    predicted_index = np.argmax(prediction)
    predicted_letter = class_labels[predicted_index]

    st.image(img_np, caption="Captured Hand Sign", use_column_width=True)
    st.success(f"Predicted Letter: **{predicted_letter}**")
