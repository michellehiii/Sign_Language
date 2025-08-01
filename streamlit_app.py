import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import string

st.set_page_config(page_title="Sign Language Recognition", layout="centered")
st.title("Sign Language Recognition")

# ----------------------------
# Load model
# ----------------------------
model = models.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(25, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.load_weights("m1_91_3.h5")
class_labels = list(string.ascii_uppercase)[:25]  # A-Y (assuming 25 classes)

# ----------------------------
# Tabs for Camera and Upload
# ----------------------------
tab1, tab2 = st.tabs(["📷 Take Live Photo", "🖼️ Upload Photo"])

# ----------------------------
# Tab 1 – Camera Input
# ----------------------------
with tab1:
    img_file_buffer = st.camera_input("Take a picture of your hand")
    st.info("📸 Tip: Use a white background when taking your photo for better prediction accuracy.")

    # Custom CSS to style the camera button
    st.markdown("""
      <style>
      div[data-testid="stCameraInputLabel"] {
          font-size: 0px;  /* Hide default label */
      }
      button[kind="secondary"] {
          background-color: #4CAF50;
          color: white;
          font-size: 18px;
          padding: 10px 20px;
          border-radius: 10px;
      }
      </style>
      """, unsafe_allow_html=True)
  
      # Camera input (with hidden label)
      img_file_buffer = st.camera_input(" ")
      if img_file_buffer is not None:
          img = Image.open(img_file_buffer)
          image = np.array(img.convert("RGB"))
          image = cv2.flip(image, 1)

        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (28, 28))
        reshaped = resized.reshape(1, 28, 28, 1).astype('float32') / 255.0

        # Predict
        pred = model.predict(reshaped)
        predicted_class = np.argmax(pred, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        st.image(image, caption="Captured Image", use_column_width=True)
        st.success(f"Predicted: {predicted_label}")

# ----------------------------
# Tab 2 – Upload Image
# ----------------------------
with tab2:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    st.info("📸 Tip: Use a white background when taking your photo for better prediction accuracy.")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        image = np.array(img.convert("RGB"))

        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (28, 28))
        reshaped = resized.reshape(1, 28, 28, 1).astype('float32') / 255.0

        # Predict
        pred = model.predict(reshaped)
        predicted_class = np.argmax(pred, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"Predicted: {predicted_label}")
