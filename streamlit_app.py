import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import string
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

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
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.predicted_class = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # mirror the image

        # Convert to grayscale and resize for model
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        norm = resized.reshape(1, 28, 28, 1).astype("float32") / 255.0

        # Predict
        pred = model.predict(norm)
        class_index = np.argmax(pred)
        self.predicted_class = class_labels[class_index]

        return img  # 🔍 No overlay on video

    def get_prediction(self):
        return self.predicted_class
# ----------------------------
# Tabs for Camera and Upload
# ----------------------------
tab1, tab2,tab3 = st.tabs(["📷 Take Live Photo", "🖼️ Upload Photo","🔤 Live Sign Prediction"])

# ----------------------------
# Tab 1 – Camera Input
# ----------------------------
with tab1:
    img_file_buffer = st.camera_input("Take a picture of your hand")
    st.info("📸 Tip: Use a white background when taking your photo for better prediction accuracy.")
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

        # st.image(image, caption="Captured Image", use_column_width=True)
        st.success(f"Predicted: {predicted_label}")

# ----------------------------
# Tab 2 – Upload Image
# ----------------------------
with tab2:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    st.info("📸 Tip: Use a white background when taking your photo for better prediction accuracy.")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        preview_image = img.resize((200, 200))
        # Display resized preview
        st.image(preview_image, caption="Thumbnail Preview", use_container_width=False)
      
        image = np.array(img.convert("RGB"))

        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (28, 28))
        reshaped = resized.reshape(1, 28, 28, 1).astype('float32') / 255.0

        # Predict
        pred = model.predict(reshaped)
        predicted_class = np.argmax(pred, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        # st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"Predicted: {predicted_label}")
with tab3:
    # Create a VideoProcessor instance and streamer
  ctx = webrtc_streamer(
        key="sign-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,)

  # Show prediction separately
  if ctx.video_processor:
    st.markdown("### 🔍 Predicted Letter:")
    st.markdown(f"<h1 style='text-align: center; color: green;'>{ctx.video_processor.get_prediction()}</h1>", unsafe_allow_html=True)
