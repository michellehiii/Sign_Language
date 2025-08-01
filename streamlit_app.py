import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "true"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from tensorflow.keras import layers, models
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ----------------------------
# Model Setup (Dummy Example)
# ----------------------------
model = models.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(26, activation='softmax'))  # Predict A-Z

# ----------------------------
# Image Preprocessing Function
# ----------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype('float32') / 255.0
    return normalized.reshape(1, 28, 28, 1)

# ----------------------------
# Live Video Frame Processor
# ----------------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            input_img = preprocess_image(img)
            pred = model.predict(input_img)
            label = chr(np.argmax(pred) + ord('A'))
            cv2.putText(img, f"Pred: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        except:
            pass
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------
# Streamlit Layout
# ----------------------------
st.set_page_config(page_title="Sign Language Recognition", layout="centered")
st.title("Sign Language Recognition")

tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Live Photo", 
    "🖼️ Upload Image", 
    "🎞️ Upload Video", 
    "🎥 Live Video Stream"
])

# ----------------------------
# 📷 Tab 1 – Live Camera (Snapshot)
# ----------------------------
with tab1:
    st.header("Capture Live Photo")
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])

    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if st.button("Predict This Frame"):
                    input_img = preprocess_image(frame)
                    prediction = model.predict(input_img)
                    st.success(f"Prediction: {chr(np.argmax(prediction) + ord('A'))}")
            cap.release()
    else:
        st.warning("Enable 'Start Camera' to begin")

# ----------------------------
# 🖼️ Tab 2 – Upload Image
# ----------------------------
with tab2:
    st.header("Upload an Image")
    uploaded_img = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded_img is not None:
        img = Image.open(uploaded_img)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_np = np.array(img)
        input_img = preprocess_image(img_np)
        prediction = model.predict(input_img)
        st.success(f"Prediction: {chr(np.argmax(prediction) + ord('A'))}")

# ----------------------------
# 🎞️ Tab 3 – Upload Video & Predict
# ----------------------------
with tab3:
    st.header("Upload a Short Video (for Frame Prediction)")
    uploaded_vid = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        cap = cv2.VideoCapture(tfile.name)
        st.info("Extracting every 30th frame for prediction...")

        frame_container = st.empty()
        frame_count = 0
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:
                frame_container.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                input_img = preprocess_image(frame)
                pred = model.predict(input_img)
                results.append(chr(np.argmax(pred) + ord('A')))
            frame_count += 1

        cap.release()
        st.success(f"Predictions from video: {' '.join(results)}")

# ----------------------------
# 🎥 Tab 4 – Live Video Stream
# ----------------------------
with tab4:
    st.header("Live Video Stream (Real-Time Prediction)")
    webrtc_streamer(key="live-video", video_processor_factory=VideoProcessor)
