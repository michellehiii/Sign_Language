# Install requirements:
# pip install streamlit streamlit-webrtc opencv-python-headless tensorflow mediapipe pillow

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "true"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import string

st.set_page_config(page_title="Sign Language Recognition")
st.title("🔁 Sign Language Recognition (Live Mirror + Capture)")

# ----- Live Mirror Preview (streamlit-webrtc) -----
st.subheader("📷 Live Mirrored Camera Preview")

class MirrorTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        mirrored = cv2.flip(img, 1)
        return mirrored

webrtc_streamer(
    key="mirror-camera",
    video_processor_factory=MirrorTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ----- Model Definition and Load -----
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
class_labels = list(string.ascii_uppercase)  # Adjust if needed

# ----- Take Photo and Predict -----
st.subheader("📸 Capture and Predict")

img_file_buffer = st.camera_input("Take a picture of your hand")

if img_file_buffer is not None:
    # Convert to mirrored NumPy image
    img = Image.open(img_file_buffer).convert("RGB")
    image = np.asarray(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.flip(image, 1)  # 👈 Mirror the captured image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # MediaPipe hand detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    results = hands.process(image)

    white_background = np.ones_like(image) * 255
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = image.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            hull = cv2.convexHull(np.array(points, dtype=np.int32))
            cv2.fillConvexPoly(mask, hull, 255)

    foreground = cv2.bitwise_and(image, image, mask=mask)
    background_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(white_background, white_background, mask=background_mask)
    final_image = cv2.add(foreground, background)

    # Preprocess for model
    gray = cv2.cvtColor(final_image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (28, 28))
    reshaped = resized.reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Predict
    pred = model.predict(reshaped)
    predicted_class = np.argmax(pred, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    # Show result
    st.image(final_image, caption=f"Processed (mirrored) hand")
    st.success(f"✅ Predicted Sign: **{predicted_label}**")
