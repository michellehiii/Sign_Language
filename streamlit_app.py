import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import string

# ✅ Mobile-friendly layout
st.set_page_config(page_title="Sign Language Recognition", layout="centered")

st.title("Sign Language Recognition")

# ✅ Load full saved model (.h5 from model.save())
model = tf.keras.models.load_model("m1_91_3.h5")

# ✅ Label set
class_labels = list(string.ascii_uppercase)[:25]  # Adjust if your model uses fewer letters

# 📸 Camera input
img_file_buffer = st.camera_input("Take a picture of your hand")

if img_file_buffer is not None:
    # Convert image to NumPy array
    img = Image.open(img_file_buffer).convert("RGB")
    image = np.array(img)

     ✅ OPTIONAL: MediaPipe mask — RE-ENABLE LATER
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    results = hands.process(image)
    if results.multi_hand_landmarks:
        # (Optional: masking code here...)
        pass

    # Preprocess image for model
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (28, 28))
    reshaped = resized.reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Predict
    pred = model.predict(reshaped)
    predicted_class = np.argmax(pred, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    # Display
    st.image(img, caption="Your Hand", use_column_width=True)
    st.success(f"Predicted Letter: {predicted_label}")
