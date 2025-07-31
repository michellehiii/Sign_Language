import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import string
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only

st.set_page_config(page_title="Sign Language Recognition")

st.title("Sign Language Recognition")

# Load model
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

# Load weights
model.load_weights("m1_91_3.h5")
class_labels = list(string.ascii_uppercase)  # Modify if needed

# 📸 Use camera input instead of file upload
img_file_buffer = st.camera_input("Take a picture of your hand")

if img_file_buffer is not None:
    # Convert to NumPy array
    img = Image.open(img_file_buffer)
    image = np.array(img.convert("RGB"))

    # MediaPipe Hand Detection
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

    # Create final image with white background
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

    # Display result
    # st.image(final_image, caption=f'Prediction: {predicted_label}', channels="RGB")
  
    st.success(f"Predicted: {predicted_label}")
