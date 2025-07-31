import streamlit as st
import cv2
import numpy as np
# import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import string

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
model.load_weights("m1_91_3.h5")  # ✅ Make sure the model is in the same repo
class_labels = list(string.ascii_uppercase)

# 📸 Use camera input
img_file_buffer = st.camera_input("Take a picture of your hand")

if img_file_buffer is not None:
    # Convert to NumPy array
    img = Image.open(img_file_buffer)
    image = np.array(img.convert("RGB"))
    final_image = image  # ✅ Use unprocessed image directly

    # Preprocess for model
    gray = cv2.cvtColor(final_image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (28, 28))
    reshaped = resized.reshape(1, 28, 28, 1).astype('float32') / 255.0

    # Predict
    pred = model.predict(reshaped)
    predicted_class = np.argmax(pred, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    # Display result
    st.success(f"Predicted: {predicted_label}")
