import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Function to preprocess the image
def preprocess_image(image, target_size=(64, 64)):
    if image.mode != "L":
        image = image.convert("L")
    image = image.resize(target_size)
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=-1)
    return image

# Load your trained model
conv_autoencoder = load_model('your_model.h5')

# Set the threshold for anomaly detection
threshold = 0.18533876256780998  # Adjust this based on your model and requirements

# Streamlit UI
st.title("Image Anomaly Detection")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing...")

    # Preprocess and predict
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    reconstructed = conv_autoencoder.predict(processed_image)
    error = np.mean(np.abs(processed_image - reconstructed))
    is_anomaly = error > threshold

    # Display results
    st.write(f"Reconstruction Error: {error}")
    if is_anomaly:
        st.warning("Anomaly detected in the image.")
    else:
        st.success("No anomaly detected in the image.")
