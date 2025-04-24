import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown

# === Direct download link to your model on Google Drive ===
file_id = "1eHlFtHhK1oRtYZ_SUIJMosLZCGRQYiur"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "plant_disease_prediction_model.h5"

# Download model if it doesn't exist
if not os.path.exists(model_path):
    st.write("📥 Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Load model
model = tf.keras.models.load_model(model_path)

# Load class labels (MUST BE IN YOUR REPO)
with open("class_indices.json") as f:
    class_indices = json.load(f)

# Image preprocessing
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Prediction
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.title('🍃 Plant Disease Classifier')

uploaded_image = st.file_uploader("📷 Upload a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((150, 150)), caption="Uploaded Image")

    with col2:
        if st.button("🔍 Classify"):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'✅ Prediction: **{prediction}**')
