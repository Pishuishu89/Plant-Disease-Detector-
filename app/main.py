import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown

# ==== Step 1: Download model from Google Drive if not already downloaded ====
file_id = st.secrets["MODEL_FILE_ID"]  # put this in .streamlit/secrets.toml
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "plant_disease_prediction_model.h5"

if not os.path.exists(model_path):
    st.write("📥 Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# ==== Step 2: Load the pre-trained model ====
model = tf.keras.models.load_model(model_path)

# ==== Step 3: Load class indices ====
with open("class_indices.json") as f:
    class_indices = json.load(f)

# ==== Step 4: Helper - Preprocess image ====
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# ==== Step 5: Helper - Predict class ====
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# ==== Step 6: Streamlit UI ====
st.title('🌿 Plant Disease Classifier')

uploaded_image = st.file_uploader("📷 Upload an image of a plant leaf...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image")

    with col2:
        if st.button('🔍 Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'✅ Prediction: **{str(prediction)}**')
