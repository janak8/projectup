import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Custom styles for a modern gradient background
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(300deg, #00bfff, #ff4c68, #ef8172);
        background-size: 180% 180%;
        animation: gradient-animation 18s ease infinite;
        font-family: 'Roboto', sans-serif;
    }

    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stButton>button {
        background-color: #ff4c68;
        color: white;
        border-radius: 8px;
        padding: 12px 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff8172;
    }

    .stTextInput>div>input, .stFileUploader>div {
        border-radius: 8px;
        padding: 10px;
    }

    .large-header {
        font-size: 60px;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
    }

    .trusted-by {
        text-align: center;
        margin-top: 30px;
        font-size: 1.2rem;
        color: #00796b;
    }

    .trusted-companies img {
        width: 120px;
        margin: 10px;
    }

    footer {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 20px;
        margin-top: 20px;
        border-radius: 8px;
    }

    footer p { margin: 0; color: #333; }
</style>
""", unsafe_allow_html=True)

# Load the trained model
model_path = r"D:\Plant disease prediction\plant-disease-prediction-cnn-deep-leanring-project-main\plant-disease-prediction-cnn-deep-leanring-project-main\app\trained_model\plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

# Load the class indices
class_indices_path = r"D:\Plant disease prediction\plant-disease-prediction-cnn-deep-leanring-project-main\plant-disease-prediction-cnn-deep-leanring-project-main\app\class_indices.json"
class_indices = json.load(open(class_indices_path))

# Group diseases by plant type
plant_classes = {}
for idx, label in class_indices.items():
    plant_name = label.split("___")[0]
    if plant_name not in plant_classes:
        plant_classes[plant_name] = []
    plant_classes[plant_name].append(label.split("___")[1])

# Load lightweight model for plant leaf validation
validation_model = tf.keras.applications.MobileNetV2(weights="imagenet")


# Validate if an image contains a plant/leaf
def validate_plant_leaf(image):
    try:
        img = image.resize((224, 224))  # MobileNet expects 224x224
        img_array = np.expand_dims(np.array(img), axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        predictions = validation_model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)

        for _, label, confidence in decoded_predictions[0]:
            if "plant" in label.lower() or "leaf" in label.lower():
                return True, confidence
        return False, 0.0
    except Exception as e:
        st.error(f"Error during plant leaf validation: {e}")
        return False, 0.0


# Preprocess image for prediction
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array


# App layout
st.markdown('<div class="large-header">Plant Disease Prediction</div>', unsafe_allow_html=True)

# File uploader
uploaded_image = st.file_uploader("Upload an image of a plant leaf...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Step 1: Validate the image as a plant leaf
    is_leaf, confidence = validate_plant_leaf(image)
    if not is_leaf:
        st.error("The uploaded image does not appear to be a plant leaf. Please upload a valid image.")
    else:
        st.success(f"Validated as a plant leaf with confidence: {confidence * 100:.2f}%")

        # Step 2: Predict plant type and associated diseases
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_label = class_indices[str(predicted_class_index)]

        # Extract plant type and disease
        predicted_plant, predicted_disease = predicted_label.split("___")
        st.info(f"Detected Plant Type: **{predicted_plant}**")

        # Ask for user confirmation
        user_confirmation = st.radio(f"Is this image of a {predicted_plant} plant?", ["Yes", "No"])

        if user_confirmation == "Yes":
            st.success(f"The detected disease is: **{predicted_disease}**")
        else:
            # Allow manual plant type selection
            selected_plant_type = st.selectbox(
                "Select the correct plant type:",
                list(plant_classes.keys())
            )

            # Show possible diseases for the selected plant type
            st.info(f"You selected: {selected_plant_type}. The possible diseases are:")
            st.write(", ".join(plant_classes[selected_plant_type]))

# Trusted by section
st.markdown("""
<div class="trusted-by">
    <p>Trusted by farmers, agriculture companies, and colleges</p>
</div>
""", unsafe_allow_html=True)

# Trusted company logos
col1, col2, col3 = st.columns(3)
img_path1 = r"D:\Plant disease prediction\plant-disease-prediction-cnn-deep-leanring-project-main\plant-disease-prediction-cnn-deep-leanring-project-main\img\rmc.png"
img_path2 = r"D:\Plant disease prediction\plant-disease-prediction-cnn-deep-leanring-project-main\plant-disease-prediction-cnn-deep-leanring-project-main\img\dk.png"
img_path3 = r"D:\Plant disease prediction\plant-disease-prediction-cnn-deep-leanring-project-main\plant-disease-prediction-cnn-deep-leanring-project-main\img\dws.png"

with col1:
    st.image(img_path1, width=120)

with col2:
    st.image(img_path2, width=120)

with col3:
    st.image(img_path3, width=120)

# Footer
st.markdown("""
<footer>
    <p>Plant Disease Prediction App &copy; 2024</p>
    <p>Developed by Janak Adhikari</p>
</footer>
""", unsafe_allow_html=True)