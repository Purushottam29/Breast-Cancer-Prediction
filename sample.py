import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64  # For encoding the image

# Function to convert local image to base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load model
model = tf.keras.models.load_model("model/densenet121_model.h5")

# Class labels
class_labels = ["Benign", "Malignant", "Normal"]

# Set Streamlit page layout
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
bg_image_path = "./background.jpg"  
base64_bg = get_base64_of_image(bg_image_path)

# Apply background image using CSS
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_bg}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
            text-shadow: 2px 2px 5px black;
        }}
        .title {{
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: white !important;
            text-shadow: 2px 2px 5px black;
        }}
        .subheader {{
            text-align: center;
            font-size: 18px;
            color: white;
            text-shadow: 1px 1px 4px black;
        }}
        .prediction-box {{
            background: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 22px;
            font-weight: bold;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.markdown('<h1 class="title">Breast Cancer Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload an ultrasound image to predict breast cancer.</p>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Uploaded Image", use_container_width=True, output_format="auto", channels="RGB")

    # Resize and preprocess image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict button
    if st.button("🔍 Predict", use_container_width=True):
        with st.spinner("Processing... Please wait ⏳"):
            # Get prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            # Display results with a styled prediction box
            st.markdown(f"""
            <div class="prediction-box">
                🎯 Prediction: **{class_labels[predicted_class]}** <br>
                Confidence: **{confidence:.2f}%**
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.progress(int(confidence))
