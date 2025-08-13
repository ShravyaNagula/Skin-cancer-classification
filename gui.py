import streamlit as st
import numpy as np
import cv2
from PIL import Image
from joblib import load

def read_image(image, target_size=(100, 100)):
    """Reads and preprocesses the uploaded image."""
    img = image.convert("L")  # Convert to grayscale
    img = img.resize(target_size)  # Resize
    return np.array(img, dtype=np.uint8)

def extract_features(img):
    """Extracts features by flattening the image."""
    return img.flatten().reshape(1, -1)  # Ensures correct feature shape

def predict_skin_cancer(image, model):
    """Predicts if an image is Benign, Malignant, Healthy Skin, or Not a Skin Image."""
    img = read_image(image)
    features = extract_features(img)
    
    pred_label = int(model.predict(features)[0])  # Convert float to int
    
    labels = ["Benign", "Malignant", "Healthy Skin", "Not a Skin Image"]
    return labels[pred_label]

# Load the trained model
best_model = load("best_model.pkl")

# Streamlit UI
st.title("Skin Cancer Detection System")
st.write("Upload an image to predict whether it's Benign, Malignant, or Healthy.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)  # Resize for display
    
    if st.button("Predict"):
        prediction = predict_skin_cancer(image, best_model)
        st.write("### Prediction:", prediction)
