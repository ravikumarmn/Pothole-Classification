import streamlit as st
from keras.models import load_model
from keras.applications import VGG16
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import io

# Constants for image dimensions
img_width, img_height = 224, 224

def load_and_preprocess_image(img_data):
    img = Image.open(img_data)
    img = img.resize((img_width, img_height))
    img_tensor = np.array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)
    return img_tensor

def extract_features_single_image(img_data):
    conv_base = VGG16(weights='imagenet', include_top=False)
    img_tensor = load_and_preprocess_image(img_data)
    features = conv_base.predict(img_tensor)
    return features

def predict_image(classifier_model, img_data, threshold=0.5):
    features = extract_features_single_image(img_data)
    prediction = classifier_model.predict(features)
    class_label = "Pothole Detected" if prediction > threshold else "No Pothole Detected"
    return class_label

# Load the model
model_path = 'my_model.keras'
classifier_model = load_model(model_path)

# Setting up the Streamlit interface
st.title('Pothole Detection App')
st.write("Upload a road image, and the app will predict whether it contains potholes.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"], key='file_uploader')

if uploaded_file is not None:
    with col2:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")
    with st.spinner('Analyzing the image...'):
        label = predict_image(classifier_model, uploaded_file)
        st.success("Classification completed")

    st.subheader('Prediction Result:')
    st.write(f"**{label}**")
    if label == "Pothole Detected":
        st.markdown('⚠️ **Caution is advised!** This road might be hazardous.')
    else:
        st.markdown('✅ The road surface appears to be in good condition.')
