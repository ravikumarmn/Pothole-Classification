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
    try:
        img = Image.open(img_data)
        img = img.resize((img_width, img_height))
        img_tensor = np.array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = preprocess_input(img_tensor)
        return img_tensor
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None

def extract_features_single_image(img_data):
    try:
        conv_base = VGG16(weights='imagenet', include_top=False)
        img_tensor = load_and_preprocess_image(img_data)
        if img_tensor is not None:
            features = conv_base.predict(img_tensor)
            return features
        else:
            return None
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def predict_image(classifier_model, img_data, threshold=0.5):
    try:
        features = extract_features_single_image(img_data)
        if features is not None:
            prediction = classifier_model.predict(features)
            class_label = "Pothole Detected" if prediction > threshold else "No Pothole Detected"
            return class_label
        else:
            return "Error in prediction"
    except Exception as e:
        st.error(f"Error in prediction process: {e}")
        return "Prediction error"

# Load the model
model_path = 'my_model.keras'
try:
    classifier_model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Setting up the Streamlit interface
st.title('Pothole Detection App')
st.write("Upload a road image, and the app will predict whether it contains potholes.")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png"], accept_multiple_files=True, key='file_uploader')

for uploaded_file in uploaded_files:
    with st.container():  # Create a new container for each image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")
        with st.spinner('Analyzing the image...'):
            label = predict_image(classifier_model, uploaded_file)
        if label not in ["Error in prediction", "Prediction error"]:
            st.success("Classification completed")
            st.subheader('Prediction Result:')
            st.write(f"**{label}**")
            if label == "Pothole Detected":
                st.markdown('⚠️ **Caution is advised!** This road might be hazardous.')
            else:
                st.markdown('✅ The road surface appears to be in good condition.')
        else:
            st.error("Failed to classify the image due to an error.")
