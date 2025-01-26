import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image

# Load the model with caching
@st.cache_resource
def loading_model():
    fp = "./model/model.h5"
    model_loader = load_model(fp)
    return model_loader

# Load the model
cnn = loading_model()

# App title and logo (full-page header)
st.image("logo.png", use_container_width=True)  # Use use_container_width instead of use_column_width
st.title("X-Ray Classification [Tuberculosis/Normal]")
st.write("---")  # Add a horizontal line for separation

# File uploader on the main page
st.header("Upload X-Ray Image")
temp = st.file_uploader("Choose an X-Ray image...", type=["jpg", "jpeg", "png"])

# Display uploaded image and make predictions
if temp is not None:
    # Display the uploaded image
    st.header("Uploaded Image")
    image_display = Image.open(temp)
    st.image(image_display, caption="Uploaded X-Ray Image", use_container_width=True)  # Updated parameter

    # Save the uploaded file to a temporary file
    temp_file = NamedTemporaryFile(delete=False)
    temp_file.write(temp.getvalue())

    # Preprocess the image
    img = image.load_img(temp_file.name, target_size=(500, 500), color_mode='grayscale')
    pp_img = image.img_to_array(img)
    pp_img = pp_img / 255
    pp_img = np.expand_dims(pp_img, axis=0)

    # Predict
    st.header("Prediction")
    preds = cnn.predict(pp_img)
    if preds >= 0.5:
        out = ('I am {:.2%} percent confirmed that this is a Tuberculosis case'.format(preds[0][0]))
        st.error(out)  # Use st.error for Tuberculosis case
    else:
        out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(1 - preds[0][0]))
        st.success(out)  # Use st.success for Normal case

else:
    st.warning("Please upload an X-Ray image to get started.")  # Prompt user to upload an image

# Add some additional styling and information
st.write("---")
st.markdown("### How It Works")
st.write("""
1. Upload an X-Ray image using the file uploader below.
2. The app will preprocess the image and use a pre-trained deep learning model to classify it.
3. The result will indicate whether the X-Ray shows signs of Tuberculosis or is Normal.
""")

st.write("---")
st.markdown("### About")
st.write("""
This app is designed to assist in the classification of X-Ray images for Tuberculosis detection. 
It uses a Convolutional Neural Network (CNN) model trained on a dataset of X-Ray images.
""")