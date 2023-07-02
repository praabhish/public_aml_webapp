# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 14:01:58 2023

@author: Dell
"""
import streamlit as st
from opencv import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle

# Load the deep learning model
cnn_model = keras.models.load_model("waste_deep.h5")

# Load the ML model
loaded_model = pickle.load(open("waste_ml_model.sav", 'rb'))

# Load the label encoder
# label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))

def process_image(image):
    # Preprocess the image
    img = cv2.resize(image, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Get predictions from deep learning model
    prediction_dl = cnn_model.predict(img)
    prediction_dl = np.argmax(prediction_dl, axis=-1)
    prediction_dl = label_encoder.inverse_transform(prediction_dl)

    # Get predictions from ML model
    features = feature_extractor.predict(img)
    prediction_ml = loaded_model.predict(features)[0]
    prediction_ml = label_encoder.inverse_transform([prediction_ml])

    return prediction_dl, prediction_ml

def main():
    st.title("Waste Classification Web App")

    # Upload and display the image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = plt.imread(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image and get predictions
        prediction_dl, prediction_ml = process_image(image)

        # Display the predictions
        st.subheader("Deep Learning Model Prediction")
        st.write("Prediction: ", prediction_dl)

        st.subheader("Machine Learning Model Prediction")
        st.write("Prediction: ", prediction_ml)

if __name__ == "__main__":
    main()


