

# Import Libraries
import streamlit as st
import torch
import cv2
from PIL import Image
import numpy as np
import google.generativeai as genai
from gtts import gTTS
import speech_recognition as sr
import os

# Set up Gemini API
genai.configure(api_key="AIzaSyA8x6vxDAOIC82KJz7eMq7NU3wj6GcU93o")  # Replace with your Gemini API key
model = genai.GenerativeModel('gemini-pro')

# Load Pre-trained YOLO Model (for simplicity, we use a pre-trained model)
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Replace with custom YOLO model if available

# Streamlit App
def main():
    st.title("AI-Powered Medical Diagnosis Chatbot 🩺")
    st.write("Upload a medical image or ask a medical question.")

    # Multimodal Input
    input_type = st.radio("Choose input type:", ["Text", "Image"])

    if input_type == "Text":
        # Text Input
        user_input = st.text_input("Ask a medical question:")
        if user_input:
            # Generate response using Gemini API
            response = model.generate_content(user_input)
            st.write("**Chatbot Response:**")
            st.write(response.text)

            # Text-to-Speech
            tts = gTTS(response.text, lang='en')
            tts.save("response.mp3")
            st.audio("response.mp3", format='audio/mp3')

    elif input_type == "Image":
        # Image Input
        uploaded_file = st.file_uploader("Upload a medical image (X-ray, CT scan, MRI, etc.):", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Convert image to OpenCV format
            image_cv = np.array(image)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

            # Perform disease detection using YOLO
            results = model_yolo(image_cv)
            results.render()  # Draw bounding boxes on the image

            # Display detection results
            st.image(results.imgs[0], caption="Detection Results", use_column_width=True)

            # Generate medical insights using Gemini API
            st.write("**Chatbot Insights:**")
            insights = model.generate_content("Provide medical insights for the detected condition.")
            st.write(insights.text)

            # Text-to-Speech
            tts = gTTS(insights.text, lang='en')
            tts.save("insights.mp3")
            st.audio("insights.mp3", format='audio/mp3')

# Run Streamlit App
if __name__ == "__main__":
    main()
