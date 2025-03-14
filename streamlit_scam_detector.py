import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title("🚨 Social Media Scam Detection System 🚨")
st.write("Enter an Instagram caption to check if it's a scam!")

# User input
user_input = st.text_area("Enter Caption:", "")

if st.button("Check for Scam"):
    if user_input:
        # Transform input using the vectorizer
        input_vectorized = vectorizer.transform([user_input])
        
        # Predict using the model
        prediction = model.predict(input_vectorized)[0]
        
        # Display result
        if prediction == "Scam":
            st.error("⚠️ This caption is likely a SCAM!")
        else:
            st.success("✅ This caption appears to be SAFE!")
    else:
        st.warning("Please enter a caption to analyze.")

# Run this script with: `streamlit run streamlit_scam_detector.py`
