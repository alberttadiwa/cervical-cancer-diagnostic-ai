import streamlit as st
# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Cervical Cancer AI", page_icon="🔬")

from fastai.vision.all import *
from PIL import Image
import os
import gdown

# --- 1. SETTINGS & MODEL DOWNLOAD ---
FILE_ID = '1D5CVlxR26-RzAjGQqtSxtJhd-ymw9OpT'
MODEL_FILENAME = 'cervix_levels_model.pkl'

@st.cache_resource
def load_model_from_drive():
    # If the file doesn't exist, or is a 'fake' small file (under 50MB)
    if not os.path.exists(MODEL_FILENAME) or os.path.getsize(MODEL_FILENAME) < 50000000:
        if os.path.exists(MODEL_FILENAME): 
            os.remove(MODEL_FILENAME)
        
        with st.spinner("Downloading AI Model from Google Drive... This may take a minute."):
            # fuzzy=True bypasses the "large file" warning page
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_FILENAME, quiet=False, fuzzy=True)
            
    try:
        return load_learner(MODEL_FILENAME)
    except Exception as e:
        if os.path.exists(MODEL_FILENAME): 
            os.remove(MODEL_FILENAME)
        st.error("Model file corrupted. Please refresh the page to try again.")
        st.stop()

# Load the "Brain"
learn = load_model_from_drive()

# --- 2. FRONTEND DESIGN ---
# Header Section
st.title("🔬 Digital Cytology Diagnostic Assistant")
st.markdown(f"**Lead Engineer:** Theresa Mapfumo")
st.markdown("---")

# Sidebar Instructions
with st.sidebar:
    st.header("How to use")
    st.write("1. Upload a microscopic image of a cell.")
    st.write("2. The AI will analyze the structure.")
    st.write("3. Review the predicted Level (0-3).")
    st.info("Level 0: Normal\n\nLevel 3: High Risk")
    
    # Optional: Maintenance button for your partner
    if st.button("Clear Cache & Re-download"):
        if os.path.exists(MODEL_FILENAME):
            os.remove(MODEL_FILENAME)
            st.rerun()

# Image Uploader
uploaded_file = st.file_uploader("Upload a cell image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Run Prediction
    with st.spinner("Analyzing cellular structure..."):
        pred, pred_idx, probs = learn.predict(img)
        confidence = float(probs[pred_idx]) * 100

    # Display Results
    st.subheader("Diagnostic Result")
    
    # Color-coded alerts based on the result
    if "Level 3" in pred:
        st.error(f"DETECTION: {pred}")
    elif "Level 2" in pred:
        st.warning(f"DETECTION: {pred}")
    else:
        st.success(f"DETECTION: {pred}")

    st.write(f"**Confidence Score:** {confidence:.2f}%")
    st.progress(int(confidence))
