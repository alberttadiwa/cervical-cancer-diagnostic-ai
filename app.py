import streamlit as st
from fastai.vision.all import *
from PIL import Image
import os
import gdown

# --- 1. SETTINGS & MODEL DOWNLOAD ---
# This is the ID Engineer Mapfumo extracted from the Drive link
FILE_ID = '1D5CVlxR26-RzAjGQqtSxtJhd-ymw9OpT'
MODEL_FILENAME = 'cervix_levels_model.pkl'

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading AI Model from Google Drive... Please wait."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_FILENAME, quiet=False)
    return load_learner(MODEL_FILENAME)

# Load the "Brain"
learn = load_model_from_drive()

# --- 2. FRONTEND DESIGN ---
st.set_page_config(page_title="Cervical Cancer AI", page_icon="🔬")

# Header Section
st.title("🔬 Digital Cytology Diagnostic Assistant")
st.markdown(f"**Project Lead:** Engineer Mapfumo")
st.markdown("---")

# Sidebar Instructions
with st.sidebar:
    st.header("How to use")
    st.write("1. Upload a microscopic image of a cell.")
    st.write("2. The AI will analyze the structure.")
    st.write("3. Review the predicted Level (0-3).")
    st.info("Level 0: Normal\n\nLevel 3: High Risk")

# Image Uploader
uploaded_file = st.file_uploader("Upload a cell image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Run Prediction
    with st.spinner("Analyzing..."):
        # Fastai prediction
        pred, pred_idx, probs = learn.predict(img)
        confidence = probs[pred_idx] * 100

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
    
    # Progress bar for visual impact
    st.progress(int(confidence))
