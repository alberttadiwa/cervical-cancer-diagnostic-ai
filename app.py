import streamlit as st
# MUST be the first command
st.set_page_config(page_title="Cervical Cancer AI", page_icon="🔬")

import sys
from types import ModuleType

# --- IPYTHON MOCK (Fixes the Import Error) ---
if 'IPython' not in sys.modules:
    mock_ipython = ModuleType('IPython')
    mock_ipython.display = ModuleType('display')
    mock_ipython.display.display = lambda *args, **kwargs: None
    mock_ipython.display.HTML = lambda *args, **kwargs: None
    mock_ipython.display.Markdown = lambda *args, **kwargs: None
    sys.modules['IPython'] = mock_ipython
    sys.modules['IPython.display'] = mock_ipython.display

from fastai.vision.all import *
from PIL import Image
import os
import gdown

# --- 1. SETTINGS ---
FILE_ID = '1D5CVlxR26-RzAjGQqtSxtJhd-ymw9OpT'
MODEL_FILENAME = 'cervix_levels_model.pkl'

@st.cache_resource
def load_model_from_drive():
    # Only download if the file is completely missing
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading AI Model... Please wait."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            # fuzzy=True is required for the large file redirect shown in your logs
            gdown.download(url, MODEL_FILENAME, quiet=False, fuzzy=True)
    
    # Final check: if the file exists, try to load it
    try:
        # We use cpu=True to ensure it works on Streamlit's servers
        return load_learner(MODEL_FILENAME, cpu=True)
    except Exception as e:
        # If loading fails, the file is likely corrupt; remove it so next refresh retries
        if os.path.exists(MODEL_FILENAME):
            os.remove(MODEL_FILENAME)
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the AI
learn = load_model_from_drive()

# --- 2. FRONTEND DESIGN ---
st.title("🔬 Digital Cytology Diagnostic Assistant")
st.markdown(f"**Lead Engineer:** Theresa Mapfumo")
st.markdown("---")

with st.sidebar:
    st.header("Project Info")
    st.write("This AI classifies cervical cells into 4 levels using a ResNet-34 architecture.")
    st.info("Level 0: Normal\n\nLevel 3: High Risk")
    
    # Manual reset in case of emergency
    if st.button("Clear App Cache"):
        st.cache_resource.clear()
        if os.path.exists(MODEL_FILENAME):
            os.remove(MODEL_FILENAME)
        st.rerun()

# Image Uploader
uploaded_file = st.file_uploader("Upload a microscopic cell image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img_display = Image.open(uploaded_file)
    st.image(img_display, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("AI is analyzing cellular morphology..."):
        # Convert for Fastai
        img_fast = PILImage.create(uploaded_file)
        pred, pred_idx, probs = learn.predict(img_fast)
        confidence = float(probs[pred_idx]) * 100

    st.subheader("Diagnostic Assessment")
    
    # Logic for result display
    res = str(pred)
    if "Level 3" in res:
        st.error(f"RESULT: {res} (Malignant/High Risk)")
    elif "Level 2" in res:
        st.warning(f"RESULT: {res} (Pre-Cancerous/Monitor)")
    elif "Level 1" in res:
        st.info(f"RESULT: {res} (Benign/Metaplastic)")
    else:
        st.success(f"RESULT: {res} (Normal/Negative)")

    st.metric("Model Confidence", f"{confidence:.2f}%")
    st.progress(int(confidence))
