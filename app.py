import streamlit as st
from fastai.vision.all import *
import gdown
import os

# 1. Page Config
st.set_page_config(page_title="Cervical AI", page_icon="🔬")

# 2. Simple Configuration
FILE_ID = '1D5CVlxR26-RzAjGQqtSxtJhd-ymw9OpT'
MODEL_PATH = 'model.pkl'

@st.cache_resource
def load_my_model():
    # Download if missing
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading Model..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    
    # Simple Load
    return load_learner(MODEL_PATH, cpu=True)

# 3. Main Interface
st.title("🔬 Cervical Cancer Diagnostic AI")
st.markdown(f"**Lead Engineer:** Theresa Mapfumo")

try:
    learn = load_my_model()
    
    uploaded_file = st.file_uploader("Upload cell image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = PILImage.create(uploaded_file)
        st.image(img, use_container_width=True)
        
        with st.spinner("Analyzing..."):
            pred, idx, probs = learn.predict(img)
            
        st.success(f"Prediction: {pred}")
        st.info(f"Confidence: {probs[idx]*100:.2f}%")

except Exception as e:
    st.error(f"System Error: {e}")
    if st.button("Reset & Redownload"):
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        st.rerun()
