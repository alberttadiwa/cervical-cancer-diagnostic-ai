import streamlit as st

# 1. Setup Page (Must be first)
st.set_page_config(page_title="Cervical AI", page_icon="🔬", layout="centered")

# 2. Environment Fixes (Prevents common crashes)
import sys
from types import ModuleType
if 'IPython' not in sys.modules:
    sys.modules['IPython'] = ModuleType('IPython')
    sys.modules['IPython'].display = ModuleType('display')

import fastcore.dispatch
if not hasattr(fastcore.dispatch.Resolver, 'dict'):
    fastcore.dispatch.Resolver.dict = property(lambda self: self.__dict__)

from fastai.vision.all import *
import gdown
import os

# 3. Configuration
FILE_ID = '1D5CVlxR26-RzAjGQqtSxtJhd-ymw9OpT'
MODEL_PATH = 'model.pkl'

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI Brain..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    return load_learner(MODEL_PATH, cpu=True)

# 4. Simple UI
st.title("🔬 Cervical Cancer Diagnostic AI")
st.caption("Lead Engineer: Theresa Mapfumo | Hardware: CPU Mode")

try:
    learn = get_model()
    
    uploaded_file = st.file_uploader("Drop a cell image here", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img = PILImage.create(uploaded_file)
        st.image(img, caption="Scanning Image...", use_container_width=True)
        
        with st.spinner("Analyzing..."):
            pred, idx, probs = learn.predict(img)
            conf = float(probs[idx]) * 100

        # Big, Easy-to-Read Result
        st.divider()
        st.subheader(f"Result: {pred}")
        
        if "Level 3" in str(pred) or "Level 2" in str(pred):
            st.error(f"High Confidence: {conf:.1f}% - Action Required")
        else:
            st.success(f"Confidence: {conf:.1f}% - Normal/Low Risk")
            
        st.progress(int(conf))

except Exception as e:
    st.error(f"System encountered an error: {e}")
    if st.button("Repair System"):
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        st.cache_resource.clear()
        st.rerun()
