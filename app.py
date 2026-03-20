import streamlit as st
from fastai.vision.all import *
from PIL import Image

# --- 🎨 PARTNER TASK: CUSTOMIZE THE THEME ---
st.set_page_config(
    page_title="Cervical Cancer AI Diagnostic System",
    page_icon="🔬",
    layout="centered"
)

# --- 📝 PARTNER TASK: ADD PROJECT BRANDING ---
st.title("🔬 Digital Cytology Assistant")
st.subheader("Final Year Project: Engineer Mapfumo & [Partner Name]")
st.markdown("---")

# 1. Load the Model (The 'Brain')
@st.cache_resource
def load_model():
    # This looks for the file in the same folder
    return load_learner('cervix_levels_model.pkl')

learn = load_model()

# 2. Sidebar - PARTNER TASK: ADD DIAGNOSTIC GUIDELINES
with st.sidebar:
    st.header("Diagnostic Guide")
    st.info("""
    **Level 0:** Normal / Benign
    **Level 1:** Low-grade lesions
    **Level 2:** High-grade lesions
    **Level 3:** Malignant / Cancerous
    """)
    st.warning("Note: This tool is for educational/research purposes only.")

# 3. The Uploader
uploaded_file = st.file_uploader("Upload a microscopic cell image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Sample", use_column_width=True)
    
    # Run AI Prediction
    with st.spinner('Analyzing cellular structure...'):
        pred, idx, probs = learn.predict(img)
        confidence = probs[idx] * 100

    # --- 📊 PARTNER TASK: DESIGN THE RESULT DISPLAY ---
    st.write("### AI Diagnostic Analysis")
    
    # Change color based on severity
    if "Level 3" in pred:
        st.error(f"**RESULT: {pred}**")
    elif "Level 2" in pred:
        st.warning(f"**RESULT: {pred}**")
    else:
        st.success(f"**RESULT: {pred}**")

    st.write(f"**System Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))
