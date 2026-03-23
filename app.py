import gradio as gr
from fastai.vision.all import *

# 1. Load the model
# Hugging Face will look for this file in the same repository
learn = load_learner('cervix_levels_model.pkl')

# 2. Define the prediction function
def predict_cell(img):
    pred, idx, probs = learn.predict(img)
    labels = ["Level 0", "Level 1", "Level 2", "Level 3"]
    # Returns a dictionary of probabilities for the UI to display
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# 3. Build the Frontend
# Your student can change 'title', 'description', and 'theme' very easily here
interface = gr.Interface(
    fn=predict_cell,
    inputs=gr.Image(type="pill"),
    outputs=gr.Label(num_top_classes=3),
    title="🔬 Cervical Cancer Diagnostic Assistant",
    description="Upload a microscopic cell image for AI-powered classification. Developed by Theresa Mapfumo.",
    theme="soft" 
)

# 4. Launch
interface.launch()
