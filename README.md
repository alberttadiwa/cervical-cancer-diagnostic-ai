Since we are moving from Streamlit to **Hugging Face + Gradio**, the README needs to reflect the new "Space" environment and the simpler deployment steps. This version is polished to look professional for a final year engineering project at HIT while giving Theresa clear guidance.

---

# 🔬 Cervical Cancer Diagnostic AI
### **Automated Cytopathology Classification using Deep Learning**

## 👩‍💻 Project Team
* **Lead Engineer:** Theresa Mapfumo
* **Institution:** Harare Institute of Technology (HIT)
* **Department:** Electronic Engineering

---

## 📌 Project Overview
This application leverages a **ResNet-34** Convolutional Neural Network (CNN) to classify microscopic cervical cell images into four diagnostic levels. The system is designed to assist pathologists by providing a rapid second opinion on cellular morphology.

### 📊 Model Performance
* **Architecture:** ResNet-34 (Transfer Learning)
* **Dataset:** SIPaKMeD (Superficial, Intermediate, Parabasal, Koilocytotic, and Metaplastic cells)
* **Validation Accuracy:** **97.8%**

### 🧬 Diagnostic Mapping
| Level | Clinical Classification | Cell Type |
| :--- | :--- | :--- |
| **Level 0** | **Normal** | Superficial-Intermediate / Parabasal |
| **Level 1** | **Benign** | Metaplastic |
| **Level 2** | **Low-Grade (LSIL)** | Koilocytotic |
| **Level 3** | **High-Grade (HSIL)** | Dyskeratotic |

---

## 🚀 Deployment Instructions (Hugging Face Spaces)

This project is optimized for **Hugging Face Spaces** using the **Gradio** SDK for a stable, high-performance web interface.

### 1. Repository Requirements
Ensure the following files are uploaded to your Hugging Face Space:
* `app.py`: The Gradio interface logic.
* `requirements.txt`: Contains `fastai`.
* `cervix_levels_model.pkl`: The exported learner file.

### 2. Online Setup
1.  Create a new **Space** on [Hugging Face](https://huggingface.co/new-space).
2.  Select **Gradio** as the SDK.
3.  Upload the files listed above via the "Files and versions" tab.
4.  The app will automatically build and go live at your Space URL.

---

## 🎨 Frontend Creative Brief
**Note to the Lead Engineer:** The AI "Engine" is fully integrated. You have creative control over the Gradio interface to make it look like a professional medical diagnostic tool.

**Recommended Enhancements:**
* **Custom Theming:** Use `theme="soft"` or `"glass"` in `app.py` to match medical software aesthetics.
* **Interpretation Guide:** Add a Markdown section below the interface explaining the clinical significance of each Level (0-3).
* **Example Gallery:** Add a folder of "Test Images" so users can demonstrate the AI's accuracy instantly without needing their own files.
* **Confidence Visualization:** Ensure the `gr.Label` component is showing the top 3 classes to reflect the model's "thinking" process.

---

## 🛠️ Technical Stack
* **Language:** Python 3.11
* **AI Framework:** Fastai, PyTorch
* **Interface:** Gradio
* **Hosting:** Hugging Face Spaces (CPU Basic)
* **License:** MIT

> **⚠️ Disclaimer:** This tool is developed for research and educational purposes as part of a Final Year Engineering Project. It is **not** a replacement for professional medical diagnosis. All results should be verified by a certified pathologist.

---

http://googleusercontent.com/interactive_content_block/0
