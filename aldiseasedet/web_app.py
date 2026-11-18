#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit UI – refined desktop‑first design for Apple Leaf Disease Detection.
- Upload an image → preview → *Analyze Leaf* button.
- Two prediction cards displayed side‑by‑side.
- Clear visual hierarchy with custom colours (white, green, dark‑green).
- Agreement indicator shown below the cards.
- No class‑probability comparison.
- All model loading, preprocessing, and inference logic unchanged.
"""

import pathlib, sys, time
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st
import numpy as np

# ----------------------------------------------------------------------
# Paths & imports (project root is the folder containing this script)
# ----------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent  # aldiseasedet folder
CHECKPOINT_CNN = ROOT / "models" / "dl" / "custom_cnn.pt"
CHECKPOINT_MOBILENET = ROOT / "models" / "dl" / "mobilenet_v2.pt"
# Make the src package importable
sys.path.append(str(ROOT))
from src.dl_models.custom_cnn import get_model as get_custom_cnn
from src.dl_models.mobilenet_v2 import get_model as get_mobilenet_v2
from src.dl_dataset import AppleLeafDataset

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=False)
def load_custom_cnn() -> torch.nn.Module:
    ds = AppleLeafDataset(split="test")
    n_classes = len(ds.class_to_idx)
    model = get_custom_cnn(num_classes=n_classes)
    if not CHECKPOINT_CNN.is_file():
        st.error(f"❌ Custom CNN checkpoint missing at `{CHECKPOINT_CNN}`")
        st.stop()
    ckpt = torch.load(CHECKPOINT_CNN, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(get_device()).eval()
    return model

@st.cache_resource(show_spinner=False)
def load_mobilenet() -> torch.nn.Module:
    ds = AppleLeafDataset(split="test")
    n_classes = len(ds.class_to_idx)
    model = get_mobilenet_v2(num_classes=n_classes)
    if not CHECKPOINT_MOBILENET.is_file():
        st.error(f"❌ MobileNetV2 checkpoint missing at `{CHECKPOINT_MOBILENET}`")
        st.stop()
    ckpt = torch.load(CHECKPOINT_MOBILENET, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(get_device()).eval()
    return model

def get_class_names() -> list:
    ds = AppleLeafDataset(split="test")
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    return [idx_to_class[i] for i in range(len(idx_to_class))]

def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    return build_transform()(pil_img).unsqueeze(0)

def run_inference(model: torch.nn.Module, batch: torch.Tensor):
    device = get_device()
    batch = batch.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        logits = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000
    probs = F.softmax(logits, dim=1).cpu().squeeze(0).numpy()
    return elapsed_ms, probs

# ----------------------------------------------------------------------
# UI configuration & styling
# ----------------------------------------------------------------------
st.set_page_config(page_title="Apple Leaf Disease Detection",
                   layout="wide",
                   page_icon="🍎")

# Custom CSS – white background, green accent, high contrast text
st.markdown(
    """
    <style>
        body {background-color: #FFFFFF; color: #26352F; font-family: Arial, sans-serif;}
        .header {background-color: #003366; padding: 2rem; text-align: center;}
        .header h1 {color: #FFFFFF; margin: 0; font-size: 2.8rem;}
        .header p {color: #E0E0E0; margin-top: 0.4rem; font-size: 1.2rem;}
        .section {padding: 1rem; margin-bottom: 1.5rem;}
        .card {background-color: #FFFFFF; border: 1px solid #E3EAE3; border-radius: 8px; padding: 1.2rem; box-shadow: 0 2px 6px rgba(0,0,0,0.08); height: 100%;}
        .card h3 {margin: 0 0 0.5rem 0; color: #354F52; font-size: 1.4rem;}
        .card p {margin: 0.2rem 0; color: #26352F; font-size: 1rem;}
        .card .secondary {color: #64746B; font-size: 0.9rem;}
        .btn {background-color: #52796F; color: #FFFFFF; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; font-size: 1rem;}
        .btn:hover {background-color: #3e5e4b;}
        .agreement {margin-top: 1rem; font-size: 1.1rem; padding: 0.8rem; border-radius: 6px;}
        .agreement.agree {background-color: #e6f4ea; color: #52796F; border: 1px solid #52796F;}
        .agreement.disagree {background-color: #fff4e5; color: #D97706; border: 1px solid #D97706;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class='header'>
        <h1>Apple Leaf Disease Detection</h1>
        <p>Upload a leaf image and obtain predictions from two deep‑learning models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Upload + Analyze section
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader("Select leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"❌ Unable to read the image file: {e}")
        img = None
    else:
        # Show preview
        st.image(img, caption="Uploaded leaf", use_column_width=True)
        # Analyze button
        if st.button("🔎 Analyze Leaf", key="analyze_btn"):
            # Load models (cached)
            custom_model = load_custom_cnn()
            mobilenet_model = load_mobilenet()
            class_names = get_class_names()
            tensor = preprocess_image(img)
            t_cnn, p_cnn = run_inference(custom_model, tensor)
            t_mob, p_mob = run_inference(mobilenet_model, tensor)
            # Store results
            st.session_state.update({
                "results": {
                    "custom": {"time": t_cnn, "probs": p_cnn},
                    "mobilenet": {"time": t_mob, "probs": p_mob},
                    "classes": class_names,
                }
            })

# Reset button (always available)
if st.button("Reset", key="reset_btn"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

# ----------------------------------------------------------------------
# Results section – horizontally placed cards
# ----------------------------------------------------------------------
if "results" in st.session_state:
    res = st.session_state.results
    classes = res["classes"]
    col1, col2 = st.columns([1, 1])
    # Prepare data for both models
    model_data = [
        ("Custom CNN", res["custom"]),
        ("MobileNetV2", res["mobilenet"]),
    ]
    preds = []
    for (model_label, data), col in zip(model_data, [col1, col2]):
        pred_idx = int(data["probs"].argmax())
        pred_class = classes[pred_idx]
        confidence = data["probs"][pred_idx] * 100
        inference_time = data["time"]
        preds.append(pred_class)
        with col:
            st.markdown(
                f"""
                <div class='card'>
                    <h3>{model_label} Prediction</h3>
                    <p><strong>Disease:</strong> {pred_class}</p>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                    <p class='secondary'>Inference time: {inference_time:.1f} ms</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    # Agreement indicator
    agree = preds[0] == preds[1]
    if agree:
        msg = "Both models agree on the predicted disease."
        css_class = "agreement agree"
    else:
        msg = "The models produced different predictions."
        css_class = "agreement dis"  # will fix below
    # Use proper class name
    css_class = "agreement agree" if agree else "agreement disagree"
    st.markdown(f"<div class='{css_class}'>{msg}</div>", unsafe_allow_html=True)
else:
    st.info("Upload an image and click **Analyze Leaf** to see the predictions.")
