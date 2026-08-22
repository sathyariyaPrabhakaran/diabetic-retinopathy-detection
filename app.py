from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

from src.models import build_lightweight, build_expert

st.set_page_config(page_title="Diabetic Retinopathy | Adaptive AI", page_icon="🩺", layout="wide")
st.title("Cost-Aware Adaptive Diabetic Retinopathy Screening")
st.caption("Research prototype — not a clinical diagnostic device")

MODEL_DIR = Path("models")
RESULTS = Path("results/evaluation.json")
ROUTER_PATH = MODEL_DIR / "router.joblib"

@st.cache_resource
def load_models():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    n = len(data["classes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    light = build_lightweight(n, pretrained=False).to(device)
    expert = build_expert(n, pretrained=False).to(device)
    light.load_state_dict(torch.load(MODEL_DIR / "lightweight.pt", map_location=device, weights_only=True))
    expert.load_state_dict(torch.load(MODEL_DIR / "expert.pt", map_location=device, weights_only=True))
    router = joblib.load(ROUTER_PATH) if ROUTER_PATH.exists() else None
    return data, light.eval(), expert.eval(), router, device

required = [RESULTS, MODEL_DIR / "lightweight.pt", MODEL_DIR / "expert.pt"]
if not all(p.exists() for p in required):
    st.warning("Train the models first: python src/train.py --data-dir data/retina --epochs 5")
    st.stop()

data, light, expert, router, device = load_models()
classes = data["classes"]

uploaded = st.file_uploader("Upload a retinal fundus image", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    c1, c2 = st.columns(2)
    c1.image(image, caption="Input fundus image", use_container_width=True)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    x = tfm(image).unsqueeze(0).to(device)
    with torch.no_grad():
        lp = torch.softmax(light(x), 1).cpu().numpy()
        ep = torch.softmax(expert(x), 1).cpu().numpy()

    if router is not None:
        escalate, scores = router.decide(lp)
        route = "Expert model" if bool(escalate[0]) else "Lightweight model"
    else:
        score = float(1 - lp.max())
        escalate = np.array([score >= 0.30])
        scores = np.array([score])
        route = "Expert model" if bool(escalate[0]) else "Lightweight model"

    final = ep[0] if bool(escalate[0]) else lp[0]
    pred = int(final.argmax())
    confidence = float(final[pred])

    c2.metric("Predicted class", classes[pred])
    c2.metric("Confidence", f"{confidence:.1%}")
    c2.metric("Route", route)
    st.progress(confidence, text="Prediction confidence")

    st.subheader("Class probabilities")
    probs = {classes[i]: float(final[i]) for i in range(len(classes))}
    st.bar_chart(probs)
    st.info(
        f"Routing score: {float(scores[0]):.4f}. "
        "This output is a research demonstration and must not be used as a medical diagnosis."
    )
