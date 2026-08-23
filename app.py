from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

from src.models import build_lightweight, build_expert

# The router was trained from inside `src`, so the joblib pickle records its
# class as `router.LearnedRouter`. Streamlit starts from the repository root,
# where that module name is not normally importable. Register the package
# module under the historical name before unpickling the router.
from src import router as router_module
sys.modules.setdefault("router", router_module)

st.set_page_config(page_title="Adaptive DR Screening", page_icon="🩺", layout="wide")

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
    st.error("Required model artifacts are missing. Run the training/evaluation pipeline first.")
    st.stop()

data, light, expert, router, device = load_models()
classes = data["classes"]
router_info = data.get("router", {})
runtime = data.get("runtime_seconds", {})
performance = data.get("performance", {}).get("learned_adaptive_router", {})

st.title("Cost-Aware Adaptive Diabetic Retinopathy Screening")
st.caption("Research prototype — not a clinical diagnostic device")

with st.sidebar:
    st.header("System status")
    st.metric("Device", str(device).upper())
    st.metric("Expert escalation", f"{router_info.get('escalation_rate', 0):.1%}")
    st.metric("Test accuracy", f"{performance.get('accuracy', 0):.1%}")
    st.metric("Test macro F1", f"{performance.get('macro_f1', 0):.3f}")
    st.metric("Adaptive runtime", f"{runtime.get('adaptive_estimate', 0):.2f}s")
    st.caption("Metrics are from the saved evaluation run.")

st.markdown("### Upload a retinal fundus image")
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

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

    # Cost-aware inference: run the lightweight model first. The expert model
    # is invoked only when the router decides that escalation is required.
    with torch.no_grad():
        light_probs = torch.softmax(light(x), 1).cpu().numpy()

    if router is not None:
        escalate, scores = router.decide(light_probs)
        should_escalate = bool(escalate[0])
        routing_score = float(scores[0])
    else:
        routing_score = float(1.0 - light_probs.max())
        should_escalate = routing_score >= 0.30

    if should_escalate:
        with torch.no_grad():
            final = torch.softmax(expert(x), 1).cpu().numpy()[0]
        route = "Expert model"
        st.warning("Router flagged this image for expert review.")
    else:
        final = light_probs[0]
        route = "Lightweight model"
        st.success("Router accepted the lightweight prediction.")

    pred = int(final.argmax())
    confidence = float(final[pred])

    c2.metric("Predicted stage", classes[pred])
    c2.metric("Confidence", f"{confidence:.1%}")
    c2.metric("Inference route", route)
    st.progress(min(max(confidence, 0.0), 1.0), text="Prediction confidence")

    st.subheader("Class probabilities")
    probs = {classes[i]: float(final[i]) for i in range(len(classes))}
    st.bar_chart(probs)

    with st.expander("Routing details"):
        st.write(f"Routing score: **{routing_score:.4f}**")
        st.write(f"Calibration threshold: **{router_info.get('threshold', 'N/A')}**")
        st.write(f"Expert escalation rate on test set: **{router_info.get('escalation_rate', 0):.2%}**")

st.divider()
st.info("Research demonstration only. A prediction from this system must not be used as a medical diagnosis or as a substitute for qualified clinical evaluation.")
