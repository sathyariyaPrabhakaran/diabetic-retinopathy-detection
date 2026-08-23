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
from src import router as router_module

# Compatibility for router.joblib files trained before the project was packaged as src.
sys.modules.setdefault("router", router_module)

st.set_page_config(
    page_title="Adaptive DR Screening | Research Prototype",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_DIR = Path("models")
RESULTS = Path("results/evaluation.json")
ROUTER_PATH = MODEL_DIR / "router.joblib"

# -----------------------------
# Visual system
# -----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --ink: #e8eef5;
    --muted: #91a1b5;
    --panel: #101923;
    --panel-2: #141f2c;
    --line: #263545;
    --accent: #38d9c4;
    --accent-2: #6ea8ff;
    --warning: #f5b94c;
    --danger: #ff7188;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 88% 5%, rgba(56,217,196,.09), transparent 24%),
        radial-gradient(circle at 10% 12%, rgba(110,168,255,.07), transparent 22%),
        #081019;
    color: var(--ink);
}

[data-testid="stHeader"] { background: rgba(8,16,25,.72); }
[data-testid="stSidebar"] { background: #0b141e; }

.block-container {
    max-width: 1380px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.brand {
    display:flex;
    align-items:center;
    gap:.75rem;
    margin-bottom:1.2rem;
}
.brand-mark {
    width:42px;
    height:42px;
    border:1px solid rgba(56,217,196,.45);
    border-radius:12px;
    display:flex;
    align-items:center;
    justify-content:center;
    color:#07151a;
    background:linear-gradient(135deg,#38d9c4,#6ea8ff);
    font-family:'Space Grotesk',sans-serif;
    font-weight:700;
}
.brand-name {
    font-family:'Space Grotesk',sans-serif;
    font-weight:700;
    letter-spacing:-.02em;
    font-size:1rem;
}
.brand-sub { color:var(--muted); font-size:.78rem; }

.hero {
    border:1px solid var(--line);
    border-radius:24px;
    padding:2rem 2.1rem;
    background:linear-gradient(135deg,rgba(20,31,44,.96),rgba(12,23,34,.92));
    box-shadow:0 20px 70px rgba(0,0,0,.22);
    margin-bottom:1.1rem;
}
.hero-eyebrow {
    color:var(--accent);
    font-size:.78rem;
    font-weight:700;
    letter-spacing:.12em;
    text-transform:uppercase;
    margin-bottom:.65rem;
}
.hero h1 {
    font-family:'Space Grotesk',sans-serif;
    font-size:clamp(2rem,4vw,3.35rem);
    line-height:1.04;
    letter-spacing:-.045em;
    margin:0 0 .8rem 0;
    color:#f5f8fb;
}
.hero p {
    max-width:840px;
    color:#a9b8c9;
    font-size:1rem;
    line-height:1.65;
    margin:0;
}
.badge {
    display:inline-flex;
    align-items:center;
    gap:.45rem;
    margin-top:1.15rem;
    padding:.42rem .72rem;
    border:1px solid rgba(56,217,196,.28);
    border-radius:999px;
    color:#bfeee8;
    background:rgba(56,217,196,.07);
    font-size:.78rem;
    font-weight:600;
}
.badge-dot { width:7px;height:7px;border-radius:50%;background:var(--accent);box-shadow:0 0 10px var(--accent); }

.section-label {
    color:#7f91a5;
    font-size:.72rem;
    letter-spacing:.12em;
    text-transform:uppercase;
    font-weight:700;
    margin:.4rem 0 .55rem;
}

.metric-card {
    border:1px solid var(--line);
    border-radius:16px;
    background:rgba(16,25,35,.86);
    padding:1rem 1.05rem;
    min-height:104px;
}
.metric-label { color:#8292a6;font-size:.75rem;font-weight:600; }
.metric-value { font-family:'Space Grotesk',sans-serif;font-size:1.65rem;font-weight:700;margin-top:.25rem;color:#f2f6fa; }
.metric-note { color:#66788d;font-size:.7rem;margin-top:.2rem; }

.panel {
    border:1px solid var(--line);
    border-radius:20px;
    background:rgba(16,25,35,.88);
    padding:1.25rem;
}
.panel-title {
    font-family:'Space Grotesk',sans-serif;
    font-weight:700;
    font-size:1.05rem;
    color:#f1f5f8;
}
.panel-sub { color:#77899d;font-size:.8rem;margin-top:.2rem; }

.route-box {
    border-radius:15px;
    padding:1rem 1.05rem;
    margin-top:.8rem;
}
.route-light { background:rgba(56,217,196,.08); border:1px solid rgba(56,217,196,.22); }
.route-expert { background:rgba(245,185,76,.08); border:1px solid rgba(245,185,76,.28); }
.route-title { font-weight:700; color:#eef6f7; }
.route-copy { color:#8fa0b3;font-size:.78rem;margin-top:.25rem;line-height:1.5; }

.upload-zone {
    border:1px dashed #3a4d61;
    border-radius:16px;
    background:rgba(8,16,25,.45);
    padding:.5rem;
}

.pipeline {
    display:flex;
    align-items:stretch;
    gap:.55rem;
    margin:.7rem 0 .2rem;
}
.pipe {
    flex:1;
    border:1px solid var(--line);
    border-radius:13px;
    padding:.85rem;
    background:#0d1721;
}
.pipe-num { color:var(--accent);font-size:.68rem;font-weight:700;letter-spacing:.08em; }
.pipe-name { font-weight:700;margin-top:.25rem;font-size:.82rem; }
.pipe-copy { color:#728398;font-size:.68rem;margin-top:.2rem;line-height:1.4; }
.arrow { display:flex;align-items:center;color:#4c6177;font-size:1rem; }

.prob-row { margin:.65rem 0; }
.prob-head { display:flex;justify-content:space-between;color:#c2cfdb;font-size:.78rem;margin-bottom:.25rem; }
.prob-track { height:7px;background:#1b2835;border-radius:99px;overflow:hidden; }
.prob-fill { height:100%;border-radius:99px;background:linear-gradient(90deg,#38d9c4,#6ea8ff); }

.small-note { color:#718298;font-size:.72rem;line-height:1.55; }
.footer-line { height:1px;background:var(--line);margin:2rem 0 1rem; }

[data-testid="stFileUploaderDropzone"] { background:rgba(8,16,25,.35); border-color:#34485d; }
[data-testid="stMetric"] { background:transparent; }
[data-testid="stTabs"] button { color:#8395a9; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#dce9f2; }
.stButton button { border-radius:10px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    n = len(data["classes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    light = build_lightweight(n, pretrained=False).to(device)
    expert = build_expert(n, pretrained=False).to(device)
    light.load_state_dict(
        torch.load(MODEL_DIR / "lightweight.pt", map_location=device, weights_only=True)
    )
    expert.load_state_dict(
        torch.load(MODEL_DIR / "expert.pt", map_location=device, weights_only=True)
    )
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

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div class="brand">
  <div class="brand-mark">DR</div>
  <div>
    <div class="brand-name">Adaptive DR Screening</div>
    <div class="brand-sub">Cost-aware retinal image analysis · Research prototype</div>
  </div>
</div>

<div class="hero">
  <div class="hero-eyebrow">AI / Medical Imaging Research</div>
  <h1>Cost-Aware Adaptive<br>Diabetic Retinopathy Screening</h1>
  <p>
    A two-stage deep-learning pipeline that uses a lightweight classifier for routine cases
    and selectively escalates uncertain retinal images to a higher-capacity expert model.
    The objective is to balance predictive performance with inference cost on CPU hardware.
  </p>
  <div class="badge"><span class="badge-dot"></span> Research demonstration · Not a clinical diagnostic device</div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Evaluation snapshot
# -----------------------------
st.markdown('<div class="section-label">Evaluation snapshot</div>', unsafe_allow_html=True)
metric_cols = st.columns(5)
metric_data = [
    ("Test accuracy", f"{performance.get('accuracy', 0):.1%}", "held-out test set"),
    ("Macro F1", f"{performance.get('macro_f1', 0):.3f}", "5-class average"),
    ("Expert escalation", f"{router_info.get('escalation_rate', 0):.1%}", "adaptive routing"),
    ("Adaptive runtime", f"{runtime.get('adaptive_estimate', 0):.2f}s", "551 test images"),
    ("Device", str(device).upper(), "local inference"),
]
for col, (label, value, note) in zip(metric_cols, metric_data):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")

# -----------------------------
# Main workspace
# -----------------------------
tab_screen, tab_system, tab_eval = st.tabs(["Screening workspace", "System architecture", "Evaluation"])

with tab_screen:
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Retinal image input</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-sub">Upload a JPG, PNG or WEBP fundus image for research inference.</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        uploaded = st.file_uploader(
            "Choose a retinal fundus image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="small-note">Images are processed locally by this prototype. '
            'No clinical decision should be made from the output.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        x = tfm(image).unsqueeze(0).to(device)

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
        else:
            final = light_probs[0]
            route = "Lightweight model"

        pred = int(final.argmax())
        confidence = float(final[pred])

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Inference result</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">Adaptive routing decision for the uploaded image</div>', unsafe_allow_html=True)
            st.write("")
            img_col, result_col = st.columns([1.1, 1])
            with img_col:
                st.image(image, caption="Input fundus image", use_container_width=True)
            with result_col:
                st.markdown('<div class="section-label">Predicted stage</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-family:Space Grotesk;font-size:1.65rem;font-weight:700;color:#f5f8fb;">{classes[pred]}</div>',
                    unsafe_allow_html=True,
                )
                st.metric("Confidence", f"{confidence:.1%}")
                st.metric("Inference route", route)
            if should_escalate:
                st.markdown(
                    '<div class="route-box route-expert"><div class="route-title">Expert escalation triggered</div>'
                    '<div class="route-copy">The router considered the lightweight prediction uncertain enough to justify the higher-capacity expert model.</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="route-box route-light"><div class="route-title">Lightweight route accepted</div>'
                    '<div class="route-copy">The router accepted the first-stage prediction, avoiding an expert-model pass for this image.</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        prob_left, prob_right = st.columns([1.4, .6], gap="large")
        with prob_left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Class probability distribution</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">Model probability across the five DR severity classes</div>', unsafe_allow_html=True)
            for idx, name in enumerate(classes):
                pct = float(final[idx])
                st.markdown(
                    f'<div class="prob-row"><div class="prob-head"><span>{name}</span><span>{pct:.1%}</span></div>'
                    f'<div class="prob-track"><div class="prob-fill" style="width:{max(0,min(100,pct*100)):.2f}%"></div></div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with prob_right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Routing telemetry</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">Decision variables used by the adaptive layer</div>', unsafe_allow_html=True)
            st.metric("Routing score", f"{routing_score:.4f}")
            st.metric("Calibration threshold", f"{router_info.get('threshold', 'N/A')}")
            st.metric("Test-set escalation", f"{router_info.get('escalation_rate', 0):.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Ready for inference</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-sub">The pipeline is loaded and waiting for an image.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="pipeline">'
                '<div class="pipe"><div class="pipe-num">01</div><div class="pipe-name">Fundus image</div><div class="pipe-copy">224×224 normalized input</div></div>'
                '<div class="arrow">→</div>'
                '<div class="pipe"><div class="pipe-num">02</div><div class="pipe-name">Lightweight model</div><div class="pipe-copy">Fast first-stage inference</div></div>'
                '<div class="arrow">→</div>'
                '<div class="pipe"><div class="pipe-num">03</div><div class="pipe-name">Router</div><div class="pipe-copy">Accept or escalate</div></div>'
                '</div>'
                '<div class="pipeline">'
                '<div class="pipe"><div class="pipe-num">04</div><div class="pipe-name">Expert model</div><div class="pipe-copy">Used only for flagged cases</div></div>'
                '<div class="arrow">→</div>'
                '<div class="pipe"><div class="pipe-num">05</div><div class="pipe-name">5-class output</div><div class="pipe-copy">Severity probability distribution</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

with tab_system:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Two-stage adaptive inference architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-sub">The design separates routine screening from expensive expert inference.</div>', unsafe_allow_html=True)
    st.write("")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="pipe"><div class="pipe-num">STAGE 01</div><div class="pipe-name">Lightweight classifier</div><div class="pipe-copy">1,522,981 parameters · low-latency first pass</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="pipe"><div class="pipe-num">DECISION LAYER</div><div class="pipe-name">Learned adaptive router</div><div class="pipe-copy">Confidence/routing score controls selective escalation</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="pipe"><div class="pipe-num">STAGE 02</div><div class="pipe-name">Expert classifier</div><div class="pipe-copy">4,013,953 parameters · invoked for flagged cases</div></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(
        '<div class="small-note">The current prototype is evaluated on 3,662 labeled images split into 2,562 training, 549 validation and 551 test images. '
        'The five output classes are no DR, mild, moderate, severe and proliferative DR.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with tab_eval:
    e1, e2 = st.columns([1, 1], gap="large")
    with e1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Model comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Held-out test-set metrics from the saved evaluation run.</div>', unsafe_allow_html=True)
        rows = []
        for key, label in [
            ("lightweight_only", "Lightweight only"),
            ("expert_only", "Expert only"),
            ("fixed_confidence_router", "Fixed router"),
            ("learned_adaptive_router", "Learned adaptive router"),
        ]:
            m = data.get("performance", {}).get(key, {})
            rows.append({
                "System": label,
                "Accuracy": f"{m.get('accuracy', 0):.1%}",
                "Macro F1": f"{m.get('macro_f1', 0):.3f}",
                "Macro sensitivity": f"{m.get('macro_sensitivity', 0):.3f}",
            })
        st.table(rows)
        st.markdown('</div>', unsafe_allow_html=True)

    with e2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Compute profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Measured CPU runtime for the 551-image test set.</div>', unsafe_allow_html=True)
        light_t = runtime.get("lightweight_all", {}).get("seconds", 0)
        expert_t = runtime.get("expert_all", {}).get("seconds", 0)
        adaptive_t = runtime.get("adaptive_estimate", 0)
        st.metric("Lightweight-only", f"{light_t:.2f}s")
        st.metric("Expert-only", f"{expert_t:.2f}s")
        st.metric("Adaptive estimate", f"{adaptive_t:.2f}s")
        st.markdown(
            f'<div class="small-note">The expert model is approximately {expert_t / max(light_t, 1e-9):.1f}× slower than the lightweight model in this CPU benchmark. '
            'Selective escalation is therefore the core cost-saving mechanism evaluated by the project.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer-line"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="small-note"><strong>Research prototype.</strong> This system is intended for academic demonstration and model evaluation only. '
    'It is not validated for clinical use and must not be used as a medical diagnosis or as a substitute for qualified clinical evaluation.</div>',
    unsafe_allow_html=True,
)
