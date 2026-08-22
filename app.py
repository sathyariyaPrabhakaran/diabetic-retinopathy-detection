from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from src.models import build_lightweight, build_expert
from src.router import LearnedRouter
from src.explain import routing_explanation

st.set_page_config(page_title='Diabetic Retinopathy | Adaptive AI', page_icon='🩺', layout='wide')
st.title('Cost-Aware Adaptive Diabetic Retinopathy Screening')
st.caption('Research prototype — not a clinical diagnostic device')

MODEL_DIR = Path('models')
RESULTS = Path('results/evaluation.json')

@st.cache_resource
def load_models():
    data = json.loads(RESULTS.read_text())
    n = len(data['classes'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    light = build_lightweight(n).to(device); expert = build_expert(n).to(device)
    light.load_state_dict(torch.load(MODEL_DIR/'lightweight.pt', map_location=device))
    expert.load_state_dict(torch.load(MODEL_DIR/'expert.pt', map_location=device))
    return data, light.eval(), expert.eval(), device

if not RESULTS.exists() or not (MODEL_DIR/'lightweight.pt').exists() or not (MODEL_DIR/'expert.pt').exists():
    st.warning('Train the models first: python src/train.py --data-dir data/retina --epochs 5')
    st.stop()

data, light, expert, device = load_models()
classes = data['classes']; threshold = data['router']['threshold']

uploaded = st.file_uploader('Upload a retinal fundus image', type=['jpg','jpeg','png','webp'])
if uploaded:
    image = Image.open(uploaded).convert('RGB')
    c1,c2 = st.columns(2)
    c1.image(image, caption='Input fundus image', use_container_width=True)
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    x = tfm(image).unsqueeze(0).to(device)
    with torch.no_grad():
        lp = torch.softmax(light(x),1).cpu().numpy(); ep = torch.softmax(expert(x),1).cpu().numpy()
    router = LearnedRouter(); router.threshold = threshold
    # Recreate the trained routing model is not possible from a threshold alone;
    # therefore this interface reports lightweight uncertainty and uses a conservative
    # threshold fallback until router serialization is added in the next training run.
    score = float(1 - lp.max())
    escalate = score >= (1-threshold)
    final = ep[0] if escalate else lp[0]
    pred = int(final.argmax())
    c2.metric('Predicted class', classes[pred])
    c2.metric('Confidence', f'{final[pred]:.1%}')
    c2.metric('Route', 'Expert model' if escalate else 'Lightweight model')
    st.progress(float(final[pred]), text='Prediction confidence')
    st.info(f'Routing uncertainty score: {score:.4f}. This routing result is a research demonstration, not a medical diagnosis.')
