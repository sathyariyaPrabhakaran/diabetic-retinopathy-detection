from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data import discover, stratified_split, RetinaDataset
from models import build_lightweight, build_expert
from router import LearnedRouter
from metrics import classification_metrics, parameter_count, measured_forward_time


def seed_everything(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train(model, loader, device, epochs, lr, weight_decay, save_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best = float("inf")
    for epoch in range(epochs):
        model.train(); running = 0.0
        for x, y, _ in tqdm(loader, desc=f"epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y); loss.backward(); optimizer.step()
            running += loss.item() * len(y)
        avg = running / len(loader.dataset)
        print(f"loss={avg:.5f}")
        if avg < best:
            best = avg; torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def predict(model, loader, device):
    model.eval(); probs=[]; ys=[]
    with torch.no_grad():
        for x, y, _ in loader:
            probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy()); ys.append(y.numpy())
    return np.vstack(probs), np.concatenate(ys)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/retina")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--min-sensitivity", type=float, default=Config.min_sensitivity)
    args=parser.parse_args()
    seed_everything(Config.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root=Path(args.data_dir); samples, classes=discover(root)
    train_s, val_s, test_s=stratified_split(samples, Config.seed)
    mk=lambda s,t=False: RetinaDataset(root,s,Config.image_size,t)
    train_loader=DataLoader(mk(train_s,True),batch_size=args.batch_size,shuffle=True)
    val_loader=DataLoader(mk(val_s),batch_size=args.batch_size,shuffle=False)
    test_loader=DataLoader(mk(test_s),batch_size=args.batch_size,shuffle=False)
    model_dir=Path("models"); result_dir=Path("results"); model_dir.mkdir(exist_ok=True); result_dir.mkdir(exist_ok=True)

    light=train(build_lightweight(len(classes)).to(device),train_loader,device,args.epochs,Config.lr,Config.weight_decay,model_dir/"lightweight.pt")
    expert=train(build_expert(len(classes)).to(device),train_loader,device,args.epochs,Config.lr,Config.weight_decay,model_dir/"expert.pt")

    lp_v,y_v=predict(light,val_loader,device); ep_v,_=predict(expert,val_loader,device)
    router=LearnedRouter().fit(lp_v,lp_v.argmax(1),ep_v.argmax(1),y_v,min_sensitivity=args.min_sensitivity)
    lp,y=predict(light,test_loader,device); ep,_=predict(expert,test_loader,device)
    light_pred=lp.argmax(1); expert_pred=ep.argmax(1)
    escalate,scores=router.decide(lp)
    adaptive_pred=np.where(escalate,expert_pred,light_pred)

    light_time,_=measured_forward_time(light,test_loader,device)
    expert_time,_=measured_forward_time(expert,test_loader,device)
    adaptive_time=light_time + expert_time*float(escalate.mean())
    baseline_fixed = lp.max(axis=1) < 0.70
    fixed_pred=np.where(baseline_fixed,expert_pred,light_pred)
    result={
        "classes":classes,"device":str(device),"counts":{"train":len(train_s),"validation":len(val_s),"test":len(test_s)},
        "models":{"lightweight_parameters":parameter_count(light),"expert_parameters":parameter_count(expert)},
        "router":{"threshold":router.threshold,"escalation_rate":float(escalate.mean())},
        "runtime_seconds":{"lightweight_all":light_time,"expert_all":expert_time,"adaptive_estimate":adaptive_time},
        "performance":{"lightweight_only":classification_metrics(y,light_pred),"expert_only":classification_metrics(y,expert_pred),"fixed_confidence_router":classification_metrics(y,fixed_pred),"learned_adaptive_router":classification_metrics(y,adaptive_pred)}
    }
    (result_dir/"evaluation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__ == "__main__": main()
