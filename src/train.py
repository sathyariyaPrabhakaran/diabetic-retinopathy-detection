from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
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


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(samples, n_classes, device):
    counts = np.bincount([y for _, y in samples], minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device), counts.astype(int).tolist()


def train(model, loader, device, epochs, lr, weight_decay, save_path, weights):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        running = 0.0
        correct = 0
        seen = 0
        for x, y, _ in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        avg = running / len(loader.dataset)
        acc = correct / max(seen, 1)
        print(f"loss={avg:.5f} accuracy={acc:.4f}")
        if avg < best:
            best = avg
            torch.save(model.state_dict(), save_path)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    return model


def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
            ys.append(y.numpy())
    return np.vstack(probs), np.concatenate(ys)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate adaptive diabetic-retinopathy screening models")
    parser.add_argument("--data-dir", default="data/retina")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--min-sensitivity", type=float, default=Config.min_sensitivity)
    args = parser.parse_args()

    seed_everything(Config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    root = Path(args.data_dir)
    samples, classes = discover(root)
    train_s, val_s, test_s = stratified_split(samples, Config.seed)
    print(f"Classes: {classes}")
    print(f"Split sizes: train={len(train_s)}, validation={len(val_s)}, test={len(test_s)}")

    mk = lambda s, t=False: RetinaDataset(root, s, Config.image_size, t)
    train_loader = DataLoader(mk(train_s, True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(mk(val_s), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(mk(test_s), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_dir = Path("models")
    result_dir = Path("results")
    model_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)

    weights, train_counts = class_weights(train_s, len(classes), device)
    print(f"Training class counts: {dict(zip(classes, train_counts))}")

    light = train(build_lightweight(len(classes)).to(device), train_loader, device, args.epochs, Config.lr, Config.weight_decay, model_dir / "lightweight.pt", weights)
    expert = train(build_expert(len(classes)).to(device), train_loader, device, args.epochs, Config.lr, Config.weight_decay, model_dir / "expert.pt", weights)

    lp_v, y_v = predict(light, val_loader, device)
    ep_v, _ = predict(expert, val_loader, device)
    router = LearnedRouter().fit(lp_v, lp_v.argmax(1), ep_v.argmax(1), y_v, min_sensitivity=args.min_sensitivity)
    joblib.dump(router, model_dir / "router.joblib")

    lp, y = predict(light, test_loader, device)
    ep, _ = predict(expert, test_loader, device)
    light_pred = lp.argmax(1)
    expert_pred = ep.argmax(1)
    escalate, _ = router.decide(lp)
    adaptive_pred = np.where(escalate, expert_pred, light_pred)

    # measured_forward_time returns a dictionary, not a tuple.
    light_time = measured_forward_time(light, test_loader, device)
    expert_time = measured_forward_time(expert, test_loader, device)
    adaptive_time = light_time["seconds"] + expert_time["seconds"] * float(escalate.mean())

    fixed_escalate = lp.max(axis=1) < 0.70
    fixed_pred = np.where(fixed_escalate, expert_pred, light_pred)

    result = {
        "project": "Cost-Aware Adaptive Diabetic Retinopathy Screening",
        "dataset": "APTOS-derived labeled retinal fundus dataset",
        "classes": classes,
        "device": str(device),
        "counts": {"train": len(train_s), "validation": len(val_s), "test": len(test_s)},
        "class_counts_train": dict(zip(classes, train_counts)),
        "models": {"lightweight_parameters": parameter_count(light), "expert_parameters": parameter_count(expert)},
        "router": {"threshold": router.threshold, "escalation_rate": float(escalate.mean()), "minimum_macro_sensitivity_target": args.min_sensitivity},
        "runtime_seconds": {"lightweight_all": light_time, "expert_all": expert_time, "adaptive_estimate": adaptive_time},
        "performance": {
            "lightweight_only": classification_metrics(y, light_pred),
            "expert_only": classification_metrics(y, expert_pred),
            "fixed_confidence_router": classification_metrics(y, fixed_pred),
            "learned_adaptive_router": classification_metrics(y, adaptive_pred),
        },
    }
    (result_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("\nArtifacts saved:")
    print("  models/lightweight.pt")
    print("  models/expert.pt")
    print("  models/router.joblib")
    print("  results/evaluation.json")


if __name__ == "__main__":
    main()
