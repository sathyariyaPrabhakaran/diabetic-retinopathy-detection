from __future__ import annotations
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def measured_forward_time(model, loader, device):
    model.eval(); total = 0.0; batches = 0
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter(); _ = model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            total += time.perf_counter() - start; batches += 1
    return total, batches
