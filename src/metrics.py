from __future__ import annotations
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, confusion_matrix


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)

    # Referable DR: mild or worse (grade >= 1). This is a screening-oriented
    # secondary endpoint and is reported separately from the 5-class task.
    referable_true = y_true >= 1
    referable_pred = y_pred >= 1
    referable_sensitivity = float(recall_score(referable_true, referable_pred, zero_division=0))
    severe_true = y_true >= 3
    severe_pred = y_pred >= 3
    severe_sensitivity = float(recall_score(severe_true, severe_pred, zero_division=0))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_sensitivity": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_sensitivity": {
            "0_no_dr": float(recalls[0]),
            "1_mild": float(recalls[1]),
            "2_moderate": float(recalls[2]),
            "3_severe": float(recalls[3]),
            "4_proliferative": float(recalls[4]),
        },
        "per_class_f1": {
            "0_no_dr": float(f1s[0]),
            "1_mild": float(f1s[1]),
            "2_moderate": float(f1s[2]),
            "3_severe": float(f1s[3]),
            "4_proliferative": float(f1s[4]),
        },
        "referable_dr_sensitivity": referable_sensitivity,
        "severe_or_proliferative_sensitivity": severe_sensitivity,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist(),
    }


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def measured_forward_time(model, loader, device):
    model.eval(); total = 0.0; images = 0
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            if device.type == "cuda": torch.cuda.synchronize()
            start = time.perf_counter(); _ = model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            total += time.perf_counter() - start; images += len(x)
    return {"seconds": total, "images": images, "seconds_per_image": total / max(images, 1)}
