from __future__ import annotations
import numpy as np


def routing_explanation(probabilities, score, threshold, escalated):
    p = np.asarray(probabilities)
    top = int(p.argmax())
    confidence = float(p[top])
    entropy = float(-(np.clip(p,1e-8,1.0) * np.log(np.clip(p,1e-8,1.0))).sum())
    margin = float(np.sort(p)[-1] - np.sort(p)[-2]) if len(p) > 1 else confidence
    return {
        'top_class_index': top,
        'confidence': confidence,
        'entropy': entropy,
        'margin': margin,
        'router_score': float(score),
        'router_threshold': float(threshold),
        'escalated': bool(escalated),
        'reason': 'uncertain case routed to expert model' if escalated else 'lightweight model retained prediction',
    }
