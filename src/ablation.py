"""Ablation helpers for routing features.

Run these experiments after training and save results alongside evaluation.json.
"""
from __future__ import annotations
import numpy as np
from .router import LearnedRouter, routing_features


def feature_set(probabilities, mode):
    x = routing_features(probabilities)
    if mode == 'all': return x
    if mode == 'confidence_only': return x[:, :1]
    if mode == 'confidence_entropy': return x[:, :2]
    if mode == 'confidence_margin': return x[:, [0, 2]]
    raise ValueError(mode)


def routing_rate(router, probabilities):
    scores = router.score(probabilities)
    return float((scores >= router.threshold).mean())
