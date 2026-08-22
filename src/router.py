from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score


def routing_features(probabilities, quality=None):
    p = np.clip(np.asarray(probabilities), 1e-8, 1.0)
    entropy = -(p * np.log(p)).sum(axis=1)
    ordered = np.sort(p, axis=1)
    margin = ordered[:, -1] - ordered[:, -2] if p.shape[1] > 1 else ordered[:, -1]
    max_probability = ordered[:, -1]
    features = [max_probability, entropy, margin]
    if quality is not None:
        features.append(np.asarray(quality).reshape(-1))
    return np.column_stack(features)


class LearnedRouter:
    """Learn which samples benefit from expert escalation.

    The router is trained only on a validation/calibration split. A target of
    one means the lightweight model is wrong while the expert is correct.
    The operating threshold is selected subject to a minimum macro-sensitivity.
    """
    def __init__(self, random_state=42):
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
        self.threshold = 0.5

    def fit(self, light_prob, light_pred, expert_pred, y_true, min_sensitivity=0.90, quality=None):
        x = routing_features(light_prob, quality)
        useful = ((light_pred != y_true) & (expert_pred == y_true)).astype(int)
        # Guard against a degenerate calibration split.
        if np.unique(useful).size < 2:
            self.model = None
            self.threshold = 0.5
            return self
        self.model.fit(x, useful)
        score = self.model.predict_proba(x)[:, 1]
        best_threshold, best_rate = 0.5, 1.0
        for threshold in np.linspace(0.05, 0.95, 91):
            escalate = score >= threshold
            final = np.where(escalate, expert_pred, light_pred)
            sensitivity = recall_score(y_true, final, average="macro", zero_division=0)
            rate = float(escalate.mean())
            if sensitivity >= min_sensitivity and rate < best_rate:
                best_threshold, best_rate = float(threshold), rate
        self.threshold = best_threshold
        return self

    def score(self, probabilities, quality=None):
        x = routing_features(probabilities, quality)
        if self.model is None:
            # Conservative fallback when calibration cannot train a classifier.
            p = np.asarray(probabilities)
            return 1.0 - p.max(axis=1)
        return self.model.predict_proba(x)[:, 1]

    def decide(self, probabilities, quality=None):
        scores = self.score(probabilities, quality)
        return scores >= self.threshold, scores
