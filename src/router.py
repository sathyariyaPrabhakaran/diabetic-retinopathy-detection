from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score


def routing_features(probabilities, quality=None):
    p = np.clip(np.asarray(probabilities), 1e-8, 1.0)
    entropy = -(p * np.log(p)).sum(axis=1)
    ordered = np.sort(p, axis=1)
    margin = ordered[:, -1] - ordered[:, -2] if p.shape[1] > 1 else ordered[:, -1]
    features = [ordered[:, -1], entropy, margin]
    if quality is not None:
        features.append(np.asarray(quality).reshape(-1))
    return np.column_stack(features)


class LearnedRouter:
    """Cost-aware escalation policy learned on a validation/calibration split."""
    def __init__(self, random_state=42):
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
        self.threshold = 0.5
        self.target_met = False
        self.validation_macro_sensitivity = 0.0
        self.validation_macro_f1 = 0.0
        self.validation_escalation_rate = 0.0

    def fit(self, light_prob, light_pred, expert_pred, y_true, min_sensitivity=0.90, quality=None):
        x = routing_features(light_prob, quality)
        useful = ((light_pred != y_true) & (expert_pred == y_true)).astype(int)
        if np.unique(useful).size < 2:
            self.model = None
            self.threshold = 0.5
            return self

        self.model.fit(x, useful)
        score = self.model.predict_proba(x)[:, 1]
        candidates = []
        for threshold in np.linspace(0.01, 0.99, 99):
            escalate = score >= threshold
            final = np.where(escalate, expert_pred, light_pred)
            sensitivity = float(recall_score(y_true, final, average="macro", zero_division=0))
            # Macro F1 is used as a secondary quality objective while the
            # escalation rate is the primary cost objective when the target is met.
            from sklearn.metrics import f1_score
            macro_f1 = float(f1_score(y_true, final, average="macro", zero_division=0))
            rate = float(escalate.mean())
            candidates.append((float(threshold), sensitivity, macro_f1, rate))

        feasible = [c for c in candidates if c[1] >= min_sensitivity]
        if feasible:
            best = min(feasible, key=lambda c: (c[3], -c[2], -c[1]))
            self.target_met = True
        else:
            # Do not silently disable routing when the requested target is
            # infeasible. Choose the best sensitivity/F1 trade-off instead.
            best = max(candidates, key=lambda c: (c[1], c[2], -c[3]))
            self.target_met = False

        self.threshold, self.validation_macro_sensitivity, self.validation_macro_f1, self.validation_escalation_rate = best
        return self

    def score(self, probabilities, quality=None):
        x = routing_features(probabilities, quality)
        if self.model is None:
            return 1.0 - np.asarray(probabilities).max(axis=1)
        return self.model.predict_proba(x)[:, 1]

    def decide(self, probabilities, quality=None):
        scores = self.score(probabilities, quality)
        return scores >= self.threshold, scores
