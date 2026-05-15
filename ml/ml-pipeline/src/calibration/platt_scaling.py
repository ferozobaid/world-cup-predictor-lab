"""Multiclass Platt-style probability calibration."""

from __future__ import annotations

import numpy as np


class PlattProbabilityCalibrator:
    def __init__(self, random_state: int = 42):
        from sklearn.linear_model import LogisticRegression

        self.random_state = random_state
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)

    def fit(self, probabilities, y):
        self.model.fit(np.asarray(probabilities), y)
        return self

    def predict_proba(self, probabilities):
        return self.model.predict_proba(np.asarray(probabilities))

