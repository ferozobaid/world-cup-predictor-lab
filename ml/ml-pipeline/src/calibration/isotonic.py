"""Multiclass isotonic probability calibration."""

from __future__ import annotations

import numpy as np


class IsotonicProbabilityCalibrator:
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression

        self._factory = IsotonicRegression
        self.calibrators = []

    def fit(self, probabilities, y_binary):
        probabilities = np.asarray(probabilities)
        y_binary = np.asarray(y_binary)
        self.calibrators = []
        for idx in range(probabilities.shape[1]):
            calibrator = self._factory(out_of_bounds="clip")
            calibrator.fit(probabilities[:, idx], y_binary[:, idx])
            self.calibrators.append(calibrator)
        return self

    def predict_proba(self, probabilities):
        probabilities = np.asarray(probabilities)
        calibrated = np.column_stack(
            [calibrator.predict(probabilities[:, idx]) for idx, calibrator in enumerate(self.calibrators)]
        )
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return calibrated / row_sums

