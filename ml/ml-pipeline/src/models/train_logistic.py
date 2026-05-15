"""Logistic regression training."""

from __future__ import annotations


def build_model(random_state: int = 42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, multi_class="auto", random_state=random_state)),
        ]
    )

