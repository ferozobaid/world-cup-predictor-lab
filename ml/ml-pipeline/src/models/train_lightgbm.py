"""LightGBM training."""

from __future__ import annotations


def build_model(random_state: int = 42):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.045,
        num_leaves=15,
        random_state=random_state,
    )

