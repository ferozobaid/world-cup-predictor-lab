"""Random forest training."""

from __future__ import annotations


def build_model(random_state: int = 42):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

