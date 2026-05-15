"""CatBoost training."""

from __future__ import annotations


def build_model(random_state: int = 42):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.045,
        loss_function="MultiClass",
        random_seed=random_state,
        verbose=False,
    )

